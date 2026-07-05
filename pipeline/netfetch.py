"""SSRF-hardened image fetch for the /analyze-url endpoint.

The original guard resolved the hostname to validate it, then let urllib re-resolve
it to connect. That is a DNS-rebinding window (first lookup public, second lookup
127.0.0.1). `fetch_image` closes it by resolving once, validating the address, and
connecting a socket to that pinned IP (TLS still verifies the cert against the real
hostname via SNI). Redirects are followed manually with the same resolve-validate-pin
on every hop. Only the forensic verdict returns to the caller (blind SSRF), but that
is enough to justify pinning.

Blocked: private, loopback, link-local, reserved, multicast, unspecified, and
RFC-6598 CGNAT (100.64.0.0/10), the last not classified private by `ipaddress`.
"""
import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urlparse, urlunparse

CGNAT = ipaddress.ip_network("100.64.0.0/10")


class SSRFError(Exception):
    """URL rejected or unfetchable. Callers map to a 4xx/502; message is user-safe."""


def _validate_ip(ip_str):
    ip = ipaddress.ip_address(ip_str)
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
            or ip.is_multicast or ip.is_unspecified or ip in CGNAT):
        raise SSRFError("URL points to a non-public address.")
    return ip


def _resolve_public(host):
    """Return one validated (family, ip_str) for host, or raise. Every resolved
    address must be public: a record mixing public + private is refused whole, so a
    rebinding answer cannot slip a private IP through on a later lookup."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, TypeError):
        raise SSRFError("That host could not be resolved.")
    chosen = None
    for info in infos:
        _validate_ip(info[4][0])          # raises on any non-public address
        if chosen is None:
            chosen = (info[0], info[4][0])
    if chosen is None:
        raise SSRFError("That host could not be resolved.")
    return chosen


def _connect_pinned(scheme, hostname, ip, port, timeout):
    """A connection whose socket is pinned to `ip` (no re-resolution). TLS verifies
    the certificate against `hostname` via SNI while the TCP peer stays the pinned IP."""
    raw = socket.create_connection((ip, port), timeout=timeout)
    if scheme == "https":
        conn = http.client.HTTPSConnection(hostname, port, timeout=timeout)
        ctx = ssl.create_default_context()
        conn.sock = ctx.wrap_socket(raw, server_hostname=hostname)
    else:
        conn = http.client.HTTPConnection(hostname, port, timeout=timeout)
        conn.sock = raw
    return conn


def fetch_image(url, max_bytes, timeout=20, max_redirects=4):
    """Fetch up to max_bytes from url, refusing any non-public target at every hop."""
    for _ in range(max_redirects + 1):
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            raise SSRFError("Only http(s) image URLs are supported.")
        if not u.hostname:
            raise SSRFError("Malformed URL.")
        _, ip = _resolve_public(u.hostname)
        port = u.port or (443 if u.scheme == "https" else 80)
        conn = _connect_pinned(u.scheme, u.hostname, ip, port, timeout)
        path = urlunparse(("", "", u.path or "/", u.params, u.query, ""))
        try:
            conn.request("GET", path, headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) image-forensics/1.0",
                "Accept": "image/*,*/*;q=0.5",
            })
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308):
                loc = resp.getheader("Location")
                if not loc:
                    raise SSRFError("Redirect without a location.")
                url = loc if urlparse(loc).scheme else urlunparse(
                    (u.scheme, u.netloc, loc, "", "", ""))
                continue
            if resp.status != 200:
                raise SSRFError(f"The image host returned HTTP {resp.status}.")
            data = resp.read(max_bytes + 1)
        except SSRFError:
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError) as e:
            raise SSRFError(f"The image could not be downloaded: {e}")
        finally:
            conn.close()
        if len(data) > max_bytes:
            raise SSRFError("The linked file is larger than the size limit.")
        if not data:
            raise SSRFError("The URL returned no data.")
        return data
    raise SSRFError("Too many redirects.")
