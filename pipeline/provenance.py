"""Provenance / metadata evidence channel.

Container-level evidence, kept out of the pixel model and its training (there it
would be a dataset shortcut). At inference on a user upload it is legitimate
forensic evidence, with asymmetric semantics:

  - AI-provenance marks present (C2PA manifest naming a generative tool, IPTC
    DigitalSourceType=trainedAlgorithmicMedia, SD/ComfyUI PNG parameter chunks,
    generator software tags) -> strong AI-positive evidence.
  - A cryptographically valid C2PA capture manifest from a camera -> real-leaning.
  - No metadata at all -> neutral. Metadata is trivially stripped, so absence
    proves nothing; this channel can argue "AI" but its silence never argues "real".
"""
import json
import logging
import re
from dataclasses import dataclass, field

from PIL import Image

logger = logging.getLogger(__name__)

# XMP / IPTC markers (case-insensitive search of the raw XMP packet)
_XMP_AI_SOURCETYPES = (
    "trainedalgorithmicmedia",            # IPTC: pure generative
    "compositewithtrainedalgorithmicmedia",  # IPTC: composite with generative elements
)
_GENERATOR_STRINGS = (
    "midjourney", "dall-e", "dall.e", "dalle", "stable diffusion", "stablediffusion",
    "adobe firefly", "firefly", "flux", "gpt-image", "chatgpt", "openai", "imagen", "gemini",
    "ideogram", "recraft", "leonardo.ai", "novelai", "comfyui", "invokeai", "automatic1111",
    "sd-webui", "dreamstudio", "runway", "kandinsky", "seedream",
)
_PNG_AI_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "dream", "generation_data")


@dataclass
class Evidence:
    source: str          # "c2pa" | "xmp" | "exif" | "png_text"
    polarity: str        # "ai" | "real" | "neutral"
    strength: str        # "conclusive" | "strong" | "moderate"
    detail: str          # plain-English, user-facing

    def as_dict(self):
        return self.__dict__.copy()


@dataclass
class ProvenanceReport:
    evidence: list = field(default_factory=list)
    checked: list = field(default_factory=list)

    @property
    def ai_positive(self):
        return [e for e in self.evidence if e.polarity == "ai"]

    def summary(self):
        if self.ai_positive:
            top = max(self.ai_positive, key=lambda e: ("moderate", "strong", "conclusive").index(e.strength))
            return ("ai", top.strength, top.detail)
        real = [e for e in self.evidence if e.polarity == "real"]
        if real:
            return ("real", real[0].strength, real[0].detail)
        return ("neutral", "none",
                "No provenance metadata found. This is uninformative: metadata is routinely stripped "
                "by social media and editing tools, so its absence is NOT evidence the image is real.")

    def as_dict(self):
        v, s, d = self.summary()
        return {"verdict": v, "strength": s, "detail": d, "checked": self.checked,
                "evidence": [e.as_dict() for e in self.evidence]}


def _check_c2pa(path, rep):
    rep.checked.append("c2pa")
    try:
        from c2pa import Reader
        try:
            with Reader(path) as reader:
                manifest_json = reader.json()
        except Exception as e:  # no manifest or unsupported format
            logger.debug("c2pa: no manifest (%s)", e)
            return
        data = json.loads(manifest_json)
        active = data.get("manifests", {}).get(data.get("active_manifest", ""), {})
        blob = json.dumps(active).lower()
        validation_ok = not data.get("validation_status")  # empty/missing list = no failures
        if any(t in blob for t in _XMP_AI_SOURCETYPES) or "c2pa.created" in blob and any(
                g in blob for g in _GENERATOR_STRINGS):
            rep.evidence.append(Evidence(
                "c2pa", "ai", "conclusive" if validation_ok else "strong",
                "The image carries a C2PA Content Credentials manifest declaring it was created by a "
                "generative-AI tool" + ("" if validation_ok else " (signature could not be fully verified)") + "."))
        elif "digitalcapture" in blob and validation_ok:
            rep.evidence.append(Evidence(
                "c2pa", "real", "strong",
                "The image carries a cryptographically valid C2PA manifest declaring a direct camera "
                "capture. Note: this attests provenance of the file, not that no AI edit ever occurred."))
        elif active:
            rep.evidence.append(Evidence(
                "c2pa", "neutral", "moderate",
                "A C2PA manifest is present but does not clearly declare generative-AI creation."))
    except ImportError:
        logger.debug("c2pa-python not installed; skipping")
        rep.checked.remove("c2pa")


def _extract_xmp(path):
    try:
        with open(path, "rb") as fh:
            head = fh.read(4 * 1024 * 1024)
        m = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", head, re.DOTALL)
        return m.group(0).decode("utf-8", "ignore").lower() if m else ""
    except OSError:
        return ""


def _check_xmp(path, rep):
    rep.checked.append("xmp")
    xmp = _extract_xmp(path)
    if not xmp:
        return
    if any(t in xmp for t in _XMP_AI_SOURCETYPES):
        rep.evidence.append(Evidence(
            "xmp", "ai", "strong",
            "The image's IPTC metadata declares DigitalSourceType = trained algorithmic media - the "
            "standardized industry marker for AI-generated content."))
        return
    hits = [g for g in _GENERATOR_STRINGS if g in xmp]
    if hits:
        rep.evidence.append(Evidence(
            "xmp", "ai", "moderate",
            f"The image's XMP metadata mentions a known AI image generator ({hits[0]})."))


def _check_exif_and_png(path, rep):
    rep.checked += ["exif", "png_text"]
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            software = str(exif.get(305, "")).lower()   # Software
            make = str(exif.get(271, ""))               # Make
            model = str(exif.get(272, ""))              # Model
            png_text = {k.lower(): str(v) for k, v in getattr(im, "text", {}).items()} \
                if im.format == "PNG" else {}
    except OSError:
        return
    hits = [g for g in _GENERATOR_STRINGS if g in software]
    if hits:
        rep.evidence.append(Evidence(
            "exif", "ai", "strong",
            f"The EXIF Software tag names an AI image generator ({software.strip()})."))
    ai_keys = [k for k in png_text if k in _PNG_AI_KEYS]
    if ai_keys:
        blob = " ".join(png_text[k] for k in ai_keys).lower()
        known = [g for g in _GENERATOR_STRINGS if g in blob]
        what = f" ({known[0]})" if known else ""
        rep.evidence.append(Evidence(
            "png_text", "ai", "strong",
            f"The PNG file embeds generation parameters ('{ai_keys[0]}' text chunk{what}) - the "
            "signature left by Stable-Diffusion-family tools."))
    if (make or model) and not hits:
        rep.evidence.append(Evidence(
            "exif", "neutral", "moderate",
            f"Camera EXIF present (Make/Model: {make} {model}). Weak context only - EXIF is easily "
            "copied or forged, so this is not treated as proof of a real photo."))


def analyze(path: str) -> dict:
    """Run all provenance checks on an image file. Returns ProvenanceReport.as_dict()."""
    rep = ProvenanceReport()
    _check_c2pa(path, rep)
    _check_xmp(path, rep)
    _check_exif_and_png(path, rep)
    return rep.as_dict()


if __name__ == "__main__":
    import sys
    print(json.dumps(analyze(sys.argv[1]), indent=2))
