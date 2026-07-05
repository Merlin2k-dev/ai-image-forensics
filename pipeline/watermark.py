"""Visible-watermark panel: match known generator stamps at their documented positions.

Some generators stamp a visible mark in a corner of the delivered image (Gemini's
sparkle, Kling's wordmark). The 512 center crop never sees it, so this runs on the
full-resolution array before cropping. Display-only evidence, same class as the
provenance panel: a match is strong (the stamp sits at a deterministic, documented
offset), absence means nothing (paid tiers, APIs and any crop remove it).

Delivered images are routinely uniform downscales of the canonical rendition (mobile
save paths, share sheets, preview saves), which shrink the stamp and its offset
together. The matcher therefore works at the scale IMPLIED BY THE IMAGE DIMENSIONS:
per mark, scale f = min(1, long_side / canonical_long_side) (Gemini's canonical size
depends on aspect class); the template and its documented margins are scaled by f and
matched at delivered resolution. Nothing is searched over scale (f is deterministic),
and f = 1 reproduces the original native-size matching exactly. Below a 24px template
the mark is skipped (not checked): reliable matching has a floor.

Per mark: dual-channel ZNCC (luma + 3x3 Laplacian residual) of a genuine stamp
capture against a small search box around the documented offset only -- no free
corner scan; real corners contain enough sparkle-like blobs that an unconstrained
scan needs thresholds too high to catch low-alpha stamps. A mark is reported only
when min(luma, residual) clears its threshold. Thresholds are per template scale,
calibrated so that no corner window in the clean reference corpora fires
(max negative * 1.15, floor 0.10) at that scale; the validated native-size
thresholds are kept as floors.

Bank: models/watermark_bank.npz (master templates, geometry, per-scale thresholds).
"""
import json
import pathlib

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

BANK = pathlib.Path(__file__).resolve().parents[1] / "models/watermark_bank.npz"
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
TOL = 8  # documented offsets are exact; +-8px absorbs re-encode/resize shifts

_bank = None


def _load():
    global _bank
    if _bank is None:
        z = np.load(BANK, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        _bank = [(m, z[m["array"]].astype(np.float32)) for m in meta]
    return _bank


def _zncc(img, tpl, min_std):
    """Sliding ZNCC, 'valid' mode. Flat patches are invalid (-1): near-zero
    variance makes the ratio unbounded and a flat patch cannot hold a stamp."""
    t0 = tpl - tpl.mean()
    nt = np.sqrt((t0 ** 2).sum()) + 1e-6
    n = tpl.size
    num = fftconvolve(img, t0[::-1, ::-1], mode="valid")
    ones = np.ones_like(tpl)
    s1 = fftconvolve(img, ones, mode="valid")
    s2 = fftconvolve(img ** 2, ones, mode="valid")
    var = np.maximum(s2 - s1 ** 2 / n, 0.0)
    z = np.clip(num / (np.sqrt(var) * nt + 1e-6), -1.0, 1.0)
    z[np.sqrt(var / n) < min_std] = -1.0
    return z


def prior_score(luma, tpl, margin_right, margin_bottom, tol=TOL):
    """Dual-channel score of tpl at its documented bottom-right offset.

    Extracts the prior region plus a 4px halo (the residual of a stamp edge
    depends on pixels just outside it), scores both channels, and returns the
    max over the (2*tol+1)^2 allowed positions. None if the image is too small.
    """
    H, W = luma.shape
    th, tw = tpl.shape
    y, x = H - margin_bottom - th, W - margin_right - tw
    pad = tol + 4
    if y - pad < 0 or x - pad < 0:
        return None
    region = luma[y - pad:min(H, y + th + pad), x - pad:min(W, x + tw + pad)]
    resid = fftconvolve(region, LAP, mode="same")
    tpl_r = fftconvolve(tpl, LAP, mode="same")
    z = np.minimum(_zncc(region, tpl / 255.0, 0.5), _zncc(resid, tpl_r, 0.1))
    c = pad - tol  # box of allowed template top-lefts within the region
    box = z[c:c + 2 * tol + 1, c:c + 2 * tol + 1]
    return float(box.max()) if box.size else None


def _implied_scale(m, H, W):
    """Scale of this delivery relative to the mark's canonical rendition.

    Gemini's canonical long side depends on the aspect class (short-side ratios
    bisect the observed canonical heights; geometry recovered from
    allenk/GeminiWatermarkTool). Uniform downscales preserve the ratio, so the
    same rule covers every delivered size. Oversize images clamp to 1.
    """
    if m["kind"] == "gemini":
        r = min(H, W) / max(H, W)
        canon = 2752 if r >= 566 / 1024 else (2816 if r >= 550 / 1024 else 2848)
    else:
        canon = m["canon"]
    return min(1.0, max(H, W) / canon)


def _resize_template(tpl, f):
    if abs(f - 1.0) < 1e-9:
        return tpl
    h, w = tpl.shape
    im = Image.fromarray(tpl.astype(np.float32), "F")
    return np.asarray(im.resize((max(1, round(w * f)), max(1, round(h * f))),
                                Image.LANCZOS), np.float32)


def _threshold(m, size):
    """Threshold for this template size: an exact rung hit uses its own calibrated
    threshold (the native size lands here); between rungs, the max of the two
    nearest brackets (conservative)."""
    exact = [t for s, t in m["rungs"] if s == size]
    if exact:
        return exact[0]
    rungs = sorted(m["rungs"], key=lambda rt: abs(rt[0] - size))[:2]
    return max(t for _, t in rungs)


def analyze(arr) -> dict:
    """Check one full-resolution RGB uint8 array against the stamp bank.

    Returns {"checked": [names], "found": [{generator, mark, score, threshold,
    corner, scale}]}. Marks whose scaled template would fall below the matching
    floor, or whose position prior does not fit the image, are skipped, never
    guessed.
    """
    luma = (arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            if arr.ndim == 3 else arr.astype(np.float32))
    H, W = luma.shape
    checked, found = [], []
    for m, master in _load():
        f = _implied_scale(m, H, W)
        size = round(m["base"] * f)
        if size < m["min_tpl"]:
            continue
        tpl = _resize_template(master, f)
        s = prior_score(luma, tpl, round(m["margin_right"] * f),
                        round(m["margin_bottom"] * f))
        if s is None:
            continue
        checked.append(m["name"])
        thr = _threshold(m, size)
        if s > thr:
            found.append({"generator": m["generator"], "mark": m["mark"],
                          "score": round(s, 3), "threshold": thr,
                          "corner": "bottom-right", "scale": round(f, 3)})
    return {"checked": checked, "found": found}
