"""Visible-watermark panel: match known generator stamps at their documented positions.

Some generators stamp a visible mark in a corner of the delivered image (Gemini's
sparkle, Kling's wordmark). The 512 center crop never sees it, so this runs on the
full-resolution array before cropping. Display-only evidence, same class as the
provenance panel: a match is strong (the stamp sits at a deterministic, documented
offset), absence means nothing (paid tiers, APIs and any crop remove it).

Per mark: dual-channel ZNCC (luma + 3x3 Laplacian residual) of a genuine stamp
capture against a small search box around the documented offset only -- no free
corner scan; real corners contain enough sparkle-like blobs that an unconstrained
scan needs thresholds too high to catch low-alpha stamps. A mark is reported only
when min(luma, residual) clears its threshold, calibrated so that no corner window
in the clean reference corpora fires (max negative * 1.15, floor 0.10).

Bank: models/watermark_bank.npz (templates, position rules, thresholds).
"""
import json
import pathlib

import numpy as np
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


def _gemini_small_margin(H, W):
    """V2 small profile: margin inherits the downscale from the canonical
    large source; short-side thresholds bisect the observed canonical heights
    (geometry recovered from allenk/GeminiWatermarkTool)."""
    canon = 2752 if min(H, W) >= 566 else (2816 if min(H, W) >= 550 else 2848)
    return int(round(192.0 * max(H, W) / canon))


def analyze(arr) -> dict:
    """Check one full-resolution RGB uint8 array against the stamp bank.

    Returns {"checked": [names], "found": [{generator, mark, score, threshold,
    corner}]}. Marks whose position prior does not fit the image are skipped,
    never guessed.
    """
    luma = (arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
            if arr.ndim == 3 else arr.astype(np.float32))
    H, W = luma.shape
    checked, found = [], []
    for m, tpl in _load():
        if m["gate"] == "both_gt_1024":
            if not (H > 1024 and W > 1024):
                continue
            mr = mb = m["margin"]
        elif m["gate"] == "both_le_1024":
            if H > 1024 and W > 1024:
                continue
            mr = mb = _gemini_small_margin(H, W)
        else:
            mr, mb = m["margin_right"], m["margin_bottom"]
        s = prior_score(luma, tpl, mr, mb)
        if s is None:
            continue
        checked.append(m["name"])
        if s > m["threshold"]:
            found.append({"generator": m["generator"], "mark": m["mark"],
                          "score": round(s, 3), "threshold": m["threshold"],
                          "corner": "bottom-right"})
    return {"checked": checked, "found": found}
