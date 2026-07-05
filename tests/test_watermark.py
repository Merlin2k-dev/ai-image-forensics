"""Watermark panel: geometry, scale handling and threshold plumbing.

Synthetic checks only. A genuine stamp capture lives in the bank, so injecting the
bank's own master at the documented offset must fire at any supported scale, clean
corners must stay silent, and sizes below the matching floor must be skipped (not
checked). Detection sensitivity on real consumer output is validated separately
(see the research log); these tests pin the code paths.
"""
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline.watermark import _implied_scale, _load, analyze  # noqa: E402

RNG = np.random.default_rng(7)


def _clean(H, W):
    """Mildly textured clean background: enough variance to pass the flat-patch
    guards, smooth enough that the Laplacian-residual channel reads the injected
    stamp rather than synthetic noise (photo corners are locally smooth)."""
    base = RNG.normal(120, 5, (H, W, 3))
    ramp = np.linspace(0, 40, W)[None, :, None]
    tilt = np.linspace(0, 25, H)[:, None, None]
    return np.clip(base + ramp + tilt, 0, 255).astype(np.uint8)


def _stamp(arr, m, master, f):
    """Composite a white stamp at the documented offset for scale f, using the
    master as the glyph's alpha shape (the masters store glyph intensity, not a
    luma patch; the Gemini capture spans 0-94)."""
    tpl = np.asarray(Image.fromarray(master, "F").resize(
        (max(1, round(master.shape[1] * f)), max(1, round(master.shape[0] * f))),
        Image.LANCZOS))
    th, tw = tpl.shape
    H, W = arr.shape[:2]
    y = H - round(m["margin_bottom"] * f) - th
    x = W - round(m["margin_right"] * f) - tw
    a = ((tpl - tpl.min()) / (tpl.max() - tpl.min() + 1e-6) * 0.6)[..., None]
    region = arr[y:y + th, x:x + tw].astype(np.float32)
    arr[y:y + th, x:x + tw] = np.clip(
        region * (1 - a) + 255.0 * a, 0, 255).astype(np.uint8)
    return arr


def _bank():
    return {m["name"]: (m, t) for m, t in _load()}


def test_clean_corners_silent():
    for H, W in ((2816, 1536), (1536, 838), (1024, 1024)):
        assert analyze(_clean(H, W))["found"] == []


def test_gemini_fires_across_enabled_scales():
    m, master = _bank()["gemini_v2_diamond"]
    for W, H in ((2816, 1536), (2048, 1117), (1536, 838), (1290, 704)):
        arr = _clean(H, W)
        f = _implied_scale(m, H, W)
        r = analyze(_stamp(arr, m, master, f))
        assert any(hit["mark"] == m["mark"] for hit in r["found"]), (W, H, r)


def test_gemini_below_enable_floor_not_checked():
    """Sub-floor deliveries (template < 44px) are skipped, never coin-flipped:
    even a stamped image reports the mark as not-checked there."""
    m, master = _bank()["gemini_v2_diamond"]
    arr = _clean(559, 1024)
    r = analyze(_stamp(arr, m, master, _implied_scale(m, 559, 1024)))
    assert "gemini_v2_diamond" not in r["checked"]
    assert not any(h["generator"] == "Google Gemini" for h in r["found"])


def test_kling_fires_across_scales():
    m, master = _bank()["kling_wordmark_30"]
    for W, H in ((1760, 2336), (1157, 1536), (772, 1024)):
        arr = _clean(H, W)
        f = _implied_scale(m, H, W)
        r = analyze(_stamp(arr, m, master, f))
        assert any(hit["mark"] == m["mark"] for hit in r["found"]), (W, H, r)


def test_below_floor_skipped_not_checked():
    r = analyze(_clean(349, 640))
    assert r["checked"] == [] and r["found"] == []


def test_implied_scale_native_and_small():
    m, _ = _bank()["gemini_v2_diamond"]
    assert _implied_scale(m, 1536, 2816) == 1.0          # canonical 16:9-class
    assert abs(_implied_scale(m, 559, 1024) - 1024 / 2816) < 1e-9
    assert _implied_scale(m, 4000, 4000) == 1.0          # oversize clamps
    k, _ = _bank()["kling_wordmark_30"]
    assert _implied_scale(k, 2336, 1760) == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} OK")
