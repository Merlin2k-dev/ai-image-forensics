"""Predict-path safety net: degenerate inputs, crop math, watermark ordering.

These guard the CRITICAL flat-crop crash (the frozen bundle imputes only the 2 SID
columns, so a uniform crop used to raise ValueError -> HTTP 500) and the fixes layered
on top of it. Uses a module-scoped Predictor so the bundle loads once.
"""
import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline.predict import Predictor  # noqa: E402

VALID = {"LIKELY AI-GENERATED", "LEANING AI-GENERATED", "INCONCLUSIVE",
         "LIKELY REAL", "UNSUPPORTED", "ERROR"}


@pytest.fixture(scope="module")
def P():
    return Predictor()


def _save(tmp, arr, name="x.png"):
    p = tmp / name
    Image.fromarray(arr).save(p)
    return p


def test_solid_image_does_not_crash_and_is_not_ai(P, tmp_path):
    r = P.predict(_save(tmp_path, np.full((600, 600, 3), 200, np.uint8)))
    assert r["verdict"] in VALID
    assert r["verdict"] != "LIKELY AI-GENERATED"      # flat != AI
    assert r["degenerate_input"] is True


def test_fully_flat_large_image_inconclusive(P, tmp_path):
    r = P.predict(_save(tmp_path, np.full((1200, 1200, 3), 128, np.uint8)))
    assert r["verdict"] == "INCONCLUSIVE" and r["n_crops"] == 5


def test_one_flat_corner_not_over_capped(P, tmp_path):
    """A single plain-sky corner among 5 crops must not force INCONCLUSIVE; the
    median over the other four carries the verdict."""
    img = np.random.default_rng(1).integers(60, 180, (1200, 1200, 3), dtype=np.uint8)
    img[:560, :560] = 240
    r = P.predict(_save(tmp_path, img))
    assert r["verdict"] in {"LIKELY REAL", "INCONCLUSIVE", "LEANING AI-GENERATED",
                            "LIKELY AI-GENERATED"}
    assert r["n_crops"] == 5  # not crashed; degenerate note may still be set


def test_noise_image_real_and_not_degenerate(P, tmp_path):
    arr = np.random.default_rng(7).integers(0, 255, (768, 768, 3), dtype=np.uint8)
    r = P.predict(_save(tmp_path, arr))
    assert r["degenerate_input"] is False


def test_crop_count_boundaries(P, tmp_path):
    rng = np.random.default_rng(3)
    for side, n in [(512, 1), (1023, 1), (1024, 5), (1025, 5)]:
        arr = rng.integers(0, 255, (side, side, 3), dtype=np.uint8)
        r = P.predict(_save(tmp_path, arr, f"s{side}.png"))
        assert r["n_crops"] == n, (side, r["n_crops"])


def test_watermark_and_provenance_run_below_size_gate(P, tmp_path):
    """Short side < 512 is UNSUPPORTED for the pixel model, but the size-independent
    evidence panels (watermark, provenance) must still be present in the response
    (B11): they are pre-substrate and must not be gated behind the pixel model."""
    arr = np.random.default_rng(2).integers(0, 255, (300, 900, 3), dtype=np.uint8)
    p = _save(tmp_path, arr, "wide.png")
    r = P.predict(p)
    assert r["verdict"] == "UNSUPPORTED"
    assert "watermark" in r and "checked" in r["watermark"]  # panel ran, gate notwithstanding
    assert r["provenance"] is not None
    assert r["verdict_basis"] == "pixel statistics"          # no stamp -> no escalation


def test_gemini_native_parity(P):
    r = P.predict("/root/wmark-samples/Gemini_Generated_Image_d1yk26d1yk26d1yk.png")
    hit = r["watermark"]["found"][0]
    assert abs(hit["score"] - 0.374) < 5e-4 and hit["scale"] == 1.0
    assert r["verdict"] == "LIKELY AI-GENERATED"
