"""Ingestion normalization + decode safety (pipeline/ingest.load_rgb).

Pins the contract that every decode path is EXIF-oriented, bit-depth-normalized,
bomb-guarded, and never raises anything but IngestError.
"""
import io
import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pipeline.ingest import IngestError, MAX_PIXELS, load_rgb, orientation_tag  # noqa: E402


def _write(tmp, im, name="x.png", **save):
    p = tmp / name
    im.save(p, **save)
    return p


def test_exif_orientation_transposed(tmp_path):
    base = np.zeros((400, 600, 3), np.uint8)
    im = Image.fromarray(base)
    exif = im.getexif()
    exif[274] = 6  # rotate 90 deg on display -> 400x600 portrait
    p = tmp_path / "o.jpg"
    im.save(p, exif=exif)
    arr = load_rgb(p)
    assert arr.shape[:2] == (600, 400)


def test_16bit_rescaled_not_clipped(tmp_path):
    g = np.linspace(0, 65535, 256 * 256).reshape(256, 256).astype(np.uint16)
    p = _write(tmp_path, Image.fromarray(g, mode="I;16"))
    arr = load_rgb(p)
    assert arr.min() < 5 and arr.max() > 250 and 100 < arr.mean() < 155


def test_decompression_bomb_rejected(tmp_path):
    side = int((MAX_PIXELS ** 0.5)) + 1000
    p = _write(tmp_path, Image.new("RGB", (side, side), (7, 7, 7)), optimize=True)
    with pytest.raises(IngestError):
        load_rgb(p)


def test_truncated_file_clean_error(tmp_path):
    p = tmp_path / "t.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0garbage-not-an-image")
    with pytest.raises(IngestError):
        load_rgb(p)


@pytest.mark.parametrize("mode", ["L", "P", "RGBA", "LA", "CMYK"])
def test_exotic_modes_become_rgb(tmp_path, mode):
    fmt = "TIFF" if mode == "CMYK" else "PNG"
    p = _write(tmp_path, Image.new(mode, (512, 512)), name=f"m.{fmt.lower()}", format=fmt)
    arr = load_rgb(p)
    assert arr.shape == (512, 512, 3) and arr.dtype == np.uint8


def test_orientation_tag_reader(tmp_path):
    im = Image.fromarray(np.zeros((32, 32, 3), np.uint8))
    exif = im.getexif()
    exif[274] = 8
    p = tmp_path / "t.jpg"
    im.save(p, exif=exif)
    assert orientation_tag(p) == 8
    plain = _write(tmp_path, Image.fromarray(np.zeros((32, 32, 3), np.uint8)))
    assert orientation_tag(plain) in (None, 1)
