"""Unit tests for pipeline/provenance.py."""
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline.provenance import analyze  # noqa: E402


def _img():
    rng = np.random.RandomState(0)
    return Image.fromarray(rng.randint(0, 255, (64, 64, 3), np.uint8))


def test_clean_jpeg_is_neutral(tmp_path):
    p = tmp_path / "clean.jpg"
    _img().save(p, "JPEG")
    r = analyze(str(p))
    assert r["verdict"] == "neutral"
    assert "NOT evidence the image is real" in r["detail"]


def test_png_parameters_chunk_flags_ai(tmp_path):
    from PIL.PngImagePlugin import PngInfo
    meta = PngInfo()
    meta.add_text("parameters", "a cat, Steps: 20, Sampler: Euler a, Model: stable diffusion v1.5")
    p = tmp_path / "sd.png"
    _img().save(p, "PNG", pnginfo=meta)
    r = analyze(str(p))
    assert r["verdict"] == "ai"
    assert any(e["source"] == "png_text" and e["strength"] == "strong" for e in r["evidence"])


def test_exif_software_generator_flags_ai(tmp_path):
    p = tmp_path / "mj.jpg"
    im = _img()
    exif = Image.Exif()
    exif[305] = "Midjourney v7"
    im.save(p, "JPEG", exif=exif)
    r = analyze(str(p))
    assert r["verdict"] == "ai"
    assert any(e["source"] == "exif" for e in r["evidence"])


def test_xmp_digitalsourcetype_flags_ai(tmp_path):
    xmp = (b"<x:xmpmeta xmlns:x='adobe:ns:meta/'>"
           b"<Iptc4xmpExt:DigitalSourceType>"
           b"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
           b"</Iptc4xmpExt:DigitalSourceType></x:xmpmeta>")
    p = tmp_path / "xmp.bin"
    p.write_bytes(b"\xff\xd8\xff\xe1" + xmp + b"\xff\xd9")
    r = analyze(str(p))
    assert r["verdict"] == "ai"
    assert any(e["source"] == "xmp" and e["strength"] == "strong" for e in r["evidence"])


def test_camera_exif_stays_neutral(tmp_path):
    p = tmp_path / "cam.jpg"
    im = _img()
    exif = Image.Exif()
    exif[271] = "Canon"
    exif[272] = "EOS 5D"
    im.save(p, "JPEG", exif=exif)
    r = analyze(str(p))
    assert r["verdict"] == "neutral"  # camera EXIF alone never proves real


def test_absence_of_c2pa_does_not_crash(tmp_path):
    p = tmp_path / "x.jpg"
    _img().save(p, "JPEG")
    r = analyze(str(p))
    assert "evidence" in r and isinstance(r["checked"], list)
