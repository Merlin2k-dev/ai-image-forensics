"""Image ingestion: one decode path for every entry point.

`predict()` and both serving apps decode uploads here so the normalization is
identical everywhere. Three failure classes the raw `Image.open(p).convert("RGB")`
mishandled, each fixed once:

  - EXIF orientation: phone/mirrorless photos store a sensor buffer plus an
    Orientation tag. Without transposing, the model sees the image rotated and the
    watermark corner prior searches the wrong corner. `exif_transpose` applies it.
  - High bit depth: I;16 / I / F modes are truncated (not scaled) by `convert("RGB")`
    -> near-white garbage. We rescale to 8-bit by the actual value range first.
  - Decompression bombs: a small highly-compressible file can declare a huge canvas.
    PIL only *raises* above 2x MAX_IMAGE_PIXELS and silently decodes fully below that,
    so we hard-check the declared pixel count against MAX_PIXELS before decoding.

`load_rgb` returns an (H, W, 3) uint8 array or raises `IngestError` (a single type
callers convert to a clean 4xx/UNSUPPORTED, never a 500).
"""
import numpy as np
from PIL import Image, ImageOps

# ~8000x8000. Above this we refuse rather than risk a multi-hundred-MB decode.
MAX_PIXELS = 64_000_000
# Keep PIL's own guard consistent and below its warn-and-decode band.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class IngestError(Exception):
    """Any decode/normalization failure. Callers map this to a clean 4xx/UNSUPPORTED."""


_HIGH_DEPTH = {"I", "F", "I;16", "I;16B", "I;16L", "I;16N"}


def _to_8bit_rgb(im):
    """Normalize any PIL mode to an (H, W, 3) uint8 array without truncation.

    High-bit-depth modes are rescaled by their observed range (honest 8-bit
    normalization; the training substrate never held >8-bit sources). Everything
    else goes through the standard RGB conversion, which handles L/P/LA/CMYK/RGBA.
    """
    if im.mode in _HIGH_DEPTH:
        a = np.asarray(im, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        a = np.zeros_like(a) if hi <= lo else (a - lo) * (255.0 / (hi - lo))
        g = a.astype(np.uint8)
        return np.repeat(g[:, :, None], 3, axis=2)
    return np.asarray(im.convert("RGB"), np.uint8)


def load_rgb(path):
    """Decode `path` to an (H, W, 3) uint8 RGB array, EXIF-oriented and bomb-guarded.

    Raises IngestError on unreadable, oversize, or degenerate files.
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
            if w <= 0 or h <= 0:
                raise IngestError(f"degenerate image size {w}x{h}")
            if w * h > MAX_PIXELS:
                raise IngestError(
                    f"image is {w}x{h} ({w * h / 1e6:.0f} MP); the limit is "
                    f"{MAX_PIXELS / 1e6:.0f} MP to avoid excessive memory use")
            im = ImageOps.exif_transpose(im)  # display orientation before anything else
            arr = _to_8bit_rgb(im)
    except IngestError:
        raise
    except (OSError, Image.DecompressionBombError, ValueError, SyntaxError) as e:
        raise IngestError(f"could not read image: {e}") from e
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise IngestError(f"unexpected decoded shape {arr.shape}")
    return arr


def orientation_tag(path):
    """The raw EXIF Orientation value (1 = upright, or None). For invariance audits."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
        return exif.get(274)  # 0x0112 Orientation
    except (OSError, Image.DecompressionBombError, ValueError):
        return None
