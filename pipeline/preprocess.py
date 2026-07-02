"""Standardization substrate: 512 center crop + JPEG q75 4:2:0.

Every image the model sees goes through the same path, regardless of label:

  native RGB (uint8, HxW x3)
    -> reject if min(H, W) < 512 (never upscale)
    -> center crop to 512x512
    -> JPEG encode quality=75, subsampling=2 (4:2:0)
    -> the decoded pixels of those bytes are the feature substrate

content_hash(arr) is a SHA-256 of the raw pixel bytes, used for dedup across
data splits. preprocess() and content_hash() are the training-time utilities,
kept here for reference; inference uses the same crop/encode constants.
"""

import hashlib
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image

TARGET_RESOLUTION = 512
JPEG_QUALITY = 75
JPEG_SUBSAMPLING = 2  # 4:2:0


def preprocess(arr: np.ndarray) -> Tuple[Optional[bytes], str]:
    """Center-crop to 512 then JPEG-harmonize to Q75 4:2:0.

    Parameters
    ----------
    arr : numpy.ndarray (H, W, 3) uint8 - native-resolution RGB.

    Returns
    -------
    (jpeg_bytes, "")            on success - raw bytes of the 512x512 Q75 4:2:0 JPEG.
    (None, reason)              if min side < 512 (never upscaled).
    """
    h, w = arr.shape[:2]
    size = TARGET_RESOLUTION
    if min(h, w) < size:
        return None, f"min_side_lt_{size}"

    cy, cx = h // 2, w // 2
    half = size // 2
    cropped = arr[cy - half: cy - half + size, cx - half: cx - half + size]
    assert cropped.shape == (size, size, 3), (
        f"Center-crop produced {cropped.shape}; expected ({size},{size},3).")

    buf = io.BytesIO()
    Image.fromarray(cropped, mode="RGB").save(
        buf, format="JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    return buf.getvalue(), ""


def content_hash(arr: np.ndarray) -> str:
    """SHA-256 hex digest of the raw uint8 pixel bytes of `arr`.

    Identity-preserving across container formats (hashes decoded pixels, not the
    compressed file). Callers building a dataset can slice [:16] of this digest as a dedup key.
    """
    return hashlib.sha256(arr.tobytes()).hexdigest()
