"""Color distribution statistics across colorspaces.

Per-channel RGB histogram entropy and smoothness, hue entropy, and a
quantization restore-bias probe, aiming at generators with narrower or
over-regular color statistics than camera output.

Rejected: color statistics separate real photo corpora from each other about
as well as they separate real from AI (fails the real-vs-real control).
"""

import numpy as np


_EPS = 1e-12   # numerical guard for log(0) in entropy

# feature column names
COLOR_FEATURE_NAMES = [
    'c_entropy_R',
    'c_entropy_G',
    'c_entropy_B',
    'c_smooth_R',
    'c_smooth_G',
    'c_smooth_B',
    'c_hue_entropy',
    'c_restore_bias',
]


# Internal helpers

def _channel_histogram(channel_uint8: np.ndarray) -> np.ndarray:
    """Compute a normalised 256-bin histogram for one uint8 channel.

    Parameters
    ----------
    channel_uint8 : np.ndarray, shape (H, W), dtype uint8
    Single channel of a uint8 image (values in 0-255).

    Returns
    -------
    np.ndarray, shape (256,), dtype float64
    Normalised histogram summing to 1.0.
    """
    counts = np.bincount(channel_uint8.ravel(), minlength=256).astype(np.float64)
    return counts / counts.sum()


def _channel_entropy(h: np.ndarray) -> float:
    """Shannon entropy of a normalised histogram in nats."""
    return float(-np.sum(h * np.log(h + _EPS)))


def _channel_smoothness(h: np.ndarray) -> float:
    """Adjacent-bin L2 smoothness of a normalised histogram."""
    return float(np.linalg.norm(np.diff(h)))


def _rgb_to_hue(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to the hue channel in [0, 1)."""
    r_u8 = rgb[..., 0]   # shape (H, W), uint8
    g_u8 = rgb[..., 1]
    b_u8 = rgb[..., 2]

    cmax_u8 = np.maximum(np.maximum(r_u8, g_u8), b_u8)
    cmin_u8 = np.minimum(np.minimum(r_u8, g_u8), b_u8)
    # Use int16 to prevent uint8 underflow in subtraction
    delta_u8 = (cmax_u8.astype(np.int16) - cmin_u8.astype(np.int16)).astype(np.int16)

    chroma_mask = delta_u8 > 0   # chromatic pixels only

    # Float versions for hue arithmetic
    r = r_u8.astype(np.float64) / 255.0
    g = g_u8.astype(np.float64) / 255.0
    b = b_u8.astype(np.float64) / 255.0
    delta = delta_u8.astype(np.float64) / 255.0   # zero for achromatic, safe under masks

    hue = np.zeros(rgb.shape[:2], dtype=np.float64)

    # Mask per dominant channel (integer comparison -> exact)
    mask_r = chroma_mask & (cmax_u8 == r_u8)
    mask_g = chroma_mask & (cmax_u8 == g_u8) & ~mask_r
    mask_b = chroma_mask & (cmax_u8 == b_u8) & ~mask_r & ~mask_g

    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    hue[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0
    hue[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0

    hue = (hue / 6.0) % 1.0   # normalise to [0, 1)
    return hue


def _hue_entropy(rgb: np.ndarray) -> float:
    """Shannon entropy of the 256-bin hue histogram (nats)."""
    hue = _rgb_to_hue(rgb)
    # Bin into 256 bins over [0, 1); clip to avoid any floating rounding to 1.0
    bins = np.floor(hue * 256.0).astype(np.int32).clip(0, 255)
    counts = np.bincount(bins.ravel(), minlength=256).astype(np.float64)
    h = counts / counts.sum()
    return _channel_entropy(h)


def _restore_bias(
    rgb: np.ndarray,
    sigma: float = 8.0 / 255.0,
    n_replicas: int = 16,
    seed: int = 42,
) -> float:
    """Restoration-bias probe energy (CoDA estimator, pure NumPy)."""
    # BT.601 luma in [0, 1]
    r = rgb[..., 0].astype(np.float64) / 255.0
    g = rgb[..., 1].astype(np.float64) / 255.0
    b = rgb[..., 2].astype(np.float64) / 255.0
    luma = 0.299 * r + 0.587 * g + 0.114 * b   # shape (H, W)

    bias_accum = np.zeros_like(luma)

    for rep in range(n_replicas):
        rng = np.random.default_rng(seed + rep)
        noise = rng.standard_normal(luma.shape) * sigma
        noisy = np.clip(luma + noise, 0.0, 1.0)
        y_hat = np.round(noisy * 255.0) / 255.0   # round to 8-bit grid
        bias_accum += y_hat - luma

    b_map = bias_accum / n_replicas   # per-pixel bias estimate B_hat
    return float(np.mean(b_map * b_map))


# Public API

def extract_color_features(rgb_uint8_512: np.ndarray) -> dict:
    """Extract global color-distribution features from a 512x512 RGB image."""
    rgb = rgb_uint8_512   # alias; shape guaranteed by caller

    # Per-channel normalised histograms
    h_r = _channel_histogram(rgb[..., 0])
    h_g = _channel_histogram(rgb[..., 1])
    h_b = _channel_histogram(rgb[..., 2])

    # Entropy
    ent_r = _channel_entropy(h_r)
    ent_g = _channel_entropy(h_g)
    ent_b = _channel_entropy(h_b)

    # Adjacent-bin smoothness
    smo_r = _channel_smoothness(h_r)
    smo_g = _channel_smoothness(h_g)
    smo_b = _channel_smoothness(h_b)

    # Hue entropy (full RGB -> HSV conversion)
    hue_ent = _hue_entropy(rgb)

    # Restoration-bias energy (CoDA estimator, luma channel)
    bias = _restore_bias(rgb)

    return {
        'c_entropy_R'   : ent_r,
        'c_entropy_G'   : ent_g,
        'c_entropy_B'   : ent_b,
        'c_smooth_R'    : smo_r,
        'c_smooth_G'    : smo_g,
        'c_smooth_B'    : smo_b,
        'c_hue_entropy' : hue_ent,
        'c_restore_bias': bias,
    }
