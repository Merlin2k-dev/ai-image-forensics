"""Fractal dimension and spectral-shape variants.

Scale self-similarity of the log power spectrum: correlation of spectrum
shape across 2x and 4x rescalings and between radial octaves, beyond the
adopted slope features.

Rejected: redundant with the adopted spectrum features; no independent
held-out gain.
"""

import logging

import numpy as np
import numpy.fft as npfft
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

_EPS = 1e-10   # division-by-zero guard

# feature column names
FRACTAL_FEATURE_NAMES = [
    'sf_selfsim_2x',
    'sf_selfsim_4x',
    'sf_selfsim_resid_2x',
    'sf_radial_octave',
]


# Internal helpers

def _luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma from uint8 RGB (H,W,3) -> float64 (H,W).

    Y = 0.299.R + 0.587.G + 0.114.B

    Coefficients follow ITU-R BT.601 and match the substrate JPEG's YCbCr
    4:2:0 internal luma channel.  Output is in [0, 255].
    """
    return (0.299  * rgb[:, :, 0].astype(np.float64)
            + 0.587 * rgb[:, :, 1].astype(np.float64)
            + 0.114 * rgb[:, :, 2].astype(np.float64))


def _hann2d(H: int, W: int) -> np.ndarray:
    """2-D Hann window as outer product of two 1-D Hann vectors."""
    h1d = np.hanning(H).astype(np.float64)
    w1d = np.hanning(W).astype(np.float64)
    return np.outer(h1d, w1d)


def _log_mag_spectrum(signal2d: np.ndarray) -> np.ndarray:
    """Hann-windowed log-magnitude spectrum, fftshifted (DC at centre)."""
    H, W = signal2d.shape
    window = _hann2d(H, W)
    F = npfft.fft2(signal2d * window)
    M = np.log(np.abs(F) + 1.0)
    return npfft.fftshift(M)


def _jpeg_suppress(M_shift: np.ndarray) -> np.ndarray:
    """Zero the on-axis JPEG lines and their +/-1 leakage neighbours."""
    H, W = M_shift.shape
    M_clean = M_shift.copy()
    # Zero fy=0 horizontal axis +/-1 leakage
    M_clean[H // 2 - 1 : H // 2 + 2, :] = 0.0
    # Zero fx=0 vertical axis +/-1 leakage
    M_clean[:, W // 2 - 1 : W // 2 + 2] = 0.0
    return M_clean


def _block_mean(M: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2-D array by non-overlapping block averaging."""
    H, W = M.shape
    H2, W2 = H // factor, W // factor
    return (M[:H2 * factor, :W2 * factor]
            .reshape(H2, factor, W2, factor)
            .mean(axis=(1, 3)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson product-moment correlation of two flat arrays.

    Parameters
    ----------
    a, b : np.ndarray
    Arrays of equal size.  Both are flattened and cast to float64.

    Returns
    -------
    float
    Pearson r in [-1, 1].  Returns 0.0 if either array has near-zero
    variance (degenerate case, e.g., all-zero spectrum after JPEG masking).
    """
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a_c = a - a.mean()
    b_c = b - b.mean()
    std_a = float(np.std(a_c))
    std_b = float(np.std(b_c))
    if std_a < _EPS or std_b < _EPS:
        logger.debug("_pearson: near-zero variance - returning 0.0")
        return 0.0
    return float(np.mean(a_c * b_c) / (std_a * std_b))


# Feature group: scale self-similarity (sf_selfsim_2x, sf_selfsim_4x)

def _selfsim_at_factor(M_clean: np.ndarray, factor: int) -> float:
    """Single-factor scale self-similarity of a log-magnitude spectrum."""
    H, W = M_clean.shape
    M_down = _block_mean(M_clean, factor)          # (H//factor, W//factor)
    dH, dW = M_down.shape
    top  = (H - dH) // 2
    left = (W - dW) // 2
    M_center = M_clean[top : top + dH, left : left + dW]
    return _pearson(M_down, M_center)


def feature_sf_selfsim(M_clean: np.ndarray) -> dict:
    """Scale self-similarity features at downscale factors 2 and 4."""
    return {
        'sf_selfsim_2x': _selfsim_at_factor(M_clean, factor=2),
        'sf_selfsim_4x': _selfsim_at_factor(M_clean, factor=4),
    }


# Feature group: residual-domain self-similarity (sf_selfsim_resid_2x)

def feature_sf_selfsim_resid(luma: np.ndarray) -> dict:
    """Scale self-similarity at 2x on the high-pass residual spectrum."""
    resid = luma - gaussian_filter(luma, sigma=1.5)
    M_resid = _log_mag_spectrum(resid)
    M_clean_resid = _jpeg_suppress(M_resid)
    return {
        'sf_selfsim_resid_2x': _selfsim_at_factor(M_clean_resid, factor=2),
    }


# Feature group: radial-profile octave periodicity (sf_radial_octave)

def _radial_profile(M_clean: np.ndarray, r_max: int) -> np.ndarray:
    """Azimuthally-averaged radial profile of a fftshifted spectrum."""
    H, W = M_clean.shape
    cy, cx = H // 2, W // 2
    ys = (np.arange(H) - cy).astype(np.float64)
    xs = (np.arange(W) - cx).astype(np.float64)
    rr = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)   # (H, W)

    # Clip so all values are addressable in the output array
    r_idx = np.clip(rr.astype(np.int64), 0, r_max).ravel()
    flat_M = M_clean.ravel()

    counts = np.bincount(r_idx, minlength=r_max + 1).astype(np.float64)
    sums   = np.bincount(r_idx, weights=flat_M, minlength=r_max + 1)

    P = np.zeros(r_max + 1, dtype=np.float64)
    nonzero = counts > 0
    P[nonzero] = sums[nonzero] / counts[nonzero]
    return P


def feature_sf_radial_octave(M_clean: np.ndarray,
                              r_lo: int = 5,
                              r_hi: int = 100) -> dict:
    """Octave-lag autocorrelation of the azimuthally-averaged radial profile."""
    # Need P up to r_hi * 2 (= 200 by default); add a small margin
    r_max = r_hi * 2 + 5
    P = _radial_profile(M_clean, r_max=r_max)

    r_vals = np.arange(r_lo, r_hi + 1, dtype=np.int64)   # [5, 6, ..., 100]
    P_r  = P[r_vals]          # shape (96,)
    P_2r = P[r_vals * 2]      # P at {10, 12, ..., 200}, shape (96,)

    return {'sf_radial_octave': _pearson(P_r, P_2r)}


# Public API

def extract_fractal_features(rgb_uint8_512: np.ndarray) -> dict:
    """Extract all 4 spectral-fractal self-similarity features from one image."""
    Y = _luma(rgb_uint8_512)
    M_shift = _log_mag_spectrum(Y)
    M_clean = _jpeg_suppress(M_shift)

    feats: dict = {}
    feats.update(feature_sf_selfsim(M_clean))
    feats.update(feature_sf_selfsim_resid(Y))
    feats.update(feature_sf_radial_octave(M_clean))
    return feats
