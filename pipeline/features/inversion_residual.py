"""Inversion-residual features.

One analytic "denoising step" (a TV gradient step, and a wavelet shrinkage)
moves an image toward the natural-image manifold. Generated images already sit
close to that manifold and leave a small, near-Gaussian residual; real photos
carry off-manifold sensor noise and leave a larger, structured one. All
operators are fixed and analytic; no learned weights.

Four columns (iv_*): TV curvature energy and kurtosis, wavelet noise floor,
sub-threshold coefficient kurtosis. Input: 512x512 RGB uint8, q75 substrate.
"""

import logging
import math

import numpy as np
from scipy import ndimage as ndi
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_EPS = 1e-10   # division-by-zero guard


# Internal helper: BT.601 luma

def _luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma from uint8 RGB (H,W,3) -> float64 (H,W).

    luma = 0.299*R + 0.587*G + 0.114*B
    """
    R = rgb[:, :, 0].astype(np.float64)
    G = rgb[:, :, 1].astype(np.float64)
    B = rgb[:, :, 2].astype(np.float64)
    return 0.299 * R + 0.587 * G + 0.114 * B


# F1 - TV mean-curvature proxy

def feature_f1_tv_curvature(rgb: np.ndarray) -> dict:
    """
    F1: TV mean-curvature field - two features (energy + kurtosis).

    Features emitted (2):
    iv_tv_curv_energy  - mean(kappa^2);  direct residual magnitude
    iv_tv_curv_kurt    - excess kurtosis of kappa.ravel() (Fisher=True)
    """
    Y   = _luma(rgb)                                           # (512, 512) float64
    Ysm = ndi.gaussian_filter(Y, sigma=1.0)                   # smooth JPEG block noise

    # Gradient via Sobel (then rescale to gradient units)
    gx = ndi.sobel(Ysm, axis=1) / 8.0   # d/dx
    gy = ndi.sobel(Ysm, axis=0) / 8.0   # d/dy

    # Regularised gradient norm (image-adaptive epsilon)
    eps   = 1e-3 * float(np.std(Ysm)) + _EPS
    gnorm = np.sqrt(gx ** 2 + gy ** 2 + eps ** 2)             # (512, 512)

    # Normalised gradient components
    nx = gx / gnorm   # (512, 512)
    ny = gy / gnorm   # (512, 512)

    # Divergence = dnx/dx + dny/dy  via np.gradient (axis 1 = x, axis 0 = y)
    dnx_dx = np.gradient(nx, axis=1)
    dny_dy = np.gradient(ny, axis=0)
    kappa  = dnx_dx + dny_dy                                   # (512, 512)

    k_flat = kappa.ravel()
    curv_energy = float(np.mean(k_flat ** 2))
    curv_kurt   = float(sp_stats.kurtosis(k_flat, fisher=True))

    return {
        'iv_tv_curv_energy': curv_energy,
        'iv_tv_curv_kurt'  : curv_kurt,
    }


# F2 - À-trous wavelet noise-Gaussianity test (scipy-only)

def feature_f2_atrous_noise(rgb: np.ndarray) -> dict:
    """
    F2: À-trous wavelet noise-Gaussianity test - two features (sigma + kurtosis).

    Features emitted (2):
    iv_wav_noise_sigma    - MAD-based noise sigma on finest detail band
    iv_wav_subthresh_kurt - excess kurtosis of pooled sub-threshold detail coefficients
    """
    Y = _luma(rgb)   # (512, 512) float64
    N = Y.size       # 512 * 512 = 262 144

    # 4-level à-trous Gaussian decomposition
    sigmas = [1.0, 2.0, 4.0, 8.0]
    approxs = [Y] + [ndi.gaussian_filter(Y, sigma=s) for s in sigmas]
    details  = [approxs[k] - approxs[k + 1] for k in range(4)]   # D_1..D_4

    finest = details[0]   # D_1

    # Noise floor via MAD on finest detail band
    sigma = float(np.median(np.abs(finest)) / 0.6745)

    # VisuShrink universal threshold
    T = sigma * math.sqrt(2.0 * math.log(N))

    # Pool sub-threshold coefficients across all 4 detail bands
    sub_pool_parts = []
    for d in details:
        flat = d.ravel()
        sub_pool_parts.append(flat[np.abs(flat) < T])
    sub_pool = np.concatenate(sub_pool_parts)

    if len(sub_pool) < 10:
        logger.debug(
            "F2 à-trous: fewer than 10 sub-threshold coefficients (%d) - returning kurt=0.0",
            len(sub_pool),
        )
        subthresh_kurt = 0.0
    else:
        subthresh_kurt = float(sp_stats.kurtosis(sub_pool, fisher=True))

    return {
        'iv_wav_noise_sigma'   : sigma,
        'iv_wav_subthresh_kurt': subthresh_kurt,
    }


# Public API

def extract_inversion_features(rgb_uint8_512: np.ndarray) -> dict:
    """
    Extract all 4 inversion-residual features from one 512x512 RGB image.
    """
    feats: dict = {}
    feats.update(feature_f1_tv_curvature(rgb_uint8_512))
    feats.update(feature_f2_atrous_noise(rgb_uint8_512))
    return feats


# Canonical ordered list of feature column names produced by extract_inversion_features.
INVERSION_FEATURE_NAMES = [
    # F1 - TV mean-curvature field (2)
    'iv_tv_curv_energy',
    'iv_tv_curv_kurt',
    # F2 - À-trous wavelet noise-Gaussianity (2)
    'iv_wav_noise_sigma',
    'iv_wav_subthresh_kurt',
]