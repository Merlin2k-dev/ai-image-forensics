"""Camera-sensor physics features.

Real photos carry acquisition physics that generators do not reproduce:
demosaicing couples high-frequency content across channels, photon noise makes
variance scale with brightness, and sensor noise keeps residuals near-Gaussian.
These features measure the presence of that physics, so they read from the
real side rather than chasing any particular generator.

Nine columns (s_*): cross-channel HF correlation, Nyquist diagonal power,
per-block noise coefficient of variation, noise-level-function slope and
intercept, residual kurtosis and skew. Input: 512x512 RGB uint8, q75 substrate.
"""

import logging
import numpy as np
from scipy import ndimage as ndi
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_EPS = 1e-10  # guard against division by zero / log(0)


# Internal helper: BT.601 luma

def _luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma from uint8 RGB (H,W,3) -> float64 (H,W).

    luma = 0.299*R + 0.587*G + 0.114*B
    Returned as float64 to preserve precision for variance/moment computation.
    """
    R = rgb[:, :, 0].astype(np.float64)
    G = rgb[:, :, 1].astype(np.float64)
    B = rgb[:, :, 2].astype(np.float64)
    return 0.299 * R + 0.587 * G + 0.114 * B


# F1 - Cross-channel high-frequency correlation

def feature_f1_cross_channel_hf_correlation(rgb: np.ndarray) -> dict:
    """
    F1: Cross-channel high-frequency correlation (s_rho).

    Features emitted (1):
    s_rho  - normalised cross-channel HF correlation (cosine similarity)
    """
    R = rgb[:, :, 0].astype(np.float64)
    G = rgb[:, :, 1].astype(np.float64)
    B = rgb[:, :, 2].astype(np.float64)

    hp_R = R - ndi.uniform_filter(R, size=3)
    hp_G = G - ndi.uniform_filter(G, size=3)
    hp_B = B - ndi.uniform_filter(B, size=3)

    delta_rg = np.abs(hp_R - hp_G)
    delta_gb = np.abs(hp_G - hp_B)

    dot   = float(np.dot(delta_rg.ravel(), delta_gb.ravel()))
    norm  = float(np.linalg.norm(delta_rg) * np.linalg.norm(delta_gb))
    s_rho = dot / (norm + _EPS)

    return {'s_rho': s_rho}


# F2 - Luma diagonal-gradient Nyquist power

def feature_f2_nyquist_power(rgb: np.ndarray) -> dict:
    """
    F2: Luma diagonal-gradient Nyquist-frequency power ratio.

    Features emitted (2):
    s_nyq_ratio      - Nyquist power ratio, row direction
    s_nyq_ratio_col  - Nyquist power ratio, column direction
    """
    Y = _luma(rgb)   # (512, 512) float64

    diag_a  = Y[:-1, 1:] - Y[1:, :-1]         # (511, 511)
    diag_b  = Y[:-1, :-1] - Y[1:, 1:]          # (511, 511)
    abs_sum = np.abs(diag_a) + np.abs(diag_b)  # (511, 511)

    # Row direction: average over columns -> length-511 signal
    row_signal  = abs_sum.mean(axis=1)                    # (511,)
    fft_mag_r   = np.abs(np.fft.rfft(row_signal))         # (256,)
    mid_r       = fft_mag_r[20:120].mean()
    s_nyq_ratio = float(fft_mag_r[-1] / (mid_r + _EPS))

    # Column direction: average over rows -> length-511 signal
    col_signal       = abs_sum.mean(axis=0)               # (511,)
    fft_mag_c        = np.abs(np.fft.rfft(col_signal))    # (256,)
    mid_c            = fft_mag_c[20:120].mean()
    s_nyq_ratio_col  = float(fft_mag_c[-1] / (mid_c + _EPS))

    return {
        's_nyq_ratio'    : s_nyq_ratio,
        's_nyq_ratio_col': s_nyq_ratio_col,
    }


# F3 - Coefficient of variation of local noise variance

def feature_f3_noise_cov(rgb: np.ndarray) -> dict:
    """
    F3: Coefficient of variation of local noise variance (spatial stationarity).

    Features emitted (2):
    s_noise_cov    - coefficient of variation of per-block noise variances
    s_noise_maxmed - max block variance / median block variance
    """
    Y        = _luma(rgb)                        # (512, 512) float64
    residual = Y - ndi.median_filter(Y, size=3)  # (512, 512)

    block_size = 16
    n_blocks   = 512 // block_size              # 32 per axis

    bvars = np.empty(n_blocks * n_blocks, dtype=np.float64)
    idx = 0
    for i in range(n_blocks):
        r0 = i * block_size
        for j in range(n_blocks):
            c0       = j * block_size
            block    = residual[r0:r0 + block_size, c0:c0 + block_size]
            bvars[idx] = np.var(block)
            idx += 1

    mean_v   = float(np.mean(bvars))
    std_v    = float(np.std(bvars))
    med_v    = float(np.median(bvars))
    max_v    = float(np.max(bvars))

    return {
        's_noise_cov'   : std_v / (mean_v + _EPS),
        's_noise_maxmed': max_v  / (med_v  + _EPS),
    }


# F4 - NLF shot-noise slope

def feature_f4_nlf_slope(rgb: np.ndarray) -> dict:
    """
    F4: Noise Level Function (NLF) shot-noise slope.

    Features emitted (2):
    s_nlf_slope      - linear NLF slope (variance ~ mean intensity)
    s_nlf_intercept  - linear NLF intercept
    """
    Y  = _luma(rgb)   # (512, 512) float64

    lm   = ndi.uniform_filter(Y,        size=7)   # (512, 512)
    lvar = np.maximum(
        ndi.uniform_filter(Y ** 2, size=7) - lm ** 2,
        0.0
    )                                              # (512, 512)

    # Centred differences on interior pixels -> shape (510, 510)
    gx   = (Y[1:-1, 2:] - Y[1:-1, :-2]) / 2.0
    gy   = (Y[2:, 1:-1] - Y[:-2, 1:-1]) / 2.0
    gmag = np.sqrt(gx ** 2 + gy ** 2)             # (510, 510)

    lm_int   = lm[1:-1, 1:-1]    # (510, 510)
    lvar_int = lvar[1:-1, 1:-1]  # (510, 510)

    # Intensity mask: exclude near-clipped pixels
    mask = (lm_int > 10.0) & (lm_int < 245.0)
    if not mask.any():
        logger.debug("F4 NLF: no pixels pass intensity mask - returning zeros")
        return {'s_nlf_slope': 0.0, 's_nlf_intercept': 0.0}

    gmag_m = gmag[mask]
    lm_m   = lm_int[mask]
    lvar_m = lvar_int[mask]

    # Select flattest 30% by gradient magnitude
    n_sel    = max(int(0.30 * len(gmag_m)), 2)
    flat_idx = np.argpartition(gmag_m, n_sel)[:n_sel]

    coeffs = np.polyfit(lm_m[flat_idx], lvar_m[flat_idx], 1)

    return {
        's_nlf_slope'    : float(coeffs[0]),
        's_nlf_intercept': float(coeffs[1]),
    }


# F5 - Residual higher-order moments

def feature_f5_residual_moments(rgb: np.ndarray) -> dict:
    """
    F5: Higher-order statistical moments of the luma high-pass residual.

    Features emitted (2):
    s_resid_kurtosis  - excess kurtosis of luma HP residual (Fisher=True)
    s_resid_skew      - skewness of luma HP residual
    """
    Y        = _luma(rgb)
    residual = Y - ndi.uniform_filter(Y, size=3)
    r_flat   = residual.ravel()

    kurt = float(sp_stats.kurtosis(r_flat, fisher=True))
    skew = float(sp_stats.skew(r_flat))

    return {
        's_resid_kurtosis': kurt,
        's_resid_skew'    : skew,
    }


# Public API

def extract_sensor_features(rgb_uint8_512: np.ndarray) -> dict:
    """
    Extract all 9 camera-sensor-absence features from one image.
    """
    feats: dict = {}
    feats.update(feature_f1_cross_channel_hf_correlation(rgb_uint8_512))
    feats.update(feature_f2_nyquist_power(rgb_uint8_512))
    feats.update(feature_f3_noise_cov(rgb_uint8_512))
    feats.update(feature_f4_nlf_slope(rgb_uint8_512))
    feats.update(feature_f5_residual_moments(rgb_uint8_512))
    return feats


# Canonical ordered list of feature column names produced by extract_sensor_features.
SENSOR_FEATURE_NAMES = [
    # F1 - Cross-channel HF correlation (1)
    's_rho',
    # F2 - Diagonal-gradient Nyquist power (2)
    's_nyq_ratio',
    's_nyq_ratio_col',
    # F3 - Noise spatial stationarity (2)
    's_noise_cov',
    's_noise_maxmed',
    # F4 - NLF shot-noise slope (2)
    's_nlf_slope',
    's_nlf_intercept',
    # F5 - Residual higher-order moments (2)
    's_resid_kurtosis',
    's_resid_skew',
]