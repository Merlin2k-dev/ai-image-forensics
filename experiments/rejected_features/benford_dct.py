"""First-digit (Benford) statistics of block DCT coefficients.

Natural images follow a generalized Benford law in the first digits of their
DCT coefficients; generator output can deviate. Measures KL and chi-squared
distance to the Benford laws, fitted generalized-Benford parameters, and
digit-distribution shape moments on the 8x8 block DCT.

Rejected: effect is weak on modern generators and digit distributions track
content density, which differs between photo sources (fails the real-vs-real
control).
"""

import logging
import numpy as np
from scipy.fftpack import dctn
from scipy import ndimage as ndi
from scipy import stats as sp_stats
from scipy import signal as sp_signal
from scipy import optimize as sp_opt

logger = logging.getLogger(__name__)

# Module-level constants

_EPS = 1e-12

# Benford probabilities for digits 1-9: P(d) = log10(1 + 1/d)
_BENFORD_PROBS = np.array([np.log10(1.0 + 1.0 / d) for d in range(1, 10)])
# Shape: (9,), sum ~ 1.0

# Threshold below which a DCT coefficient absolute value is treated as zero
# and excluded from the first-significant-digit analysis.
# 1e-3 avoids numerical noise near zero (residuals after filtering can be tiny).
_COEFF_THRESHOLD = 1e-3

# Zig-zag scan order: list of (row, col) tuples for an 8x8 DCT block.
# Index 0 = DC coefficient at (0,0).
# Indices 1-63 = AC coefficients in increasing spatial-frequency order.
# See _build_zigzag_scan() for construction details.
_ZIGZAG_SCAN: list = []   # populated at module import by _build_zigzag_scan()


# Zig-zag scan helper

def _build_zigzag_scan() -> list:
    """Build standard JPEG zig-zag scan order for an 8x8 DCT block.

    The zig-zag order traverses 8x8 DCT coefficients in order of roughly
    increasing 2-D spatial frequency (consistent with the JPEG standard,
    Annex B).  Diagonals alternate direction: even sums (r+c) go up-right
    (row decreasing), odd sums go down-left (row increasing).

    Returns
    -------
    list of (int, int)
    64 (row, col) pairs ordered by zig-zag position 0 (DC) through 63.
    """
    N = 8
    pairs: list = []
    for s in range(2 * N - 1):
        if s % 2 == 0:           # up-right diagonal
            r = min(s, N - 1)
            c = s - r
            while r >= 0 and c < N:
                pairs.append((r, c))
                r -= 1
                c += 1
        else:                    # down-left diagonal
            c = min(s, N - 1)
            r = s - c
            while c >= 0 and r < N:
                pairs.append((r, c))
                r += 1
                c -= 1
    assert len(pairs) == 64, "zig-zag must have exactly 64 entries"
    return pairs


_ZIGZAG_SCAN = _build_zigzag_scan()

# Pre-compute (rows, cols) arrays for vectorised indexing into (n_blocks, 8, 8)
_ZIGZAG_ROWS = np.array([r for r, _ in _ZIGZAG_SCAN], dtype=np.intp)
_ZIGZAG_COLS = np.array([c for _, c in _ZIGZAG_SCAN], dtype=np.intp)


# Luma conversion

def _luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma from uint8 RGB (H, W, 3) -> float64 (H, W).

    Coefficients: Y = 0.299.R + 0.587.G + 0.114.B
    Returned as float64 for precision in variance / moment computation.
    """
    R = rgb[:, :, 0].astype(np.float64)
    G = rgb[:, :, 1].astype(np.float64)
    B = rgb[:, :, 2].astype(np.float64)
    return 0.299 * R + 0.587 * G + 0.114 * B


# Vectorised block DCT

def _block_dct_8x8(img: np.ndarray) -> np.ndarray:
    """Non-overlapping 8x8 block DCT-II (ortho norm) via scipy.fftpack.

    Reshape/stride trick avoids a Python loop over 4096 blocks.

    Parameters
    ----------
    img : np.ndarray, shape (512, 512), float64
    Single-channel image (luma or residual).

    Returns
    -------
    np.ndarray, shape (4096, 8, 8)
    DCT coefficients for each of the 64x64 = 4096 non-overlapping 8x8
    blocks.  Coefficient [k, 0, 0] is the DC term of block k.
    """
    H, W = img.shape
    bs = 8
    nb_r = H // bs   # 64
    nb_c = W // bs   # 64

    # Reshape to (nb_r, bs, nb_c, bs), then to (n_blocks, bs, bs)
    blocks = img.reshape(nb_r, bs, nb_c, bs).transpose(0, 2, 1, 3).reshape(-1, bs, bs)

    # Apply 2-D DCT-II (ortho) over the last two axes simultaneously
    # scipy.fftpack.dctn with axes=(-2,-1) applies an independent 2-D DCT
    # to each of the 4096 slices along axis 0.
    dct_blocks = dctn(blocks, type=2, norm='ortho', axes=(-2, -1))
    return dct_blocks   # (4096, 8, 8)


# First-significant-digit extraction (vectorised)

def _first_sig_digit(coeffs: np.ndarray,
                     threshold: float = _COEFF_THRESHOLD) -> np.ndarray:
    """Extract the first significant digit (1-9) for each coefficient."""
    a = np.abs(coeffs.ravel())
    a = a[a > threshold]
    if len(a) == 0:
        return np.empty(0, dtype=int)

    exp = np.floor(np.log10(a))
    digits = np.floor(a / 10.0 ** exp).astype(int)
    return np.clip(digits, 1, 9)


# Histogram and divergence utilities

def _digit_hist(digits: np.ndarray):
    """Normalised first-significant-digit histogram over d = 1..9.

    Parameters
    ----------
    digits : 1-D int array with values in {1,...,9}.

    Returns
    -------
    hist_norm : np.ndarray, shape (9,), float64
    Normalised probabilities summing to ~1.  Returns None if empty.
    n_total   : int
    Total count.
    """
    if len(digits) == 0:
        return None, 0
    counts = np.bincount(digits, minlength=10)[1:10].astype(np.float64)  # d=1..9
    n = int(counts.sum())
    if n == 0:
        return None, 0
    return counts / n, n


def _kl_vs_benford(hist_norm: np.ndarray) -> float:
    """KL divergence  KL(P_obs || P_Benford).

    Uses the convention  KL = sum_d P_obs(d) . log(P_obs(d) / P_Benford(d)).
    Terms where P_obs(d) = 0 contribute 0 (0.log0 = 0 by convention).

    Parameters
    ----------
    hist_norm : np.ndarray, shape (9,), already normalised observed probabilities.

    Returns
    -------
    float - non-negative KL divergence in nats.
    """
    kl = 0.0
    for i in range(9):
        p = hist_norm[i]
        if p > 0.0:
            kl += p * np.log(p / (_BENFORD_PROBS[i] + _EPS))
    return float(kl)


def _chi2_vs_benford(digits: np.ndarray) -> float:
    """Chi-squared statistic  chi^2 = sum_d (O_d - E_d)^2 / E_d  vs Benford.

    Parameters
    ----------
    digits : 1-D int array with values in {1,...,9}.

    Returns
    -------
    float - chi-squared statistic (larger = further from Benford).
    """
    n = len(digits)
    if n == 0:
        return float('nan')
    counts = np.bincount(digits, minlength=10)[1:10].astype(np.float64)
    expected = _BENFORD_PROBS * n
    return float(np.sum((counts - expected) ** 2 / (expected + _EPS)))


# Generalized-Benford fitting (B3)

def _fit_generalized_benford(hist_norm: np.ndarray):
    """Fit the Fu/Shi/Su generalized Benford law to an observed digit histogram."""
    d = np.arange(1.0, 10.0)                # d = 1..9

    def _mse(alpha: float) -> float:
        P_model = np.log((d + 1.0) / d) / np.log(alpha)
        return float(np.mean((hist_norm - P_model) ** 2))

    result = sp_opt.minimize_scalar(
        _mse,
        bounds=(1.01, 100.0),
        method='bounded',
        options={'xatol': 1e-4},
    )
    return float(result.fun), float(result.x)


# Feature group implementations

def feature_b1_imgdomain_midband_kl(dct_blocks: np.ndarray) -> dict:
    """B1: Image-domain mid-band Benford KL divergence (compression-prone control).

    Features emitted (1):
    b_img_midband_kl  - KL(observed || Benford) for mid-band AC digits.
    """
    # Extract zig-zag modes 10-35 from each block
    # _ZIGZAG_ROWS/COLS[10:36] gives the (r, c) positions of those modes
    mode_slice = slice(10, 36)
    rows = _ZIGZAG_ROWS[mode_slice]
    cols = _ZIGZAG_COLS[mode_slice]
    coeffs = dct_blocks[:, rows, cols].ravel()   # (4096 * 26,)

    digits = _first_sig_digit(coeffs)
    hist_norm, n = _digit_hist(digits)

    if hist_norm is None:
        logger.debug("B1: empty coefficient pool - returning NaN")
        return {'b_img_midband_kl': float('nan')}

    kl = _kl_vs_benford(hist_norm)
    return {'b_img_midband_kl': kl}


def feature_b2_residual_benford(Y: np.ndarray) -> dict:
    """B2: Residual-domain Benford KL and chi-squared (priority, compression-decoupled)."""
    residual = Y - ndi.median_filter(Y, size=3)
    dct_blocks = _block_dct_8x8(residual)

    # All non-DC AC coefficients (modes 1..63 = all except [0,0])
    # Flatten (4096, 8, 8), zero out DC, flatten
    dct_flat = dct_blocks.copy()
    dct_flat[:, 0, 0] = 0.0           # zero DC in all blocks
    coeffs = dct_flat.ravel()

    digits = _first_sig_digit(coeffs)
    hist_norm, n = _digit_hist(digits)

    if hist_norm is None:
        logger.debug("B2: empty residual coefficient pool - returning NaN")
        return {
            'b_resid_kl'   : float('nan'),
            'b_resid_chi2' : float('nan'),
            '_resid_hist'  : None,
            '_resid_digits': None,
        }

    kl   = _kl_vs_benford(hist_norm)
    chi2 = _chi2_vs_benford(digits)

    return {
        'b_resid_kl'   : kl,
        'b_resid_chi2' : chi2,
        '_resid_hist'  : hist_norm,   # shared with B3
        '_resid_digits': digits,      # shared with B3
    }


def feature_b3_generalized_benford(hist_norm) -> dict:
    """B3: Generalized-Benford fit error and fitted alpha (compression-aware by design).

    Features emitted (2):
    b_genben_mse    - minimum MSE of generalised Benford fit.
    b_genben_alpha  - optimal alpha (~ 10 for Benford-consistent distributions).
    """
    if hist_norm is None:
        return {'b_genben_mse': float('nan'), 'b_genben_alpha': float('nan')}

    mse, alpha = _fit_generalized_benford(hist_norm)
    return {'b_genben_mse': mse, 'b_genben_alpha': alpha}


def feature_b4_midband_moments(dct_blocks: np.ndarray) -> dict:
    """B4: Mid-band block-DCT higher-order statistical moments.

    Features emitted (2):
    b_dct_kurtosis - excess kurtosis (Fisher) of pooled mid-band AC coefficients.
    b_dct_skew     - skewness of pooled mid-band AC coefficients.
    """
    mode_slice = slice(10, 31)
    rows = _ZIGZAG_ROWS[mode_slice]
    cols = _ZIGZAG_COLS[mode_slice]
    pool = dct_blocks[:, rows, cols].ravel()   # (4096 * 21,)

    if len(pool) < 4:
        logger.debug("B4: pool too small for moments - returning NaN")
        return {'b_dct_kurtosis': float('nan'), 'b_dct_skew': float('nan')}

    kurt = float(sp_stats.kurtosis(pool, fisher=True))
    skew = float(sp_stats.skew(pool))
    return {'b_dct_kurtosis': kurt, 'b_dct_skew': skew}


def feature_b5_wiener_residual_benford(Y: np.ndarray) -> dict:
    """B5: Wiener-residual DCT Benford chi-squared (fine-scale cross-check).

    Features emitted (1):
    b_resid_ovlp_chi2 - chi^2 vs Benford for Wiener-residual AC digits.
    """
    # Wiener filter on float64.  Suppress divide-by-zero / invalid-value
    # warnings that occur in flat image regions (where local variance = 0);
    # scipy.signal.wiener replaces those pixels with the local mean, which
    # is numerically correct (noise = 0 there -> residual = 0).
    with np.errstate(divide='ignore', invalid='ignore'):
        filtered = sp_signal.wiener(Y, mysize=3)
    # Replace any NaN/inf produced for constant-region pixels with Y itself
    # (residual -> 0, which will be below the coefficient threshold and ignored).
    filtered = np.where(np.isfinite(filtered), filtered, Y)
    residual = Y - filtered

    dct_blocks = _block_dct_8x8(residual)

    # All non-DC AC coefficients (zero out DC)
    dct_flat = dct_blocks.copy()
    dct_flat[:, 0, 0] = 0.0
    coeffs = dct_flat.ravel()

    digits = _first_sig_digit(coeffs)
    if len(digits) == 0:
        logger.debug("B5: empty Wiener residual coefficient pool - returning NaN")
        return {'b_resid_ovlp_chi2': float('nan')}

    chi2 = _chi2_vs_benford(digits)
    return {'b_resid_ovlp_chi2': chi2}


# Public API

def extract_benford_features(rgb_uint8_512: np.ndarray) -> dict:
    """Extract all 8 Benford/block-DCT features from one image."""
    Y = _luma(rgb_uint8_512)   # (512, 512) float64

    dct_img = _block_dct_8x8(Y)   # (4096, 8, 8)

    feats = {}
    feats.update(feature_b1_imgdomain_midband_kl(dct_img))

    b2_result = feature_b2_residual_benford(Y)
    resid_hist   = b2_result.pop('_resid_hist', None)
    b2_result.pop('_resid_digits', None)
    feats.update(b2_result)

    feats.update(feature_b3_generalized_benford(resid_hist))

    feats.update(feature_b4_midband_moments(dct_img))

    feats.update(feature_b5_wiener_residual_benford(Y))

    return feats


# Canonical ordered list of feature column names (8 scalars).
BENFORD_FEATURE_NAMES: list = [
    # B1 - Image-domain mid-band Benford KL (compression-prone control, 1)
    'b_img_midband_kl',
    # B2 - Residual-domain Benford statistics (priority, 2)
    'b_resid_kl',
    'b_resid_chi2',
    # B3 - Generalized-Benford fit (compression-aware, 2)
    'b_genben_mse',
    'b_genben_alpha',
    # B4 - Mid-band higher-order moments (Pontorno 2024, 2)
    'b_dct_kurtosis',
    'b_dct_skew',
    # B5 - Wiener-residual Benford chi-squared (fine-scale cross-check, 1)
    'b_resid_ovlp_chi2',
]
