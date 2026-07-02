"""Spectral features targeting VAE decoder grid artifacts.

Latent-diffusion decoders upsample with a fixed stride (8 px for SD2.1/SDXL,
16 px for FLUX/SD3.5) and leave a periodic bias at that pitch. JPEG 4:2:0 adds
its own axis-aligned energy at 8/16 px, but that lives on the frequency axes
and is identical for both classes on this substrate, so the measurements below
either null the axes or isolate off-axis grid energy.

Five groups, 18 columns: chroma period asymmetry (f1_*), cross-difference
anti-diagonal SNR (f2_*), gaussian-residual grid SNR (f3_*), radial spectrum
shape (f4_*), DCT grid coefficient excess (f5_*). Input: 512x512 RGB uint8,
already q75-harmonized.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.fft import dctn         # scipy >= 1.4; separable 2-D DCT-II


# Module-level constants  (pure arithmetic - no I/O)

_N   = 512          # image size
_K8  = 64           # rfft2 bin for period-8  (f = 1/8  -> bin 512/8  = 64)
_K16 = 32           # rfft2 bin for period-16 (f = 1/16 -> bin 512/16 = 32)
_EPS = 1e-10        # guard against zero-division and log(0)

# BT.601 full-swing RGB->YCbCr coefficients (input values in [0,255])
_YR,  _YG,  _YB  =  0.299,      0.587,     0.114
_CBR, _CBG, _CBB = -0.168736,  -0.331264,  0.500
_CRR, _CRG, _CRB =  0.500,     -0.418688, -0.083312

# Pre-computed frequency-fold row index array for 512-row rfft2 output.
# Row i in rfft2(512, ?) represents f_y = i/512 for i <= 256, or
# f_y = (i-512)/512 for i > 256.  The folded magnitude index is min(i, 512-i).
_RFFT_ROWS_512 = np.minimum(np.arange(512), 512 - np.arange(512))  # shape (512,)

# Same for 511-row rfft2 output (cross-difference image is 511x511)
_RFFT_ROWS_511 = np.minimum(np.arange(511), 511 - np.arange(511))  # shape (511,)


# Internal helpers

def _to_ycbcr(rgb: np.ndarray):
    """Convert uint8 RGB (H,W,3) -> float32 Y, Cb, Cr (H,W) each.

    Uses BT.601 full-swing coefficients.  DC offsets (128 for Cb/Cr) are
    included but have no effect on FFT power away from the DC bin.
    """
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    Y  = _YR  * R + _YG  * G + _YB  * B
    Cb = 128.0 + _CBR * R + _CBG * G + _CBB * B
    return Y, Cb


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma from uint8 RGB -> float32 (H, W)."""
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    return _YR * R + _YG * G + _YB * B


def _rfft2_power(channel: np.ndarray) -> np.ndarray:
    """|rfft2|^2  of a 2-D float32 channel."""
    F = np.fft.rfft2(channel)
    return F.real ** 2 + F.imag ** 2


def _box_sum_at_k(P: np.ndarray, k: int, hw: int = 3) -> float:
    """
    Sum power in three +/-hw-bin boxes around harmonic bin k.
    """
    H, W = P.shape
    r0  = max(0, k - hw);  r1 = min(H, k + hw + 1)
    c0  = max(0, k - hw);  c1 = min(W, k + hw + 1)
    box_h    = P[0 : hw + 1, c0:c1].sum()          # near f_y = 0
    box_v    = P[r0:r1,       0 : hw + 1].sum()     # near f_x = 0
    box_diag = P[r0:r1,       c0:c1].sum()          # at (k, k)
    return float(box_h + box_v + box_diag)


def _radial_mask(M_shape, k_ref, r_low, r_high, row_fold_arr):
    """Boolean mask for an annular ring in rfft2 magnitude space.

    Radial distance uses the folded row index so that negative-frequency rows
    (aliases at r > N/2) are correctly mapped to their positive-freq magnitude.

    Parameters
    ----------
    M_shape      : (H, W) of the magnitude array
    k_ref        : reference bin (the measured harmonic)  [unused here]
    r_low, r_high: inner and outer radii of the annulus (in bins from origin)
    row_fold_arr : pre-computed min(i, N-i) array of length H
    """
    H, W = M_shape
    jj = np.arange(W, dtype=np.float32)                  # col indices -> f_x bins
    # row_fold_arr already has length H; broadcast with jj for full 2-D map
    rf = row_fold_arr[:H].astype(np.float32)[:, np.newaxis]  # (H, 1)
    r_map = np.sqrt(rf ** 2 + jj[np.newaxis, :] ** 2)    # (H, W)
    return (r_map >= r_low) & (r_map <= r_high)


# F1 - Chroma-channel period asymmetry

def feature_f1_chroma_asymmetry(rgb: np.ndarray) -> dict:
    """
    F1: Chroma (Cb) vs luma (Y) power at the VAE-decoder grid harmonics.

    Features emitted (6):
    f1_Cb_box_k8, f1_Cb_box_k16    - raw Cb box sums
    f1_Y_box_k8,  f1_Y_box_k16     - raw Y box sums
    f1_Cb_8_over_16                 - Cb@k8 / Cb@k16
    f1_Y_16_over_8                  - Y@k16 / Y@k8
    """
    Y, Cb = _to_ycbcr(rgb)

    P_Y  = _rfft2_power(Y)
    P_Cb = _rfft2_power(Cb)

    Cb_k8  = _box_sum_at_k(P_Cb, _K8,  hw=3)
    Cb_k16 = _box_sum_at_k(P_Cb, _K16, hw=3)
    Y_k8   = _box_sum_at_k(P_Y,  _K8,  hw=3)
    Y_k16  = _box_sum_at_k(P_Y,  _K16, hw=3)

    return {
        'f1_Cb_box_k8'      : Cb_k8,
        'f1_Cb_box_k16'     : Cb_k16,
        'f1_Y_box_k8'       : Y_k8,
        'f1_Y_box_k16'      : Y_k16,
        'f1_Cb_8_over_16'   : Cb_k8  / (Cb_k16 + _EPS),
        'f1_Y_16_over_8'    : Y_k16  / (Y_k8   + _EPS),
    }


# F2 - Cross-difference anti-diagonal spectral peak (Synthbuster filter)

def feature_f2_cross_diff_snr(rgb: np.ndarray) -> dict:
    """
    F2: Diagonal spectral SNR after the cross-difference (Synthbuster) filter.

    Features emitted (4):
    f2_cd_snr_k8, f2_cd_snr_k16    - diagonal SNR at each period
    f2_cd_peak_k8, f2_cd_peak_k16  - raw diagonal peak (|rfft2|)
    """
    gray = _to_gray(rgb)

    # Cross-difference filter - kills all axis-aligned (JPEG) energy
    cd = (gray[:-1, :-1] + gray[1:, 1:]
          - gray[1:, :-1] - gray[:-1, 1:])          # shape (511, 511)
    M  = np.abs(np.fft.rfft2(cd))                   # shape (511, 256)

    cd_H = cd.shape[0]   # 511
    hw_peak = 2

    out = {}
    for p in (8, 16):
        k = cd_H // p                                # 63 or 31

        # Diagonal peak box: (k, k) +/-2
        r0 = max(0, k - hw_peak);  r1 = min(M.shape[0], k + hw_peak + 1)
        c0 = max(0, k - hw_peak);  c1 = min(M.shape[1], k + hw_peak + 1)
        peak = float(M[r0:r1, c0:c1].max())

        # Annular noise floor: radial ring [k+5, k+20] from origin
        ring_mask = _radial_mask(
            M.shape, k,
            r_low  = k + 5,
            r_high = k + 20,
            row_fold_arr = _RFFT_ROWS_511,
        )
        noise = float(np.median(M[ring_mask])) if ring_mask.any() else _EPS

        out[f'f2_cd_snr_k{p}']  = peak / (noise + _EPS)
        out[f'f2_cd_peak_k{p}'] = peak

    return out


# F3 - Gaussian-residual grid power

def feature_f3_gauss_residual(rgb: np.ndarray) -> dict:
    """
    F3: Grid power in the Gaussian high-pass residual.

    Features emitted (4):
    f3_gauss_snr_k8, f3_gauss_snr_k16   - normalised SNR
    f3_gauss_ongrid_k8, f3_gauss_ongrid_k16  - raw mean on-grid power
    """
    gray     = _to_gray(rgb)
    residual = gray - ndi.gaussian_filter(gray, sigma=1.5)
    P        = _rfft2_power(residual)                # shape (512, 257)

    hw = 2
    out = {}
    for p in (8, 16):
        k = _N // p    # 64 or 32

        r0 = max(0, k - hw);  r1 = min(P.shape[0], k + hw + 1)
        c0 = max(0, k - hw);  c1 = min(P.shape[1], k + hw + 1)

        # On-grid: three boxes at (k, col~0), (row~0, k), (k, k)
        box_vaxis  = P[r0:r1, 0:hw + 1]              # near f_x=0, at f_y~k/N
        box_haxis  = P[0:hw + 1, c0:c1]              # near f_y=0, at f_x~k/N
        box_diag   = P[r0:r1, c0:c1]                 # at (k, k)
        ongrid_mean = float(
            (box_vaxis.mean() + box_haxis.mean() + box_diag.mean()) / 3.0
        )

        # Background annular ring
        ring_mask = _radial_mask(
            P.shape, k,
            r_low  = k + 8,
            r_high = k + 20,
            row_fold_arr = _RFFT_ROWS_512,
        )
        bg_mean = float(P[ring_mask].mean()) if ring_mask.any() else _EPS

        out[f'f3_gauss_snr_k{p}']    = ongrid_mean / (bg_mean + _EPS)
        out[f'f3_gauss_ongrid_k{p}'] = ongrid_mean

    return out


# F4 - Azimuthal radial spectrum deviation

def feature_f4_radial_deviation(rgb: np.ndarray) -> dict:
    """
    F4: High-frequency excess / deficit relative to a mid-freq power law.

    Features emitted (2):
    f4_hf_deviation    - mean (measured - baseline) over r in [128, 220]
    f4_spectral_slope  - slope of line fit in log2P vs r space
    """
    gray = _to_gray(rgb)

    # Full FFT, shift DC to centre, log2 power
    F_shift  = np.fft.fftshift(np.fft.fft2(gray))
    log_P    = np.log2(F_shift.real ** 2 + F_shift.imag ** 2 + _EPS)

    H, W  = log_P.shape
    cy, cx = H // 2, W // 2

    # Radial distance map (integer bins)
    ii   = np.arange(H, dtype=np.float32) - cy
    jj   = np.arange(W, dtype=np.float32) - cx
    r_map = np.sqrt(ii[:, np.newaxis] ** 2 + jj[np.newaxis, :] ** 2)
    r_int = r_map.astype(np.int32)

    max_r = int(r_map.max()) + 1
    rp    = np.zeros(max_r, dtype=np.float64)
    cnt   = np.zeros(max_r, dtype=np.int64)
    np.add.at(rp,  r_int.ravel(), log_P.ravel())
    np.add.at(cnt, r_int.ravel(), 1)
    valid = cnt > 0
    rp[valid] /= cnt[valid]

    # Linear fit over mid radii [20, 100]
    mid_r = np.arange(20, 101)
    y_mid = rp[mid_r]
    A     = np.column_stack([mid_r, np.ones(len(mid_r))])
    coeffs, *_ = np.linalg.lstsq(A, y_mid, rcond=None)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # Compare measured vs extrapolated baseline over high radii [128, 220]
    # (clip to actual profile length)
    hi_end  = min(221, len(rp))
    high_r  = np.arange(128, hi_end)
    baseline_high  = slope * high_r + intercept
    measured_high  = rp[high_r]
    hf_deviation   = float(np.mean(measured_high - baseline_high))

    return {
        'f4_hf_deviation'   : hf_deviation,
        'f4_spectral_slope' : slope,
    }


# F5 - DCT grid coefficient excess (Frank method, no resize)

def feature_f5_dct_grid_excess(rgb: np.ndarray) -> dict:
    """
    F5: Excess log-DCT energy at grid harmonics vs neighbouring off-grid bins.

    Features emitted (2):
    f5_dct_excess_k8    - grid excess at period 8
    f5_dct_excess_k16   - grid excess at period 16
    """
    gray = _to_gray(rgb)

    # 2-D Type-II orthonormal DCT (separable, applied along both axes)
    C = dctn(gray, type=2, norm='ortho')             # shape (512, 512)
    L = np.log(np.abs(C) + _EPS)

    # Marginal profiles
    row_prof = L.mean(axis=1)   # average over columns -> shape (512,)
    col_prof = L.mean(axis=0)   # average over rows    -> shape (512,)

    out = {}
    for p in (8, 16):
        k = _N // p    # 64 or 32

        # Grid bins: k +/-1
        grid_r   = np.arange(max(0, k - 1), min(_N, k + 2))
        grid_c   = grid_r.copy()
        grid_mean = 0.5 * (row_prof[grid_r].mean() + col_prof[grid_c].mean())

        # Off-grid bins: k +/- d, d in {3..8}
        offsets  = np.arange(3, 9)                   # d = 3,4,5,6,7,8
        og_idx_r = np.concatenate([
            np.clip(k - offsets, 0, _N - 1),
            np.clip(k + offsets, 0, _N - 1),
        ])
        og_idx_c = og_idx_r.copy()
        offgrid_mean = 0.5 * (row_prof[og_idx_r].mean() + col_prof[og_idx_c].mean())

        out[f'f5_dct_excess_k{p}'] = float(grid_mean - offgrid_mean)

    return out


# Public API

def extract_vae_features(rgb_uint8_512: np.ndarray) -> dict:
    """
    Extract all 18 VAE-decoder fingerprint features from one image.
    """
    feats: dict = {}
    feats.update(feature_f1_chroma_asymmetry(rgb_uint8_512))
    feats.update(feature_f2_cross_diff_snr(rgb_uint8_512))
    feats.update(feature_f3_gauss_residual(rgb_uint8_512))
    feats.update(feature_f4_radial_deviation(rgb_uint8_512))
    feats.update(feature_f5_dct_grid_excess(rgb_uint8_512))
    return feats


# Canonical ordered list of feature column names produced by extract_vae_features.
FEATURE_NAMES = [
    # F1 - Chroma period asymmetry (6)
    'f1_Cb_box_k8', 'f1_Cb_box_k16',
    'f1_Y_box_k8',  'f1_Y_box_k16',
    'f1_Cb_8_over_16', 'f1_Y_16_over_8',
    # F2 - Cross-difference diagonal SNR (4)
    'f2_cd_snr_k8', 'f2_cd_snr_k16',
    'f2_cd_peak_k8', 'f2_cd_peak_k16',
    # F3 - Gaussian-residual grid SNR (4)
    'f3_gauss_snr_k8', 'f3_gauss_snr_k16',
    'f3_gauss_ongrid_k8', 'f3_gauss_ongrid_k16',
    # F4 - Radial spectrum deviation (2)
    'f4_hf_deviation', 'f4_spectral_slope',
    # F5 - DCT grid excess (2)
    'f5_dct_excess_k8', 'f5_dct_excess_k16',
]