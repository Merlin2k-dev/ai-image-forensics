"""Local cross-channel correlation fields.

Demosaicing couples R/G/B in a spatially patterned way. Maps local
inter-channel correlation over the frame and summarizes its texture,
looking for the absence of that coupling pattern.

Rejected: the correlation texture fingerprints the source pipeline
(fails the real-vs-real control).
"""

import logging

import numpy as np
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

_EPS       = 1e-12   # division guard for Pearson normalisation
_TILE_SIZE = 32      # tile edge length (pixels); yields 16x16 = 256 tiles on 512^2
_N_TILES   = (512 // _TILE_SIZE) ** 2   # 256


# Internal helpers

def _hp_residual(channel: np.ndarray) -> np.ndarray:
    """High-pass residual via 3x3 box-filter subtraction."""
    return channel - ndi.uniform_filter(channel, size=3)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1-D arrays.

    Returns np.nan when either array has near-zero variance (degenerate tile),
    which signals the caller to skip this tile's contribution to the std.

    Parameters
    ----------
    a, b : np.ndarray
    1-D float64 arrays of equal length.

    Returns
    -------
    float
    Pearson r in [-1, 1], or np.nan if degenerate.
    """
    if len(a) < 2:
        return np.nan
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < _EPS or sb < _EPS:
        return np.nan
    cov = float(np.mean((a - np.mean(a)) * (b - np.mean(b))))
    return cov / (sa * sb)


def _tile_generator(arr: np.ndarray, tile: int = _TILE_SIZE):
    """Yield (row_start, col_start, tile_2d) for non-overlapping tiles."""
    H, W = arr.shape
    for r in range(0, H - tile + 1, tile):
        for c in range(0, W - tile + 1, tile):
            yield r, c, arr[r:r + tile, c:c + tile]


# LC1 - Spatial Variance of Lag-1 Autocorrelation (H and V)

def feature_lc1_svlac(g_hp: np.ndarray) -> dict:
    """LC1: Spatial std of lag-1 horizontal and vertical autocorrelation."""
    h_corrs: list = []
    v_corrs: list = []

    for _, _, tile in _tile_generator(g_hp):
        # Horizontal lag-1: adjacent column pairs
        rh = _pearson(tile[:, :-1].ravel(), tile[:, 1:].ravel())
        if not np.isnan(rh):
            h_corrs.append(rh)

        # Vertical lag-1: adjacent row pairs
        rv = _pearson(tile[:-1, :].ravel(), tile[1:, :].ravel())
        if not np.isnan(rv):
            v_corrs.append(rv)

    svlac_h = float(np.std(h_corrs)) if len(h_corrs) >= 2 else 0.0
    svlac_v = float(np.std(v_corrs)) if len(v_corrs) >= 2 else 0.0

    if len(h_corrs) < _N_TILES or len(v_corrs) < _N_TILES:
        logger.debug(
            "LC1: %d/%d H-valid, %d/%d V-valid tiles (degenerate tiles skipped)",
            len(h_corrs), _N_TILES, len(v_corrs), _N_TILES,
        )

    return {'lc_svlac_h': svlac_h, 'lc_svlac_v': svlac_v}


# LC2 - Markov second-order dependency std

def feature_lc2_markov_d(g_hp: np.ndarray) -> dict:
    """LC2: Spatial std of lag-1 Markov-difference autocorrelation."""
    d_corrs: list = []

    for _, _, tile in _tile_generator(g_hp):
        D = np.diff(tile, axis=1)            # (32, 31); first horizontal difference
        d_seq = D.ravel()                    # 992 elements; lag-1 AC over sequence
        rd = _pearson(d_seq[:-1], d_seq[1:])
        if not np.isnan(rd):
            d_corrs.append(rd)

    markov_std = float(np.std(d_corrs)) if len(d_corrs) >= 2 else 0.0

    if len(d_corrs) < _N_TILES:
        logger.debug(
            "LC2: %d/%d valid tiles (degenerate tiles skipped)", len(d_corrs), _N_TILES
        )

    return {'lc_markov_d_std': markov_std}


# LC3 - Cross-channel HP residual correlation std

def feature_lc3_xchcorr(r_hp: np.ndarray,
                         g_hp: np.ndarray,
                         b_hp: np.ndarray) -> dict:
    """LC3: Spatial std of cross-channel HP residual Pearson correlation."""
    rg_corrs: list = []
    gb_corrs: list = []

    r_tiles = _tile_generator(r_hp)
    g_tiles = _tile_generator(g_hp)
    b_tiles = _tile_generator(b_hp)

    for (_, _, tr), (_, _, tg), (_, _, tb) in zip(r_tiles, g_tiles, b_tiles):
        tr_flat = tr.ravel()
        tg_flat = tg.ravel()
        tb_flat = tb.ravel()

        rg = _pearson(tr_flat, tg_flat)
        if not np.isnan(rg):
            rg_corrs.append(rg)

        gb = _pearson(tg_flat, tb_flat)
        if not np.isnan(gb):
            gb_corrs.append(gb)

    xch_rg = float(np.std(rg_corrs)) if len(rg_corrs) >= 2 else 0.0
    xch_gb = float(np.std(gb_corrs)) if len(gb_corrs) >= 2 else 0.0

    if len(rg_corrs) < _N_TILES or len(gb_corrs) < _N_TILES:
        logger.debug(
            "LC3: %d/%d RG-valid, %d/%d GB-valid tiles (degenerate skipped)",
            len(rg_corrs), _N_TILES, len(gb_corrs), _N_TILES,
        )

    return {'lc_xchcorr_rg_std': xch_rg, 'lc_xchcorr_gb_std': xch_gb}


# Public API

def extract_localcorr_features(rgb_uint8_512: np.ndarray) -> dict:
    """Extract all 5 local-correlation feature scalars from one 512x512 RGB image."""
    # Extract float64 channels
    R = rgb_uint8_512[:, :, 0].astype(np.float64)
    G = rgb_uint8_512[:, :, 1].astype(np.float64)
    B = rgb_uint8_512[:, :, 2].astype(np.float64)

    # High-pass residuals
    r_hp = _hp_residual(R)
    g_hp = _hp_residual(G)
    b_hp = _hp_residual(B)

    feats: dict = {}
    feats.update(feature_lc1_svlac(g_hp))
    feats.update(feature_lc2_markov_d(g_hp))
    feats.update(feature_lc3_xchcorr(r_hp, g_hp, b_hp))
    return feats


# Canonical ordered list of feature column names (exactly 5 columns).
LOCALCORR_FEATURE_NAMES = [
    # LC1 - Spatial Variance of Lag-1 Autocorrelation (2)
    'lc_svlac_h',
    'lc_svlac_v',
    # LC2 - Markov second-order dependency std (1)
    'lc_markov_d_std',
    # LC3 - Cross-channel HP correlation std (2)
    'lc_xchcorr_rg_std',
    'lc_xchcorr_gb_std',
]
