"""Illuminant estimation consistency.

Computes gray-world and shades-of-gray illuminant estimates, their cast
distances from neutral, and the dispersion of per-tile illuminant estimates;
camera ISPs impose one global illuminant, generators may not.

Rejected: illuminant statistics differ systematically between photo sources
(fails the real-vs-real control).
"""

import numpy as np


_EPS = 1e-6   # chromaticity denominator guard (prevents div-by-zero in black px)

# Number of tiles per side (8x8 = 64 tiles of 64x64 px each)
_N_TILES = 8
_TILE_PX  = 64   # 512 / 8

# Canonical feature names 
WB_FEATURE_NAMES = [
    'wb_grayworld_cast',
    'wb_shadesofgray_cast',
    'wb_tile_illum_std_r',
    'wb_tile_illum_std_b',
    'wb_tile_illum_range',
    'wb_chroma_spread',
]


# Internal helpers

def _chromaticity(rgb_f64: np.ndarray) -> tuple:
    """Compute per-pixel chromaticity (r, b) from a float64 RGB image."""
    denom = rgb_f64[..., 0] + rgb_f64[..., 1] + rgb_f64[..., 2] + _EPS
    r_ch  = rgb_f64[..., 0] / denom
    b_ch  = rgb_f64[..., 2] / denom
    return r_ch, b_ch


def _grayworld_cast(rgb_f64: np.ndarray) -> float:
    """Gray-world illuminant cast distance from equal-energy neutral."""
    mu_r = rgb_f64[..., 0].mean()
    mu_g = rgb_f64[..., 1].mean()
    mu_b = rgb_f64[..., 2].mean()
    denom = mu_r + mu_g + mu_b + _EPS
    r_bar = mu_r / denom
    b_bar = mu_b / denom
    return float(np.sqrt((r_bar - 1.0/3.0)**2 + (b_bar - 1.0/3.0)**2))


def _shadesofgray_cast(rgb_f64: np.ndarray, p: int = 6) -> float:
    """Shades-of-gray (Minkowski p-norm) illuminant cast distance from neutral."""
    est_r = float(np.mean(rgb_f64[..., 0] ** p) ** (1.0 / p))
    est_g = float(np.mean(rgb_f64[..., 1] ** p) ** (1.0 / p))
    est_b = float(np.mean(rgb_f64[..., 2] ** p) ** (1.0 / p))
    denom = est_r + est_g + est_b + _EPS
    r_sog = est_r / denom
    b_sog = est_b / denom
    return float(np.sqrt((r_sog - 1.0/3.0)**2 + (b_sog - 1.0/3.0)**2))


def _tile_illuminant_stats(rgb_f64: np.ndarray) -> tuple:
    """Per-tile gray-world illuminant statistics over the 8x8 tile grid."""
    r_tiles = np.empty(_N_TILES * _N_TILES, dtype=np.float64)
    b_tiles = np.empty(_N_TILES * _N_TILES, dtype=np.float64)

    idx = 0
    for ti in range(_N_TILES):
        for tj in range(_N_TILES):
            tile = rgb_f64[
                ti * _TILE_PX : (ti + 1) * _TILE_PX,
                tj * _TILE_PX : (tj + 1) * _TILE_PX,
                :
            ]
            mu_r = tile[..., 0].mean()
            mu_g = tile[..., 1].mean()
            mu_b = tile[..., 2].mean()
            denom = mu_r + mu_g + mu_b + _EPS
            r_tiles[idx] = mu_r / denom
            b_tiles[idx] = mu_b / denom
            idx += 1

    std_r = float(np.std(r_tiles, ddof=1))
    std_b = float(np.std(b_tiles, ddof=1))

    # Per-tile chromaticity distance from neutral (1/3, 1/3)
    dist_tiles = np.sqrt((r_tiles - 1.0/3.0)**2 + (b_tiles - 1.0/3.0)**2)
    illum_range = float(np.percentile(dist_tiles, 95) - np.percentile(dist_tiles, 5))

    return std_r, std_b, illum_range


def _chroma_spread(rgb_f64: np.ndarray) -> float:
    """Std of per-pixel chromaticity radial distance from neutral."""
    denom = rgb_f64[..., 0] + rgb_f64[..., 1] + rgb_f64[..., 2] + _EPS
    r_px  = rgb_f64[..., 0] / denom
    b_px  = rgb_f64[..., 2] / denom
    dist  = np.sqrt((r_px - 1.0/3.0)**2 + (b_px - 1.0/3.0)**2)
    return float(np.std(dist, ddof=1))


# Public API

def extract_wb_features(rgb_uint8_512: np.ndarray) -> dict:
    """Extract white-balance / illuminant-consistency features from a 512x512 RGB image."""
    # Cast once to float64; channel values in [0, 255]
    rgb_f64 = rgb_uint8_512.astype(np.float64)

    gw_cast   = _grayworld_cast(rgb_f64)
    sog_cast  = _shadesofgray_cast(rgb_f64, p=6)

    std_r, std_b, illum_range = _tile_illuminant_stats(rgb_f64)

    chroma_sp = _chroma_spread(rgb_f64)

    return {
        'wb_grayworld_cast'     : gw_cast,
        'wb_shadesofgray_cast'  : sog_cast,
        'wb_tile_illum_std_r'   : std_r,
        'wb_tile_illum_std_b'   : std_b,
        'wb_tile_illum_range'   : illum_range,
        'wb_chroma_spread'      : chroma_sp,
    }
