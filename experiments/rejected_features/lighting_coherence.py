"""Lighting-direction consistency.

Estimates local illumination direction from shading gradients across
segments of the image and measures their agreement; composited or generated
scenes may disagree.

Rejected: clean on the controls but produced no measurable separation.
"""
import numpy as np
from scipy import ndimage

SHADE_SIGMA = 16.0     # low-pass to isolate the large-scale shading field from albedo/texture
T = 64                 # tile -> 8x8 grid

def extract_lighting_features(rgb_uint8_512):
    Y = (0.299*rgb_uint8_512[:,:,0] + 0.587*rgb_uint8_512[:,:,1] + 0.114*rgb_uint8_512[:,:,2]).astype(np.float64)
    shade = ndimage.gaussian_filter(Y, SHADE_SIGMA, truncate=3.0)
    gy, gx = np.gradient(shade)
    theta = np.arctan2(gy, gx)
    m = np.hypot(gx, gy)
    msum = m.sum() + 1e-12
    # global resultant length (magnitude-weighted circular concentration)
    z1 = (m * np.exp(1j * theta)).sum() / msum
    R = float(np.abs(z1))
    z2 = (m * np.exp(2j * theta)).sum() / msum
    bimod = float(np.abs(z2) / (np.abs(z1) + 1e-9))
    # per-tile dominant direction -> circular std across tiles
    n = Y.shape[0] // T
    angles = []
    for i in range(n):
        for j in range(n):
            sl = (slice(i*T,(i+1)*T), slice(j*T,(j+1)*T))
            zz = (m[sl] * np.exp(1j*theta[sl])).sum() / (m[sl].sum() + 1e-12)
            angles.append(zz / (np.abs(zz) + 1e-12))      # unit vector of tile mean direction
    Rt = np.abs(np.mean(angles))
    tile_cstd = float(np.sqrt(-2.0 * np.log(max(Rt, 1e-6))))   # circular std from resultant length
    return {
        "f_light_R":         R,
        "f_light_tile_cstd": tile_cstd,
        "f_light_bimod":     bimod,
    }

LIGHT_COLS = ["f_light_R", "f_light_tile_cstd", "f_light_bimod"]
