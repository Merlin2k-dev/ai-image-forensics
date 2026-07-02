"""Perspective and vanishing-point coherence.

Measures edge-orientation structure: Sobel orientation concentration, the
fraction of near-horizontal/vertical (Manhattan) edges, and per-tile
orientation agreement; generated scenes can carry subtle geometry errors.

Rejected: the apparent lift did not replicate across real photo corpora.
"""
import numpy as np
from scipy import ndimage

def _edges(Y):
    gx = ndimage.sobel(Y, axis=1); gy = ndimage.sobel(Y, axis=0)
    mag = np.hypot(gx, gy)
    edge_ang = np.arctan2(gy, gx) + np.pi/2.0      # edge direction = grad dir + 90
    return mag, edge_ang

def extract_perspective_features(rgb_uint8_512):
    Y = (0.299*rgb_uint8_512[:,:,0] + 0.587*rgb_uint8_512[:,:,1] + 0.114*rgb_uint8_512[:,:,2]).astype(np.float64)
    mag, ang = _edges(Y)
    thr = np.quantile(mag, 0.80)
    strong = mag >= thr
    a = ang[strong]; m = mag[strong]
    msum = m.sum() + 1e-9
    # global axial concentration (orientation mod pi -> 2*angle)
    conc = float(np.abs((m * np.exp(2j*a)).sum()) / msum)
    # Manhattan fraction: edge energy within +-15deg of horizontal(0) or vertical(pi/2), mod pi
    amod = np.mod(a, np.pi)
    near_h = (amod < np.deg2rad(15)) | (amod > np.pi - np.deg2rad(15))
    near_v = np.abs(amod - np.pi/2) < np.deg2rad(15)
    hv_frac = float(m[near_h | near_v].sum() / msum)
    # per-tile local straightness (mean axial concentration over 8x8 tiles)
    T = 64; n = Y.shape[0]//T; tconc = []
    for i in range(n):
        for j in range(n):
            sl = (slice(i*T,(i+1)*T), slice(j*T,(j+1)*T))
            tm = mag[sl]; ta = ang[sl]
            s = tm.sum() + 1e-9
            tconc.append(np.abs((tm * np.exp(2j*ta)).sum())/s)
    tile_conc = float(np.mean(tconc))
    return {
        "f_persp_orient_conc": conc,
        "f_persp_hv_frac":     hv_frac,
        "f_persp_tile_conc":   tile_conc,
    }

PERSP_COLS = ["f_persp_orient_conc", "f_persp_hv_frac", "f_persp_tile_conc"]
