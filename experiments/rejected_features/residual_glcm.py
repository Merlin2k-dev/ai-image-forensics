"""Texture co-occurrence (GLCM) of the noise residual.

Gray-level co-occurrence statistics computed on the high-pass residual
rather than the image, looking for generator-typical residual textures.

Rejected: redundant with adopted families and shifts under recompression.
"""
import numpy as np
from scipy import ndimage

L = 16          # quantization levels for the residual
CLIP = 12.0     # residual clipped to [-CLIP, CLIP] before quantizing
OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1)]   # horiz, vert, two diagonals (averaged -> rotation-ish invariant)

def _glcm(q, di, dj):
    h, w = q.shape
    i0, i1 = max(0, -di), h - max(0, di)
    j0, j1 = max(0, -dj), w - max(0, dj)
    a = q[i0:i1, j0:j1].ravel()
    b = q[i0 + di:i1 + di, j0 + dj:j1 + dj].ravel()
    m = np.zeros((L, L), float)
    np.add.at(m, (a, b), 1.0)
    m = m + m.T                              # symmetric
    s = m.sum()
    return m / s if s > 0 else m

def _haralick(P):
    i = np.arange(L)[:, None] * np.ones((1, L))
    j = np.ones((L, 1)) * np.arange(L)[None, :]
    contrast = float((P * (i - j) ** 2).sum())
    homog = float((P / (1.0 + (i - j) ** 2)).sum())
    asm = float((P ** 2).sum())              # energy / angular second moment
    ent = float(-(P[P > 0] * np.log(P[P > 0])).sum())
    mu_i = (i * P).sum(); mu_j = (j * P).sum()
    si = np.sqrt(((i - mu_i) ** 2 * P).sum()); sj = np.sqrt(((j - mu_j) ** 2 * P).sum())
    corr = float(((i - mu_i) * (j - mu_j) * P).sum() / (si * sj)) if si > 1e-9 and sj > 1e-9 else 0.0
    return contrast, homog, asm, ent, corr

def extract_residual_glcm_features(rgb_uint8_512):
    Y = (0.299 * rgb_uint8_512[:, :, 0] + 0.587 * rgb_uint8_512[:, :, 1] + 0.114 * rgb_uint8_512[:, :, 2]).astype(np.float64)
    resid = Y - ndimage.median_filter(Y, 3)                 # high-pass residual
    r = np.clip(resid, -CLIP, CLIP)
    q = np.floor((r + CLIP) / (2 * CLIP) * (L - 1e-6)).astype(np.intp)   # 0..L-1
    feats = np.array([_haralick(_glcm(q, di, dj)) for di, dj in OFFSETS]).mean(axis=0)
    return {
        "f_rglcm_contrast":    float(feats[0]),
        "f_rglcm_homogeneity": float(feats[1]),
        "f_rglcm_energy":      float(feats[2]),
        "f_rglcm_entropy":     float(feats[3]),
        "f_rglcm_correlation": float(feats[4]),
    }

RGLCM_COLS = ["f_rglcm_contrast", "f_rglcm_homogeneity", "f_rglcm_energy",
              "f_rglcm_entropy", "f_rglcm_correlation"]
