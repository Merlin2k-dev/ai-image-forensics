"""Defocus / depth-of-field gradient profiles.

Real lenses produce a consistent sharpness falloff between the focal plane
and background; generators often paint uniform or inconsistent blur.
Measures per-tile sharpness statistics and the structure of the resulting
sharpness map.

Rejected: detectable on a single generator only, unstable across photo
sources.
"""
import numpy as np
from scipy import ndimage

T = 32                      # tile size -> 16x16 grid on 512
_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)

def _sharpness_grid(Y):
    lap = ndimage.convolve(Y, _LAP, mode="reflect")
    n = Y.shape[0] // T
    lt = lap[:n*T, :n*T].reshape(n, T, n, T).transpose(0, 2, 1, 3).reshape(n, n, T*T)
    return lt.var(axis=2)               # (n,n) per-tile Laplacian variance = focus measure

def _autocorr1(g):
    a, b = g[:, :-1].ravel(), g[:, 1:].ravel()
    c, d = g[:-1, :].ravel(), g[1:, :].ravel()
    x = np.r_[a, c]; y = np.r_[b, d]
    if x.std() < 1e-12 or y.std() < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def extract_dof_features(rgb_uint8_512):
    Y = (0.299*rgb_uint8_512[:,:,0] + 0.587*rgb_uint8_512[:,:,1] + 0.114*rgb_uint8_512[:,:,2]).astype(np.float64)
    S = _sharpness_grid(Y)
    logS = np.log(S + 1e-6)
    cv = float(S.std() / (S.mean() + 1e-9))
    autocorr = _autocorr1(logS)
    jump = float(np.mean(np.abs(np.r_[np.diff(logS, axis=0).ravel(), np.diff(logS, axis=1).ravel()])) / (logS.std() + 1e-9))
    # compactness of the in-focus region: normalized spatial inertia of top-quartile-sharp tiles
    thr = np.quantile(S, 0.75)
    ys, xs = np.where(S >= thr)
    if len(xs) > 1:
        cy, cx = ys.mean(), xs.mean()
        inertia = np.sqrt(((ys - cy)**2 + (xs - cx)**2).mean())
        compact = float(inertia / S.shape[0])    # smaller = one compact focal blob
    else:
        compact = 0.0
    return {
        "f_dof_sharp_cv": cv,
        "f_dof_autocorr": autocorr,
        "f_dof_jump":     jump,
        "f_dof_compact":  compact,
    }

DOF_COLS = ["f_dof_sharp_cv", "f_dof_autocorr", "f_dof_jump", "f_dof_compact"]
