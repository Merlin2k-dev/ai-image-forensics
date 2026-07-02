"""Upsampling-grid periodicity at non-standard pitches.

The adopted features look for grid energy at 8 and 16 px. This module scans
other candidate periods to catch decoders with different upsampling strides.

Rejected: chance-level. q75 compression erases faint decoder grids at
non-JPEG-aligned periods.
"""
import numpy as np
from scipy import ndimage

NONJPEG_PERIODS = [5, 6, 7, 9, 10, 11, 12, 13, 14]
N = 512

def _resid_spectrum_marginals(rgb):
    Y = (0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]).astype(np.float64)
    R = Y - ndimage.gaussian_filter(Y, 1.5)          # remove low-freq content
    win = np.outer(np.hanning(N), np.hanning(N))
    P = np.abs(np.fft.fft2(R*win))**2                # not shifted: bin k = k cycles/image
    # marginal spectra (axis-aligned grid peaks): sum over the orthogonal axis, keep [0..N/2]
    Sx = P.sum(axis=0)[:N//2]   # horizontal-frequency marginal (vertical grid lines)
    Sy = P.sum(axis=1)[:N//2]   # vertical-frequency marginal
    return Sx, Sy

def _peak_snr(S, bin_idx, half=6, guard=2):
    lo = max(1, bin_idx-half); hi = min(len(S)-1, bin_idx+half+1)
    peak = S[max(1,bin_idx-1):bin_idx+2].max()
    nb = np.r_[S[lo:max(1,bin_idx-guard)], S[bin_idx+guard+1:hi]]
    med = np.median(nb) if nb.size else np.nan
    return float(peak/(med+1e-30)) if med==med else np.nan

def _grid_snr(Sx, Sy, period):
    b = int(round(N/period))
    return 0.5*(_peak_snr(Sx, b) + _peak_snr(Sy, b))

def extract_gridperiod_features(rgb_uint8_512):
    Sx, Sy = _resid_spectrum_marginals(rgb_uint8_512)
    snr = {p: _grid_snr(Sx, Sy, p) for p in NONJPEG_PERIODS}
    nonjpeg = np.array([snr[p] for p in NONJPEG_PERIODS], float)
    out = {f"gp_snr_p{p}": snr[p] for p in (5,6,7,9,10,12)}   # 6 representative non-JPEG periods
    out["gp_max_nonjpeg"]  = float(np.nanmax(nonjpeg))
    out["gp_mean_nonjpeg"] = float(np.nanmean(nonjpeg))
    # audit-only (not model features): JPEG-aligned periods for the confound comparison
    out["_gp_snr_p8"]  = _grid_snr(Sx, Sy, 8)
    out["_gp_snr_p16"] = _grid_snr(Sx, Sy, 16)
    return out

GP_COLS = ["gp_snr_p5","gp_snr_p6","gp_snr_p7","gp_snr_p9","gp_snr_p10","gp_snr_p12",
           "gp_max_nonjpeg","gp_mean_nonjpeg"]
