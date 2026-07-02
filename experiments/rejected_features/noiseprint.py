"""Sensor noise-print residual statistics.

Real cameras leave a near-white sensor noise floor present even in flat
regions; generator output lacks it. Measures flat-region residual energy,
whiteness and spatial uniformity.

Rejected: the residual fingerprints the ISP and compression pipeline of the
source, not AI-ness (fails the real-vs-real control).
"""
import numpy as np
from scipy import ndimage

def _luma(rgb):
    return (0.299*rgb[:,:,0] + 0.587*rgb[:,:,1] + 0.114*rgb[:,:,2]).astype(np.float64)

def _residual(Y):
    # edge-preserving fine-scale residual ~ noise: Y - median3x3(Y)
    return Y - ndimage.median_filter(Y, size=3)

def _acf(R, lag, axis):
    a = R - R.mean()
    if axis == 0:
        x = a[:-lag, :]; y = a[lag:, :]
    else:
        x = a[:, :-lag]; y = a[:, lag:]
    num = (x*y).mean(); den = a.var() + 1e-12
    return num/den

def _spectral_flatness(R):
    # Wiener entropy of the 2D power spectrum over the mid-high annulus (0.25..0.95 Nyquist)
    s = R.shape[0]
    win = np.outer(np.hanning(s), np.hanning(s))
    P = np.abs(np.fft.fftshift(np.fft.fft2(R*win)))**2
    c = s//2; y, x = np.indices((s, s)); r = np.hypot(x-c, y-c)/c
    m = (r >= 0.25) & (r <= 0.95)
    p = P[m]; p = p[p > 0]
    if p.size < 16: return np.nan
    return float(np.exp(np.mean(np.log(p))) / (np.mean(p) + 1e-30))

def _radial_slope(R):
    # slope of log radial power vs log freq over mid-high band: white noise ~ flat (slope ~0)
    s = R.shape[0]; win = np.outer(np.hanning(s), np.hanning(s))
    P = np.abs(np.fft.fftshift(np.fft.fft2(R*win)))**2
    c = s//2; y, x = np.indices((s, s)); rr = np.hypot(x-c, y-c).astype(int)
    prof = np.bincount(rr.ravel(), P.ravel())/np.maximum(np.bincount(rr.ravel()), 1)
    k = np.arange(len(prof)); band = (k >= int(0.25*c)) & (k <= int(0.95*c)) & (prof > 0)
    if band.sum() < 8: return np.nan
    lf = np.log(k[band].astype(float)); lp = np.log(prof[band])
    A = np.vstack([lf, np.ones_like(lf)]).T
    slope = np.linalg.lstsq(A, lp, rcond=None)[0][0]
    return float(slope)

def _flat_region_energy(Y, R, tile=16):
    # residual energy in the flattest tiles (low local gradient) - real noise present even where flat
    gx = ndimage.sobel(Y, axis=1); gy = ndimage.sobel(Y, axis=0)
    grad = np.hypot(gx, gy)
    s = Y.shape[0]; nt = s//tile
    gt = grad[:nt*tile, :nt*tile].reshape(nt, tile, nt, tile).mean(axis=(1,3))
    rt = (R**2)[:nt*tile, :nt*tile].reshape(nt, tile, nt, tile).mean(axis=(1,3))
    flat = gt <= np.percentile(gt, 20)
    return float(np.median(rt[flat])) if flat.any() else float(np.median(rt))

def extract_noiseprint_features(rgb_uint8_512):
    Y = _luma(rgb_uint8_512)
    R = _residual(Y)
    acf1 = 0.5*(abs(_acf(R,1,0)) + abs(_acf(R,1,1)))   # lag-1 whiteness (|.|)
    acf2 = 0.5*(abs(_acf(R,2,0)) + abs(_acf(R,2,1)))   # lag-2 whiteness
    return {
        "np_acf1": float(acf1),                 # white noise -> ~0
        "np_acf2": float(acf2),
        "np_spec_flat": _spectral_flatness(R),  # white -> ~1
        "np_radial_slope": _radial_slope(R),    # white -> ~0 (flat)
        "np_flat_energy": _flat_region_energy(Y, R),  # log later if needed
    }

NP_COLS = ["np_acf1","np_acf2","np_spec_flat","np_radial_slope","np_flat_energy"]
