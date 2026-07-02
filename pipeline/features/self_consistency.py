"""Local patch self-consistency features.

A real photo comes from one optical system and one ISP, so local statistics
(spectral slope, high-frequency fraction, noise floor) are consistent across
the frame. Patch-based or multi-pass generation pipelines can be locally
inconsistent. The image is tiled and the spatial dispersion of per-tile
statistics is measured; slope and HF fraction are content-robust, and slope
dispersion is additionally residualized against a per-tile content proxy.

Six columns (sc_*). Input: 512x512 RGB uint8, q75 substrate.
"""
import numpy as np
from scipy import ndimage

T = 64                       # tile size -> 8x8 = 64 tiles on 512
def _tiles(A):
    n = A.shape[0]//T
    return A[:n*T,:n*T].reshape(n,T,n,T).transpose(0,2,1,3).reshape(n*n,T,T), n

def _tile_spec(tile):
    win = np.outer(np.hanning(T), np.hanning(T))
    P = np.abs(np.fft.fftshift(np.fft.fft2((tile-tile.mean())*win)))**2
    c = T//2; y,x = np.indices((T,T)); r = np.hypot(x-c,y-c)
    rk = r.astype(int); prof = np.bincount(rk.ravel(), P.ravel())/np.maximum(np.bincount(rk.ravel()),1)
    k = np.arange(len(prof)); band = (k>=int(0.15*c)) & (k<=int(0.9*c)) & (prof>0)
    if band.sum() < 6: return np.nan, np.nan
    lf = np.log(k[band].astype(float)); lp = np.log(prof[band])
    slope = np.linalg.lstsq(np.vstack([lf,np.ones_like(lf)]).T, lp, rcond=None)[0][0]
    hf = prof[k>=int(0.5*c)].sum(); tot = prof[k>=1].sum()
    return float(slope), float(hf/(tot+1e-30))

def _adj_jump(grid):
    # mean abs difference between 4-neighbours on the tile grid
    d = []
    d.append(np.abs(np.diff(grid, axis=0)))
    d.append(np.abs(np.diff(grid, axis=1)))
    return float(np.nanmean(np.concatenate([x.ravel() for x in d])))

def extract_selfconsistency_features(rgb_uint8_512):
    Y = (0.299*rgb_uint8_512[:,:,0]+0.587*rgb_uint8_512[:,:,1]+0.114*rgb_uint8_512[:,:,2]).astype(np.float64)
    tiles, n = _tiles(Y)
    slopes = np.full(len(tiles), np.nan); hffr = np.full(len(tiles), np.nan); nfloor = np.full(len(tiles), np.nan)
    for i,t in enumerate(tiles):
        s,h = _tile_spec(t); slopes[i]=s; hffr[i]=h
        nfloor[i] = np.median(np.abs(t - ndimage.median_filter(t,3)))
    sg = slopes.reshape(n,n); hg = hffr.reshape(n,n)
    grad = np.array([np.nanmean(np.abs(np.gradient(t))) for t in tiles])   # per-tile content proxy
    # slope dispersion after removing content trend (regress slope on grad)
    ok = ~np.isnan(slopes)
    if ok.sum() > 8:
        A = np.vstack([grad[ok], np.ones(ok.sum())]).T
        coef = np.linalg.lstsq(A, slopes[ok], rcond=None)[0]
        resid_disp = float(np.std(slopes[ok] - A@coef))
    else:
        resid_disp = np.nan
    return {
        "sc_slope_disp":     float(np.nanstd(slopes)),
        "sc_slope_adjjump":  _adj_jump(sg),
        "sc_hffrac_cv":      float(np.nanstd(hffr)/(np.nanmean(hffr)+1e-9)),
        "sc_hffrac_adjjump": _adj_jump(hg),
        "sc_noisefloor_logdisp": float(np.nanstd(np.log(nfloor+1e-6))),
        "sc_slope_resid_disp":   resid_disp,
    }

SC_COLS = ["sc_slope_disp","sc_slope_adjjump","sc_hffrac_cv","sc_hffrac_adjjump",
           "sc_noisefloor_logdisp","sc_slope_resid_disp"]
