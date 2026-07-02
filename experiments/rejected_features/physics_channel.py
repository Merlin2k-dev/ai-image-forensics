"""Physically-motivated inter-channel correlations.

Sensor physics couples channels multiplicatively (illumination) while
generators synthesize channels jointly in learned latents; measures
correlation structure between chromaticity and intensity.

Rejected: color-adjacent statistics are source-coupled (fails the
real-vs-real control).
"""
import numpy as np
from scipy import ndimage

def _hf(ch):
    return ch - ndimage.gaussian_filter(ch, 1.0)

def _pred_resid_ratio(target, guide):
    # 1 - R^2 of predicting target HF from guide HF (least squares, with lag-0 and 1-px shifts)
    t = target.ravel(); g0 = guide.ravel()
    gx = np.roll(guide,1,axis=1).ravel(); gy = np.roll(guide,1,axis=0).ravel()
    A = np.vstack([g0, gx, gy, np.ones_like(g0)]).T
    coef,_,_,_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - A@coef
    return float(np.var(resid)/(np.var(t)+1e-12))

def extract_physics_channel_features(rgb_uint8_512):
    R=_hf(rgb_uint8_512[:,:,0].astype(np.float64)); G=_hf(rgb_uint8_512[:,:,1].astype(np.float64)); B=_hf(rgb_uint8_512[:,:,2].astype(np.float64))
    def corr(a,b):
        a=a.ravel()-a.mean(); b=b.ravel()-b.mean(); return float((a*b).mean()/(a.std()*b.std()+1e-12))
    return {
        "cc_corr_rg": corr(R,G),                       # demosaicing ties R-HF to G-HF -> high in real
        "cc_corr_bg": corr(B,G),
        "cc_resid_rg": _pred_resid_ratio(R,G),         # G-unexplained R variance -> low in real
        "cc_resid_bg": _pred_resid_ratio(B,G),
        "cc_corr_rb": corr(R,B),
    }

CC_COLS=["cc_corr_rg","cc_corr_bg","cc_resid_rg","cc_resid_bg","cc_corr_rb"]
