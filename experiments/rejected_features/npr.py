"""Down/up-resampling pixel residual (NPR-style).

One bilinear down-up cycle removes content but keeps Nyquist-adjacent
generator artifacts; the residual magnitude and structure differ between
real and generated images.

Rejected: borderline on the real-vs-real control and highly correlated with
the adopted grid features, so it added nothing.
"""
import numpy as np
from scipy import ndimage
def _lag1(a):
    x=a[:,:-1].ravel(); y=a[:,1:].ravel()
    return float(np.corrcoef(x,y)[0,1]) if x.std()>1e-9 else 0.0
def extract_npr_features(rgb_uint8_512):
    Y=(rgb_uint8_512.astype(np.float64)@[0.299,0.587,0.114])
    d=ndimage.zoom(ndimage.zoom(Y,0.5,order=1),2.0,order=1)[:Y.shape[0],:Y.shape[1]]
    r=Y-d
    Fr=np.abs(np.fft.fftshift(np.fft.fft2(r-r.mean()))); c=Y.shape[0]//2
    nyq=(Fr[c,:3].mean()+Fr[c,-3:].mean()+Fr[:3,c].mean()+Fr[-3:,c].mean())/4
    return {"npr_var":float(r.var()), "npr_nyq_snr":float(nyq/(np.median(Fr)+1e-9)), "npr_acf1":_lag1(r)}
NPR_COLS=["npr_var","npr_nyq_snr","npr_acf1"]
