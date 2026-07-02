"""Natural scene statistics (BRISQUE-style).

MSCN coefficient shape parameters and pairwise-product statistics, the
classical no-reference quality features, tested as an AI/real separator.

Rejected: heavily source-coupled; these features are designed to be
sensitive to processing pipelines, which is exactly what varies between
photo corpora (fails the real-vs-real control).
"""
import numpy as np
from scipy import ndimage
from scipy.special import gamma as G

# GGD shape lookup (Sharifi & Leon-Garcia moment-matching): r(g)=Gamma(1/g)Gamma(3/g)/Gamma(2/g)^2
_GAMMAS = np.arange(0.20, 10.0, 0.001)
_RHO = G(1.0/_GAMMAS) * G(3.0/_GAMMAS) / (G(2.0/_GAMMAS)**2)

def _ggd_shape(x):
    x = x.ravel()
    s2 = np.mean(x**2); e = np.mean(np.abs(x))
    rho = s2 / (e**2 + 1e-12)
    g = _GAMMAS[np.argmin(np.abs(_RHO - rho))]
    return float(g), float(np.sqrt(s2))

def _mscn(Y):
    # 7x7 Gaussian (sigma=7/6), BRISQUE convention
    mu = ndimage.gaussian_filter(Y, 7.0/6.0, truncate=3.0)
    mu2 = ndimage.gaussian_filter(Y*Y, 7.0/6.0, truncate=3.0)
    sigma = np.sqrt(np.abs(mu2 - mu*mu))
    return (Y - mu) / (sigma + 1.0)

def extract_nss_features(rgb_uint8_512):
    Y = (0.299*rgb_uint8_512[:,:,0] + 0.587*rgb_uint8_512[:,:,1] + 0.114*rgb_uint8_512[:,:,2]).astype(np.float64)
    M = _mscn(Y)
    alpha, sig = _ggd_shape(M)
    kurt = float(((M - M.mean())**4).mean() / (M.var()**2 + 1e-12) - 3.0)
    # neighbour correlations (characteristic NSS signature)
    h1, h2 = M[:, :-1].ravel(), M[:, 1:].ravel()
    v1, v2 = M[:-1, :].ravel(), M[1:, :].ravel()
    corr_h = float(np.corrcoef(h1, h2)[0, 1])
    corr_v = float(np.corrcoef(v1, v2)[0, 1])
    return {
        "nss_mscn_alpha": alpha,    # real ~ Gaussian (high shape); AI deviates
        "nss_mscn_sigma": sig,
        "nss_mscn_kurt":  kurt,     # real MSCN ~ Gaussian (kurt ~ 0)
        "nss_pair_corr_h": corr_h,
        "nss_pair_corr_v": corr_v,
    }

NSS_COLS = ["nss_mscn_alpha", "nss_mscn_sigma", "nss_mscn_kurt", "nss_pair_corr_h", "nss_pair_corr_v"]
