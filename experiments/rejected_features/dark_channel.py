"""Dark-channel prior statistics.

In haze-free real photos, most local patches contain at least one channel
near zero (the dark-channel prior of real light transport). Generators can
violate this. Measures dark-channel histograms and patch statistics.

Rejected: the q75 substrate compresses exactly the near-zero structure the
prior relies on; what remained was source-coupled.
"""
import numpy as np
from scipy import ndimage

PATCH = 15            # local patch radius for the dark-channel min-filter (He et al. use 15 on larger imgs)
ZERO_THRESH = 0.02    # "near zero" cutoff in [0,1] intensity for the p0 fraction

def extract_darkchannel_features(rgb_uint8_512):
    """Return dark-channel-prior scalars on a 512x512x3 uint8 RGB array."""
    A = rgb_uint8_512.astype(np.float64) / 255.0
    min_chan = A.min(axis=2)                                   # per-pixel min over RGB
    dark = ndimage.minimum_filter(min_chan, size=PATCH)        # local patch min (grayscale erosion)
    # near-zero fraction: in real photos most patches reach ~0; AI smooth regions do not
    p0 = float((dark < ZERO_THRESH).mean())
    return {
        "f_dc_mean": float(dark.mean()),     # AI -> higher (fewer truly dark patches)
        "f_dc_var":  float(dark.var()),      # spatial spread of the dark channel
        "f_dc_p0":   p0,                     # AI -> lower near-zero fraction
    }

DC_COLS = ["f_dc_mean", "f_dc_var", "f_dc_p0"]
