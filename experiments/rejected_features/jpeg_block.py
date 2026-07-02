"""JPEG blockiness and block-boundary energy.

Measures 8x8 block-boundary discontinuities and their regularity. The idea
was that generator output re-encoded once differs from camera pipelines with
longer compression histories.

Rejected: this is the textbook shortcut. It measures compression history,
which varies by photo source, not by AI-ness (fails the real-vs-real
control).
"""
import numpy as np

def _marginal_block(Y):
    # horizontal first differences; boundary cols are index j where j%8==7 (between block j and j+1)
    dh = np.abs(np.diff(Y, axis=1))                  # shape (H, W-1), col k = |Y[:,k+1]-Y[:,k]|
    dv = np.abs(np.diff(Y, axis=0))
    cols = np.arange(dh.shape[1])
    rows = np.arange(dv.shape[0])
    hb = dh[:, cols % 8 == 7].mean(); hi = dh[:, cols % 8 != 7].mean()
    vb = dv[rows % 8 == 7, :].mean(); vi = dv[rows % 8 != 7, :].mean()
    # period-8 SNR of the column-mean difference profile (blocking shows as an 8-periodic comb)
    prof = dh.mean(axis=0)
    prof = prof - prof.mean()
    sp = np.abs(np.fft.rfft(prof))
    k8 = round(len(prof) / 8.0)
    band = sp[max(1, k8 - 1):k8 + 2].max() if k8 + 2 <= len(sp) else 0.0
    noise = np.median(sp[1:]) + 1e-9
    return hb, hi, vb, vi, float(band / noise)

def extract_jpegblock_features(rgb_uint8_512):
    Y = (0.299 * rgb_uint8_512[:, :, 0] + 0.587 * rgb_uint8_512[:, :, 1] + 0.114 * rgb_uint8_512[:, :, 2]).astype(np.float64)
    hb, hi, vb, vi, p8 = _marginal_block(Y)
    return {
        "f_block_h_excess": float(hb - hi),                 # boundary minus interior (horizontal)
        "f_block_v_excess": float(vb - vi),
        "f_block_ratio":    float((hb + vb) / (hi + vi + 1e-9)),
        "f_block_p8_snr":   float(p8),                      # strength of the 8-periodic blocking comb
    }

BLOCK_COLS = ["f_block_h_excess", "f_block_v_excess", "f_block_ratio", "f_block_p8_snr"]
