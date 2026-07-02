"""Fourier phase-spectrum statistics.

Magnitude-only spectra are widely studied; this measures the phase side:
phase congruency distributions, circular variance of phase over radial
rings, and a phase-only reconstruction correlation.

Rejected: no held-out gain.
"""

import logging
import math

import numpy as np
import numpy.fft as npfft

logger = logging.getLogger(__name__)

_EPS = 1e-10   # universal division-by-zero guard

# feature column names
PHASE_FEATURE_NAMES = [
    # P3 - phase congruency (2)
    'p_phasecong_mean',
    'p_phasecong_std',
    # P1 - global phase entropy (2)
    'p_phase_entropy_Y',
    'p_phase_entropy_Cb',
    # P2 - radial phase circular variance (2)
    'p_radial_circvar_mean',
    'p_radial_circvar_std',
    # P4 - phase-only reconstruction NCC (1)
    'p_phaseonly_ncc',
]

# Internal helpers: channel conversions

def _luma(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma from uint8 RGB (H,W,3) -> float64 (H,W).

    Y = 0.299.R + 0.587.G + 0.114.B

    The coefficients follow ITU-R BT.601 and match the substrate JPEG's
    internal luma channel (YCbCr 4:2:0).  Output is in [0, 255].
    """
    return (0.299  * rgb[:, :, 0].astype(np.float64)
            + 0.587 * rgb[:, :, 1].astype(np.float64)
            + 0.114 * rgb[:, :, 2].astype(np.float64))


def _chroma_cb(rgb: np.ndarray) -> np.ndarray:
    """BT.601 Cb (blue-difference chroma) from uint8 RGB -> float64 (H,W).

    Cb = -0.16875.R - 0.33126.G + 0.5.B  (range ~ [-127.5, 127.5])

    using the analogue BT.601 forward-transform
    coefficients (pre-offset form, no +128 bias) so the mean is near 0
    for typical images and the phase histogram is centred.  The annulus
    excludes DC (r<5) so the bias does not affect P1.
    """
    return (-0.16875 * rgb[:, :, 0].astype(np.float64)
            - 0.33126 * rgb[:, :, 1].astype(np.float64)
            + 0.5     * rgb[:, :, 2].astype(np.float64))


# Internal helper: mid-frequency annulus mask (fftshifted coordinates)

def _annulus_mask(h: int, w: int, r_min: float = 5.0, r_max: float = 100.0) -> np.ndarray:
    """Boolean mask for the mid-frequency annulus on a fftshifted DFT grid."""
    cy, cx = h // 2, w // 2
    ys = np.arange(h, dtype=np.float64) - cy
    xs = np.arange(w, dtype=np.float64) - cx
    rr = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)   # (h, w)
    return (rr >= r_min) & (rr <= r_max)


# P3 - Phase congruency via Log-Gabor bank  (Kovesi 1999; 2 features)

def _log_gabor_bank(h: int, w: int,
                    n_scales: int = 4,
                    min_wavelength: float = 10.0,
                    mult: float = 2.0,
                    sigma_on_f: float = 0.55) -> list:
    """Build a bank of Log-Gabor filters in the frequency domain."""
    cy, cx = h // 2, w // 2
    ys = np.arange(h, dtype=np.float64) - cy
    xs = np.arange(w, dtype=np.float64) - cx
    rr = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)   # fftshifted radius (pixels)
    rr_norm = rr / max(h, w)                              # cycles / pixel

    # Avoid log(0) at DC
    rr_norm[cy, cx] = 1.0   # temporary; will be zeroed by the filter

    filters = []
    for k in range(n_scales):
        f_k = 1.0 / (min_wavelength * (mult ** k))
        log_sigma2 = (math.log(sigma_on_f)) ** 2 * 2.0   # 2.(log(sigma/f_k))^2
        G = np.exp(-(np.log(rr_norm / f_k) ** 2) / log_sigma2)
        G[cy, cx] = 0.0   # suppress DC
        # ifftshift to align with npfft.fft2 (unshifted) output
        G = npfft.ifftshift(G)
        filters.append(G)

    return filters


def feature_p3_phase_congruency(rgb: np.ndarray,
                                 n_scales: int = 4,
                                 sigma_on_f: float = 0.55) -> dict:
    """P3: Phase congruency from isotropic Log-Gabor bank - 2 features.

    Features emitted (2):
    p_phasecong_mean  - mean(PC_map);  AI -> lower
    p_phasecong_std   - std(PC_map);   direction varies, use as confirmatory
    """
    Y = _luma(rgb)   # (512, 512) float64
    h, w = Y.shape

    F = npfft.fft2(Y)   # (512, 512) complex128

    gabors = _log_gabor_bank(h, w, n_scales=n_scales, sigma_on_f=sigma_on_f)

    # Scale responses in the spatial domain (complex)
    resps = [npfft.ifft2(F * G) for G in gabors]

    # Mean-phase: angle of sum of responses
    resp_sum = sum(resps)                          # (512, 512) complex128
    mean_phase = np.angle(resp_sum)                # (512, 512) float64

    # PC numerator: energy-weighted phase alignment
    amplitudes = np.array([np.abs(r) for r in resps])         # (n_scales, H, W)
    phase_diffs = np.array([np.angle(r) - mean_phase
                            for r in resps])                    # (n_scales, H, W)

    numerator   = np.sum(amplitudes * np.cos(phase_diffs), axis=0)   # (H, W)
    denominator = np.sum(amplitudes, axis=0) + 1e-3                   # (H, W)
    pc_map = numerator / denominator                                    # (H, W)

    return {
        'p_phasecong_mean': float(np.mean(pc_map)),
        'p_phasecong_std' : float(np.std(pc_map)),
    }


# P1 - Global phase entropy in mid-frequency annulus (2 features)

def _phase_entropy(channel: np.ndarray,
                   mask: np.ndarray,
                   n_bins: int = 256) -> float:
    """Differential entropy of the phase histogram over a masked DFT region."""
    F = npfft.fft2(channel)                     # unshifted DFT
    F_shift = npfft.fftshift(F)                 # DC at centre
    phases = np.angle(F_shift[mask])            # 1-D array in (-pi, pi]

    if phases.size == 0:
        logger.debug("_phase_entropy: empty mask - returning 0.0")
        return 0.0

    bin_edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    # density=True -> hist values are probability densities (integral = 1)
    hist, _ = np.histogram(phases, bins=bin_edges, density=True)

    # Differential entropy Riemann sum: -sum f_i . log(f_i) . delta
    # where f_i = hist[i] is the density value (not probability mass).
    # For uniform over [-pi,pi]: f_i = 1/(2pi) -> H = log(2pi) ~ 1.838 nats.
    # Convention: 0.log(0) = 0 (handled by masking).
    delta = 2.0 * math.pi / n_bins
    with np.errstate(divide='ignore', invalid='ignore'):
        log_f = np.where(hist > 0.0, np.log(hist), 0.0)
    entropy = float(-np.sum(hist * log_f) * delta)
    return entropy


def feature_p1_phase_entropy(rgb: np.ndarray) -> dict:
    """P1: Global phase entropy in the mid-frequency annulus - 2 features.

    Features emitted (2):
    p_phase_entropy_Y   - diff. entropy of luma phase in annulus;  AI > real
    p_phase_entropy_Cb  - diff. entropy of Cb phase in annulus;    AI > real
    """
    h, w = rgb.shape[:2]
    mask = _annulus_mask(h, w, r_min=5.0, r_max=100.0)

    Y  = _luma(rgb)
    Cb = _chroma_cb(rgb)

    return {
        'p_phase_entropy_Y' : _phase_entropy(Y,  mask),
        'p_phase_entropy_Cb': _phase_entropy(Cb, mask),
    }


# P2 - Radial phase circular variance over concentric rings (2 features)

def feature_p2_radial_circvar(rgb: np.ndarray,
                               n_rings: int = 8,
                               r_min: float = 5.0,
                               r_max: float = 100.0) -> dict:
    """P2: Radial phase circular variance in concentric rings - 2 features.

    Features emitted (2):
    p_radial_circvar_mean - mean of 8 per-ring circular variances;  AI > real
    p_radial_circvar_std  - std of 8 per-ring circular variances
    """
    Y = _luma(rgb)
    h, w = Y.shape

    # Phase in fftshifted coordinates
    F = npfft.fft2(Y)
    F_shift = npfft.fftshift(F)
    phase_map = np.angle(F_shift)   # (H, W) in (-pi, pi]

    # Radial distance map (fftshifted)
    cy, cx = h // 2, w // 2
    ys = np.arange(h, dtype=np.float64) - cy
    xs = np.arange(w, dtype=np.float64) - cx
    rr = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)

    delta_r = (r_max - r_min) / n_rings
    circ_vars = []

    for k in range(n_rings):
        r_lo = r_min + k * delta_r
        r_hi = r_min + (k + 1) * delta_r

        if k == n_rings - 1:
            # Last ring: inclusive upper bound to capture all annulus pixels
            ring_mask = (rr >= r_lo) & (rr <= r_hi)
        else:
            ring_mask = (rr >= r_lo) & (rr < r_hi)

        phases_k = phase_map[ring_mask]

        if phases_k.size < 2:
            logger.debug(
                "P2 ring %d: fewer than 2 pixels - setting circ_var=0.5", k
            )
            circ_vars.append(0.5)
            continue

        # Circular mean resultant length
        R = float(np.abs(np.mean(np.exp(1j * phases_k))))
        circ_vars.append(1.0 - R)

    circ_vars = np.array(circ_vars, dtype=np.float64)

    return {
        'p_radial_circvar_mean': float(np.mean(circ_vars)),
        'p_radial_circvar_std' : float(np.std(circ_vars)),
    }


# P4 - Phase-only reconstruction NCC (1 feature; weakest / confirmatory)

def feature_p4_phaseonly_ncc(rgb: np.ndarray) -> dict:
    """P4: Phase-only reconstruction NCC - 1 feature (confirmatory).

    Features emitted (1):
    p_phaseonly_ncc - z-normalised NCC between original luma and phase-only
    reconstruction;  AI -> lower (confirmatory)
    """
    Y = _luma(rgb)   # (512, 512) float64

    F     = npfft.fft2(Y)                       # complex DFT
    F_po  = np.exp(1j * np.angle(F))            # unit-magnitude phase-only spectrum
    img_po = np.real(npfft.ifft2(F_po))         # spatial reconstruction (real part)

    # Z-normalise both
    std_Y  = float(np.std(Y))
    std_po = float(np.std(img_po))

    if std_Y < _EPS or std_po < _EPS:
        logger.debug("P4: degenerate image (near-flat Y or img_po) - returning NCC=0.0")
        return {'p_phaseonly_ncc': 0.0}

    z_Y  = (Y  - np.mean(Y))  / std_Y
    z_po = (img_po - np.mean(img_po)) / std_po

    ncc = float(np.mean(z_Y * z_po))

    return {'p_phaseonly_ncc': ncc}


# Public API

def extract_phase_features(rgb_uint8_512: np.ndarray) -> dict:
    """Extract all 7 Fourier phase-spectrum features from one 512x512 RGB image."""
    feats: dict = {}
    feats.update(feature_p3_phase_congruency(rgb_uint8_512))
    feats.update(feature_p1_phase_entropy(rgb_uint8_512))
    feats.update(feature_p2_radial_circvar(rgb_uint8_512))
    feats.update(feature_p4_phaseonly_ncc(rgb_uint8_512))
    return feats
