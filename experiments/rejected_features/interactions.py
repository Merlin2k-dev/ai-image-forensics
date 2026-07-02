"""Pairwise interaction terms of the base features.

Pairwise products of base features, testing whether a linear model plus
hand-made interactions could capture curvature a plain linear model misses.

Rejected: no held-out gain over the linear model on the base features.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


# Interaction specification table
# Each entry: (output_col_name, col_A, col_B, one-line mechanistic reason)
INTERACTION_SPECS = [
    (
        'p_cdsnr16_x_noisecov',
        'f2_cd_snr_k16',
        's_noise_cov',
        'AI = 16px diagonal grid peak (FLUX/SD3.5) and content-correlated noise; '
        'real = no grid and uniform sensor noise.',
    ),
    (
        'p_gausssnr16_x_noisecov',
        'f3_gauss_snr_k16',
        's_noise_cov',
        'Same conjunction in the Gaussian-residual grid domain.',
    ),
    (
        'p_slope_x_noisecov',
        'f4_spectral_slope',
        's_noise_cov',
        "FLUX's broadband spectral deficit modulated by noise spatial structure.",
    ),
    (
        'p_slope_x_skew',
        'f4_spectral_slope',
        's_resid_skew',
        'FLUX rides spectral slope, SD3.5 rides residual skew; product probes whether '
        'ONE linear direction captures both eras where marginals diverge.',
    ),
    (
        'p_dctexc16_x_skew',
        'f5_dct_excess_k16',
        's_resid_skew',
        'DCT 16px grid excess with residual non-Gaussianity.',
    ),
    (
        'p_cdsnr16_x_skew',
        'f2_cd_snr_k16',
        's_resid_skew',
        'Grid peak with residual asymmetry.',
    ),
    (
        'p_gausssnr16_x_slope',
        'f3_gauss_snr_k16',
        'f4_spectral_slope',
        'Conjunction of two VAE-side signals (grid SNR AND broadband deficit).',
    ),
]

# Canonical ordered list of interaction column names (exactly 7)
INTERACTION_NAMES: list = [spec[0] for spec in INTERACTION_SPECS]

# All source columns required in the input DataFrame
_REQUIRED_SOURCE_COLS = sorted({
    col
    for _, a, b, _ in INTERACTION_SPECS
    for col in (a, b)
})


def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 7 pairwise interaction features from a merged DataFrame."""
    # Guard: verify all required source columns are present
    missing = [c for c in _REQUIRED_SOURCE_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"build_interactions: required source columns missing from input: {missing}. "
            "Check that the base feature tables were correctly merged."
        )

    out = pd.DataFrame(index=df.index)
    out['generator'] = df['generator']
    out['label']     = df['label']

    for out_col, col_a, col_b, reason in INTERACTION_SPECS:
        out[out_col] = df[col_a].values * df[col_b].values
        logger.debug(
            "Computed %s = %s * %s  [%s]",
            out_col, col_a, col_b, reason,
        )

    logger.info(
        "build_interactions: produced %d interaction columns for %d rows.",
        len(INTERACTION_NAMES),
        len(out),
    )
    return out
