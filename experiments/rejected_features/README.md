# Rejected feature experiments

Feature families that were implemented and evaluated but did not make it into
the model. Kept for reference; nothing here is imported by the pipeline.

Every candidate had to pass the same three checks before adoption:

1. Frozen transfer: train once, then predict on data from a source the model
   has never seen. Cross-validation numbers alone were never trusted.
2. Real-vs-real control: the feature must NOT separate two sets of real photos
   from different sources. If it does, it is reading acquisition history
   (compression, resizing, site pipeline) instead of generation artifacts, and
   it will misfire on user uploads.
3. No regression on generators the model already handles.

Most of the rejections below failed check 2. The recurring lesson: the more
impressive a feature looked in cross-validation, the more likely it was
measuring the dataset rather than the image.

| Module | Idea | Why rejected |
|---|---|---|
| benford_dct.py | first-digit statistics of DCT coefficients | weak effect on modern generators; digit distributions track content density (fails control 2) |
| color_distribution.py | color statistics across colorspaces | strongly separates real corpora from each other (fails control 2) |
| dark_channel.py | haze / dark-channel prior of real light transport | signal destroyed by the JPEG substrate; residual was source-coupled |
| defocus_dof.py | depth-of-field and defocus gradients | worked on a single generator only, unstable across sources |
| grid_period.py | grid periodicity at non-standard pitches | chance-level: q75 compression erases faint decoder grids at other periods |
| interactions.py | polynomial interactions of base features | no held-out gain over the linear model |
| jpeg_block.py | JPEG blockiness measures | reads compression history, the textbook shortcut (fails control 2) |
| lighting_coherence.py | lighting direction consistency | clean on the controls but no detectable signal |
| local_corr.py | local cross-channel correlation fields | separates real corpora (fails control 2) |
| noiseprint.py | sensor noise-print residual | fingerprints the ISP/source pipeline, not AI-ness (fails control 2) |
| npr.py | local pixel residual (down/up resampling) | borderline on control 2 and redundant with the adopted grid features |
| perspective_geom.py | perspective / vanishing-point coherence | apparent lift was inconsistent across real corpora |
| phase_spectrum.py | Fourier phase statistics, phase congruency | no held-out gain |
| physics_channel.py | physics-grounded channel correlations | color axis is source-coupled (fails control 2) |
| residual_glcm.py | texture co-occurrence of noise residuals | redundant with adopted families; shifts under compression |
| scene_nss.py | natural scene statistics | heavily corpus-coupled (fails control 2) |
| spectral_fractal.py | fractal dimension and spectral shape variants | redundant with the adopted spectrum features |
| white_balance.py | illuminant / white-balance estimation | color-based, source-coupled (fails control 2) |
