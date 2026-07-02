# AI image forensics

Detects AI-generated images from pixel statistics alone. No neural networks:
27 hand-crafted signal-processing features (NumPy/SciPy) feeding a logistic
regression, plus a metadata/provenance check. Every part of the decision is
inspectable and explainable.

## How it decides

Each upload is center-cropped to 512px, re-encoded to a fixed JPEG substrate,
and measured for generator artifacts: VAE upsampling grids, spectral shape,
missing camera-sensor physics, local self-consistency, and a spectral
signature specific to Google's image generators. The combined score is
compared against a reference distribution of real photos from several
independent sources, and the answer comes back in one of three bands:

- **likely AI-generated**: beyond the 95th percentile of real photos. Fewer
  than 9% of real photos land here across my test corpora.
- **inconclusive**: the evidence is not sufficient either way. This is a
  deliberate answer, not a failure mode.
- **likely real**: well inside the real-photo envelope.

Provenance metadata (C2PA manifests, IPTC AI markers, generator EXIF tags,
Stable Diffusion PNG chunks) is reported as a separate panel. It is never
mixed into the pixel score, since metadata is trivial to strip or forge.

## What it can and cannot detect

Measured on held-out data from corpora the model never trained on:

| Generator family | AUC | status |
|---|---|---|
| 2022-era generators | 0.88 | covered |
| FLUX / Stable Diffusion lineage | 0.66-0.82 | covered |
| Seedream, Kling, Hunyuan, recent partial tier | 0.55-0.65 | partial |
| Google family (Gemini, Imagen) | 0.55-0.70 | evidence panel |
| GPT-Image, Midjourney v7, Qwen, GLM | ~0.5 | not detectable from pixels |

The last row is a property of the field, not just this tool: state-of-the-art
detectors, including deep-learning ones, perform far below their advertised
numbers on these generators in independent tests. This tool abstains there
instead of guessing. Heavy re-processing (social media pipelines) degrades
detection for all rows; provenance metadata may still identify such images.

All evaluation is frozen and cross-corpus: the model is trained once, then
tested only on data from sources it has never seen, with per-generator
breakdowns and real-vs-real checks to rule out dataset shortcuts. See
`docs/how-it-works.md` for the plain-English version.

## Usage

Requires Python 3.12.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pipeline.predict path/to/image.jpg
```

Returns JSON: verdict, real-photo percentile, per-channel evidence panels,
provenance report, and limitations. Images under 512px on the short side are
rejected rather than upscaled (upscaling would destroy the statistics being
measured).

```python
from pipeline.predict import Predictor
result = Predictor().predict("image.jpg")
```

## Layout

    pipeline/predict.py       prediction entry point
    pipeline/preprocess.py    crop + JPEG substrate
    pipeline/provenance.py    metadata evidence channel
    pipeline/features/        feature extractors (pure NumPy/SciPy)
    models/                   frozen model bundle + spectral templates
    tests/                    unit tests
    experiments/              feature families tried and rejected, with reasons
