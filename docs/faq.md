# FAQ

## What does this tool do?

You give it an image, and it tells you whether the pixels look like they came
from a camera or from an AI image generator. It answers with one of three
verdicts: likely AI-generated, likely real, or inconclusive. It also shows you
the evidence behind the verdict, panel by panel, in plain English.

## What happens to images that get uploaded?

Nothing stays on the server. An uploaded image is measured in memory and discarded as
soon as the verdict is computed. It is never written to disk, never logged and never
used for anything else. Anyone who prefers not to send an image anywhere can run the
pipeline locally from this repository, in which case nothing leaves their machine.

## Why does it sometimes say "inconclusive"?

Because sometimes the honest answer is "I can't tell", and I would rather say
that than guess. Two situations cause it: the image came from one of the
newest generators that current methods genuinely cannot detect from pixels
alone, or the image has been re-compressed and re-processed so many times that
the evidence is destroyed. In both cases the verdict degrades into "I don't
know", never into a confident wrong answer.

## What is AUC?

AUC is the standard way to score a detector, and it is easier than it sounds.
Imagine you hand the tool one random real photo and one random AI image, and
ask it which is which. AUC is the probability it ranks them correctly.

- 1.0 means it gets that comparison right every time (perfect)
- 0.5 means it does no better than flipping a coin (useless)
- 0.8 means it ranks the pair correctly 8 times out of 10

So when I say "0.88 AUC on 2022-era generators", it means: show the tool a
real photo and a 2022 AI image, and it picks the AI one about 9 times out
of 10.

## What accuracy does it actually achieve?

Depends on what generated the image, and I publish that honestly instead of
one flattering number:

- 2022-era generators: 0.88 AUC
- FLUX / Stable Diffusion family (most open models): 0.66 to 0.82
- The newest closed-lab generators (GPT-Image, Midjourney v7, Qwen and
  similar): around 0.5, coin-flip territory, which is why the tool abstains
  there
- Averaged across all 29 modern generators I tested: about 0.60

One more number matters for trust: when the tool says "likely AI-generated",
fewer than 9% of real photos ever get that verdict by mistake, and that held
across four independent photo collections. The full evaluation protocol and
per-generator tables are documented in the accompanying dissertation; this
repository ships the resulting detector.

## Other detectors advertise 95%+ accuracy. Why are your numbers lower?

Because I measure differently, not because the tool is worse.

Most published numbers come from testing on images that are similar to the
training images: same websites, same compression, same era of generators. In
that setting, a detector can score high by memorizing incidental details, like
which site the photos came from, rather than learning what AI images look
like. It's a bit like an art expert who "detects" forgeries by noticing they
all arrived in new frames. He looks brilliant in his own workshop and fails in
a real gallery.

Independent tests keep confirming this: the Deepfake-Eval-2024 study found
commercial detectors advertising 99% drop to roughly 78% on in-the-wild
images, and 2026 open-benchmark evaluations report state-of-the-art
deep-learning detectors catching only 18-42% of images from the newest
generators.

I test the hard way on purpose. The model is trained once, then evaluated
only on photos and generators from sources it has never seen. Every candidate
feature is also checked against a control that asks: "does this separate two
sets of real photos from different websites?" If it does, it is reading the
frame, not the painting, and I throw it away, even when it made my numbers
look better. Eighteen feature families died in that check; they are in the
`experiments/` folder with the reasons.

So my numbers are lower and true. When this tool says 0.88 on a generator
family, that is what you should expect on your own images, not a lab-only
best case.

## What is "de-confounding" and why did it get so much effort?

A confound is a clue that is accidentally correlated with the right answer in
your data, but has nothing to do with the real question. In this field the
data is full of them: the real photos in public datasets tend to come from
news sites and photo archives (older, resized, compressed many times), while
the AI images come fresh out of a generator (full resolution, compressed
once). A detector can score brilliantly by just noticing that difference,
without learning anything about AI images. It is the picture-frame problem
from the previous answer.

De-confounding means actively hunting those false clues down and removing
them. Concretely, every candidate measurement had to pass a control test
before being allowed into the model: take two collections of only real
photos, from two different sources, and check the measurement cannot tell
them apart. If it can, it is reading source plumbing, not AI-ness, and it
gets removed no matter how much it improved the benchmark score.

It got so much effort because it is where this kind of project quietly goes
wrong. The false clues are not rare; they are the majority of what a model
finds first, and they always make the numbers look better. Out of the
feature ideas I implemented and tested, eighteen families failed this exact
control and were rejected, several of them with benchmark scores far above
the features I kept. Skipping this step would have produced a tool with
impressive numbers that falls apart on real uploads: flagging real photos
because they were exported at full resolution, passing AI images because
someone screenshotted them. The effort is the difference between measuring
the detector and measuring the dataset.

## Does it matter where my image has been? (WhatsApp, screenshots, editing)

Where it has been: no. The tool is built so that compression history, resizing
or the website an image passed through cannot push the verdict either way.
Heavy processing can, however, erase the evidence itself, in which case you
get "inconclusive" rather than a wrong answer. See `how-it-works.md` for the
longer version.

## Can it tell me which generator made an image?

Partly. The evidence panels show which family of artifacts fired, and one
panel specifically detects a spectral pattern of Google's image generators
(Gemini/Imagen). Metadata, when present, often names the exact tool, and
that is shown in its own panel.

## Can it be fooled?

Yes, and I say so. Determined laundering (repeated recompression, heavy
edits) erases pixel evidence, and metadata can always be stripped. The newest
closed-lab generators are largely undetectable from pixels alone, for every
tool, not just this one. Treat the output as forensic evidence that supports
a judgment, not as a verdict machine.

## Which other white-box tools did you compare against?

Two are worth naming, because both are honest illustrations of the
measurement problem rather than bad work.

The strongest published white-box peer is Uhlenbrock et al. (ACM IH&MMSec
2024): hand-crafted color-texture statistics with a random forest, reporting
91% accuracy and 0.98 AUC. It is a genuinely careful study, but the
evaluation reuses the same real-photo collections on both the training and
testing side, never runs the "does this separate two real-photo sources?"
control, and does not include any of the newest generators. Its strongest
signals also live in the color channels that heavy JPEG compression flattens,
so much of that performance would not survive the test conditions used here.

The classic example is the Benford's-law DCT detector (Bonettini et al.,
2020), which reported over 99% on the GAN images of its day. I implemented
and tested that idea under my controls: it scored near coin-flip on modern
generators, and its digit statistics partly track photo content, which
differs between sources. It sits in the `experiments/` folder with the other
seventeen rejections.

The pattern is the same both times: the features are legitimate, the
headline numbers are real, but they were measured under conditions that
flatter the detector. Measured the deployment way, the numbers shrink. That
is the gap this project was built to close, and it is why the smaller
numbers here can be taken at face value.

## Why no neural networks?

Two reasons. First, explainability: every one of the 27 measurements has a
physical meaning you can state in a sentence, so the tool can show its
reasoning instead of asking to be trusted. Second, honesty: a small
transparent model makes it much harder to fool yourself about why it works,
which is the whole point of this project.
