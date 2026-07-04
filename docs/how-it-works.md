# How the detector works

Imagine an art expert checking whether a painting is a forgery.

A lazy expert might notice that every forgery he has ever been shown arrived in
a shiny new frame, while the originals came in dusty old ones. So he starts
judging by the frame. In his workshop he looks brilliant, 95% correct. Then he
goes to a real gallery, someone brings an original that was recently reframed,
and he confidently calls it a fake. He was never detecting forgery. He was
detecting frames.

My detector is trained like the careful expert. During development I
repeatedly showed it only genuine photos that arrived in different "frames":
from a news site, from a phone, from a photo archive. If any clue it uses
reacts to the frame instead of the painting, I throw that clue out, even when
it made my results look better. What is left are clues that live in the paint
itself: the microscopic brush strokes that AI generators leave in the pixels.

That is why, when you upload a photo, the detector does not care where it has
been. WhatsApp, Instagram, screenshots, ten years on a hard drive. None of that can
make your real photo look "AI" to it, or make an AI image look real.

One honest limit: if an image has been through very heavy processing, it is
like a painting that has been scrubbed and re-varnished. The brush strokes get
smudged, and no tool can invent evidence that has been destroyed. In that case
it will not guess: it tells you honestly, "inconclusive."

When it does say "likely AI-generated", it is because the brush strokes are
there, and fewer than 1 in 10 genuine photos ever triggers that verdict by
mistake.
