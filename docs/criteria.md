# Evaluation criteria (paper structural pass)

This document states the pass / match rules used for Tables 1–5 and related text.
Internal Chinese revision notes (`S_STRUCT_*.md`) are **not** released.

## Fringe spacing \(\Lambda\)

For each frame, \(\Lambda\) is the median adjacent spacing of the human centerline radii (pixels).
(Legacy JSON fields may still say `err_s` / `within_1s`; those mean “in units of \(\Lambda\)”, not seconds.)
Tolerances below are multiples of this frame-wise \(\Lambda\).

## Pole tolerances (four reported gates)

A predicted pole \((x,y)\) passes the pole gate when its Euclidean distance to the human pole is at most:

| Gate | Tolerance |
|------|-----------|
| 0.5 \(\Lambda\) | primary / strict |
| 1.0 \(\Lambda\) | open-arc–robust |
| 2.0 \(\Lambda\) | loose |
| untested | pole not tested; structural ring-list only |

## Ring matching window

Cast rays from the **human** pole along directions sampled on the annotated arc.
Take the median forward intersection radius of the predicted circle with those rays.
A predicted circle matches an annotated dark band when that median lies within

\[
\max(0.4\,\Lambda,\; w_{\mathrm{half}} + 2\,\mathrm{px})
\]

of the band centerline, where \(w_{\mathrm{half}}\) is the annotated half-width.
Matching is one-to-one (smallest residual first).
An arc that misses the circle on more than half of its rays does not match.

Circles that lie entirely inside the bright central hole (inward of the first band by 2 px) are removed and count neither as matches nor as extras.

## Structural pass (joint with a chosen pole gate)

After hole removal, **all** of the following must hold:

1. Pole distance \(\le\) the chosen gate (unless “untested”).
2. At least 3 predicted circles remain (otherwise **unformed**; stays in the denominator).
3. At least 3 matched bands.
4. Structural misses \(\le \lfloor 0.3 \times N_{\mathrm{struct}} \rfloor\) (see dense-fringe rule).
5. Unmatched predicted circles \(\le \lfloor 0.3 \times N_{\mathrm{pred}} \rfloor\).
6. Matched bands are strictly increasing in intersection radius from inside out.

Lens curvature \(R\) is **not** part of the pass line.

## Dense-fringe (fine-spacing) rule

Local spacing of a band = min distance to its neighbours.
Let \(\Lambda_{\mathrm{clear}}\) be the median local spacing of the inner half of bands.
Bands with local spacing \(< 0.75\,\Lambda_{\mathrm{clear}}\) are **dense**; misses on dense bands are recorded but do not consume the 30% miss budget.
If fewer than 3 structural (non-dense) bands exist, the dense relaxation is disabled.

## Arc-band overlap (in-band fraction)

Reported separately from the pass line.
Each annotated band is sampled every 2° over its annotated angular span.
A sample is in-band when \(|r_{\mathrm{fit}}-r_{\mathrm{ann}}|\) does not exceed the local half-width.
Per-ring fractions are averaged with equal ring weight; unpaired rings contribute zero.

## Matched-order recall

Fraction of annotated rings that obtain a one-to-one match under the matching window above (reported as matched-ring fraction in the paper).
