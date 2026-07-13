# d6 real-test regression analysis (variant `s` → `s_v2`)

**Date:** 2026-07-13
**Trigger:** `docs/superpowers/specs/2026-07-11-datagen-v2-closeup-design.md`,
"Verdict (2026-07-13)" section — d6 real-test mAP50 dropped 0.519 → 0.275
when the datagen-v2 close-up slice was added to training, while every other
class improved or held. This note is the dedicated analysis that section
called for before any iteration-3 fix is designed.

**Method summary:** CPU-only inference (`device="cpu"`, no GPU allocated,
`/tmp/rl_isaac_sim.lock` untouched) with `vision/.venv/bin/python` and both
`models/runs/s/weights/best.pt` and `models/runs/s_v2/weights/best.pt`, over
the full frozen real test set (1,376 images, `data/real/test`), at the same
conf=0.25 threshold `scripts/evaluate.py` uses. Predictions and GT were
matched by greedy IoU≥0.5 (highest-confidence prediction first, one-to-one),
mirroring the matching convention used for the v1 confusion analysis in
`docs/results/2026-07-dice-detector-v1/summary.md`. No generator or training
code was modified; nothing was retrained.

## 1. Confusion tables

### Variant `s` (rows = true class, cols = predicted class, IoU≥0.5)

| true\pred | d4 | d6 | d8 | d10 | d12 | d20 | missed |
|---|---|---|---|---|---|---|---|
| d4 | 114 | 0 | 0 | 0 | 0 | 14 | 18 |
| d6 | 9 | **183** | 5 | 23 | 11 | 68 | 69 |
| d8 | 16 | 1 | 23 | 4 | 0 | 58 | 58 |
| d10 | 2 | 6 | 0 | 29 | 59 | 75 | 44 |
| d12 | 0 | 0 | 0 | 0 | 204 | 0 | 7 |
| d20 | 0 | 0 | 0 | 0 | 1 | 254 | 21 |
| background (FP source) | 53 | 64 | 70 | 79 | 59 | 314 | — |

### Variant `s_v2`

| true\pred | d4 | d6 | d8 | d10 | d12 | d20 | missed |
|---|---|---|---|---|---|---|---|
| d4 | 121 | 0 | 12 | 1 | 0 | 0 | 12 |
| d6 | 3 | **93** | 23 | **183** | 1 | 1 | 64 |
| d8 | 12 | 0 | 84 | 15 | 0 | 15 | 34 |
| d10 | 1 | 0 | 1 | 174 | 4 | 4 | 31 |
| d12 | 0 | 0 | 0 | 2 | 203 | 0 | 6 |
| d20 | 0 | 0 | 0 | 1 | 1 | 258 | 16 |
| background (FP source) | 30 | 48 | 44 | **220** | 10 | **19** | — |

### Precision / recall at conf=0.25, IoU≥0.5 (fixed-threshold snapshot — complements, doesn't replace, the mAP50 headline numbers)

| class | s recall | s precision | s_v2 recall | s_v2 precision |
|---|---|---|---|---|
| d4 | 0.781 | 0.588 | 0.829 | 0.725 |
| d6 | 0.497 | 0.720 | **0.253** | 0.660 |
| d8 | 0.144 | 0.235 | 0.525 | 0.512 |
| d10 | 0.135 | **0.215** | 0.809 | **0.292** |
| d12 | 0.967 | 0.611 | 0.962 | 0.927 |
| d20 | 0.920 | 0.324 | 0.935 | **0.869** |

### d6 breakdown, `s` → `s_v2`

| GT d6 → | s | s_v2 |
|---|---|---|
| d6 (correct) | 183 (49.7%) | 93 (25.3%) |
| d10 | 23 (6.2%) | **183 (49.7%)** |
| d20 | 68 (18.5%) | 1 (0.3%) |
| d8 | 5 (1.4%) | 23 (6.2%) |
| d4 | 9 (2.4%) | 3 (0.8%) |
| d12 | 11 (3.0%) | 1 (0.3%) |
| missed | 69 (18.8%) | 64 (17.4%) |

**Reading these tables together:** d6's own recall collapsed almost
entirely because of one specific reassignment — GT d6 → predicted d10 rose
from 6.2% to 49.7%, while GT d6 → predicted d20 (`s`'s dominant error mode,
matching the v1 summary's "shape-complexity-ladder" finding) collapsed from
18.5% to 0.3%. d6's own false-positive rate (background → d6) *fell*
(64→48); this is not a d6 precision problem, it is a d6 recall problem
caused by one specific competing class.

That competing class, d10, did not just absorb d6 — it absorbed
background clutter network-wide: background false positives labeled d10
rose 79→220, while background false positives labeled d20 (the dominant
"default guess" class in `s`) collapsed 314→19. d10's own recall jumped
13.5%→80.9% (matching the spec's headline mAP50 win), but its precision
barely moved (0.215→0.292) despite six times more mass being predicted as
d10 (135→596 total d10 predictions). **d10 became the new default/attractor
class for ambiguous evidence, replacing d20's role in variant `s`** — d6 is
the specific victim with the largest overlap with d10's decision boundary,
but the effect is general, not d6-exclusive (d8 and background also bleed
into d10 more than before, just by smaller absolute amounts).

## 2. Size/framing distribution comparison

Bounding-box height as a fraction of image height (same metric the spec's
own "Real framing measurement" section used), computed directly from each
source's own annotations (`data/detection_v1/coco.json`,
`data/detection_v2_closeup/coco.json`, `data/real/test/labels/*.txt`):

| source | class | n | p5 | p25 | median | p75 | p95 |
|---|---|---|---|---|---|---|---|
| detection_v1 (train) | d6 | 7,526 | 0.092 | 0.143 | 0.191 | 0.250 | 0.365 |
| v2_closeup (train) | d6 | 651 | 0.256 | 0.430 | **0.579** | 0.729 | 0.947 |
| real test (GT) | d6 | 368 | 0.382 | 0.487 | **0.539** | 0.604 | 0.688 |

**This kills the spec's original scale-sensitivity lead as stated.** The
close-up slice's own d6 examples are centered almost exactly where real d6
photos sit (median 0.579 vs 0.539; the real IQR 0.487–0.604 sits well
inside the closeup slice's IQR 0.430–0.729) — d6 was not undersampled at
the apparent size real photos show it. Contrast with d8/d10, whose
closeup-slice medians (0.330 / 0.362) also land close to their own real
medians (0.512 / 0.469) and whose mAP50 improved by 0.35–0.44; d6's slice
coverage at the real scale is *comparably good*, yet d6 is the one class
that regressed. Per-class scene counts in the closeup slice (651 d6
instances out of 3,000 images, vs. 590–692 for every other class) show no
undersampling either — d6 is squarely mid-pack.

A finer check: within `s_v2`'s matched d6 detections, does the d6→d10
misclassification concentrate in the largest real d6 boxes (i.e., is there
still a residual "not quite large enough" tail effect)? No — GT boxes that
got relabeled d10 have almost the same size distribution (mean height-frac
0.548, n=183) as GT boxes that stayed correctly labeled d6 (mean 0.544,
n=93). The confusion is spread uniformly across the observed real-size
range, not concentrated at any scale extreme.

**Glyph-style convention, a pre-existing (not v1→v2_closeup-introduced)
imbalance worth surfacing:** d10's synthetic training pool is
overwhelmingly `arabic_numerals` (77.5% in v1, 79.3% in v2_closeup — by
design, since the domain config only allows d10 the `arabic_numerals`/
`cjk_numerals` glyph pair). Every other polyhedral class, including d6,
draws roughly evenly across `arabic_numerals`/`roman_numerals`/
`greek_numerals`/`cjk_numerals` (d6: ~27–28% arabic in both v1 and
v2_closeup — unchanged between the two, so this ratio itself is not what
changed). Manually reviewed real d6 photos (`data/real/test/images/dd_dice_d6_top_*`,
`dnd_dices_d6_*`) are effectively 100% plain arabic-digit numerals. So real
d6's glyph convention matches d10's training convention (heavily
arabic-numeral) far more closely than it matches d6's *own* training
convention (~72% non-arabic). This asymmetry existed in variant `s` too,
but at `s`'s small apparent training scale for d6 (median 0.191) the glyph
was rarely legible enough to be a usable cue at all — see section 3 for
qualitative confirmation that legible glyph only becomes a decisive
learned cue once the close-up slice makes it consistently readable during
training.

## 3. Qualitative evidence

Seeded (42), stratified sample (small/medium/large tercile of real d6 GT
box height-fraction, 4 each = 12 images) — both models' predictions
side-by-side with GT, saved to
`models/eval/d6_regression_analysis/{small,medium,large}_<image-stem>.jpg`.

| image | tercile | s prediction | s_v2 prediction | outcome |
|---|---|---|---|---|
| `dnd_dices_d6_wood0018` | small | d4@0.53 (wrong) | d6@0.94 (correct) | improved |
| `dd_dice_d6_top_0387` | small | d6@0.95 (correct) | d10@0.89 (wrong) | **regressed** |
| `dd_dice_d6_top_0947` | small | d10@0.68 (wrong) | d10@0.89 (wrong) | unchanged wrong |
| `dnd_dices_d6_top_0436` | small | d10@0.51 / d6@0.28 (wrong-leading) | d10@0.78 (wrong) | unchanged wrong |
| `dnd_dices_d6_wood1103` | medium | d20@0.58 (wrong) | d10@0.51 / d6@0.51 (tie) | ambiguous |
| `dnd_dices_d6_wood0034` | medium | d4@0.54 (wrong) | d6@0.89 (correct) | improved |
| `dnd_dices_d6_color337` | medium | d20@0.77 (wrong) | d10@0.89 (wrong) | unchanged wrong |
| `dnd_dices_d6_color032` | medium | d10@0.75 (wrong) | d10@0.88 (wrong) | unchanged wrong |
| `dnd_dices_d6_45angle_0452` | large | d20@0.92 (wrong) | d6@0.80 (correct) | improved |
| `dnd_dices_d6_45angle_1009` | large | d6@0.93 (correct) | d10@0.93 (wrong) | **regressed** |
| `dnd_dices_d6_off-angle_557` | large | d6@0.98 (correct) | d6@0.67 (correct, lower conf) | unchanged correct |
| `dnd_dices_d6_off-angle_136` | large | d20@0.90 (wrong) | d10@0.74 (wrong) | unchanged wrong |

n=12 is too small to reproduce the aggregate mAP50 drop by itself (this
sample happens to net out roughly flat, 3/12→4/12 correct) — it is included
for qualitative inspection, not as a second quantitative estimate. The one
pattern that holds with no exception across all 12: **every wrong s_v2
prediction is d10 or d10_pct.** Never d20, d12, d4, or d8. In `s`, wrong
predictions were d20 (4 cases), d4 (2 cases), or d10 (2 cases). This
single-attractor-class pattern in the hand-reviewed sample matches the
full-dataset confusion table in section 1 exactly.

**A framing/pose-mismatch conjecture was checked and is not supported.**
Initial visual review suggested real d6 photos split into two rough
framing styles: near-planar single-face top-down crops (`dd_dice_d6_top_*`
filenames) vs. oblique views showing 2–3 cube faces (`*_45angle_*`,
`*_off-angle_*`, `*_wood*`, `*_color*` filenames) — and that the synthetic
generator's camera elevation range (15°–65°, unchanged between v1 and the
closeup slice) can produce the oblique multi-face look but never the
near-vertical top-down single-face look. The conjecture was that the
top-down subset, having no matching pose in training, would show the worst
regression. A full breakdown of GT-d6→prediction by filename-inferred
framing subgroup (regex heuristic, not a verified per-image camera-angle
label) contradicts this:

| subgroup (filename heuristic) | s correct rate | s_v2 correct rate | s_v2 d10-flip rate |
|---|---|---|---|
| top-down (`*_top*`) | 59.3% (51/86) | 23.3% (20/86) | 50.0% (43/86) |
| 45°/off-angle oblique | 52.1% (63/121) | 27.3% (33/121) | **64.5% (78/121)** |
| "wood" material subset | 38.8% (45/116) | 31.9% (37/116) | 31.9% (37/116) |
| "color" material subset | 55.8% (24/43) | 7.0% (3/43) | 58.1% (25/43) |

The regression is broad across every framing subgroup, and the oblique
multi-face subgroup — the one whose pose *does* have a training analogue —
is actually the worst-hit by the d10 flip, not the best. The framing/pose
mismatch is not the mechanism; dropping it.

## 4. Candidate hypotheses for datagen iteration 3

**H1 (primary, best supported): d10's glyph-convention imbalance became a
decisive, over-generalizing cue once scale stopped separating classes.**
Decoupling apparent size from class (the closeup slice's whole point)
removed "medium size" as d10's separator from d12/d20, and removed "small
size" as the reason d10's own glyph convention (77–79% arabic numerals,
a domain-config-level restriction unique to d10 among the polyhedral
classes) was never actually legible during `s`'s training. Once the
closeup slice made d10 legible at training time, the model learned
something close to "a legible, plain arabic-numeral digit centered on a
compact faceted object, at real-photo scale" as d10's dominant decision
rule — a rule real d6 photos (near-100% plain arabic numerals, unlike d6's
own ~28%-arabic synthetic pool) satisfy about as well as real d10 photos
do.

*Evidence:* d6→d10 confusion rose from 6.2%→49.7% with no size-tail
concentration (section 2); every wrong s_v2 prediction in the qualitative
sample is d10/d10_pct with no exception (section 3); d10's glyph mix is the
one systematically different convention among the polyhedral classes and
was already true in variant `s`, only becoming actionable once scale
stopped masking it.

*Falsifiable prediction:* rebalancing d10's synthetic glyph-style
distribution toward the same ~25–29% arabic-numeral split the other
classes already use (or, conversely, increasing d6's arabic-numeral share
toward the real-world-representative near-100% target — the v1 summary's
recommendation 3, still not done) should measurably reduce d6→d10
confusion in a rerun, without giving back d10's own real-test gains.
*Falsification condition:* if d6→d10 confusion does not drop after that
rebalance, the glyph-convention-imbalance mechanism is wrong and the
broader calibration/attractor account (H2) should be treated as the
primary explanation instead.

**H2 (secondary, also supported, more general than H1): the closeup slice
overshot d10's decision-boundary size, making it a general default guess
for ambiguous evidence, not a d6-specific effect.** d10 recall jumped
13.5%→80.9% while precision barely moved (0.215→0.292, on 6x more total
d10 predictions); background false positives shifted from d20 (314→19) to
d10 (79→220) network-wide. d6 is the largest specific casualty because it
overlaps d10's new decision boundary most (H1's mechanism), but d8's
background bleed also grew slightly, consistent with a broader
under-constrained d10 boundary rather than a purely d6-specific
phenomenon.

*Falsifiable prediction:* a hard-negative pass that explicitly penalizes
d10 predictions on non-d10 ground truth (including background) should
recover most of d6's lost recall without materially hurting d10's own
mAP50 gain. *Falsification condition:* if d6 recall stays flat despite
that intervention while d10's background-FP rate drops, the effect is not
a generic calibration/attractor issue and is specific to d6↔d10 visual
similarity (glyph or otherwise) in a way H1's glyph fix should be
retried/refined for, rather than H2's more general hard-negative fix.

**Leads explicitly ruled out by this analysis:** the spec's original
scale-sensitivity-of-cues lead (section 2 — d6 is not undersampled at the
real-matching apparent size, and confusion doesn't correlate with size
within the observed range); a cube-specific single-face top-down
framing/pose mismatch (section 3 — regression is broad across all framing
subgroups, worst in the subgroup whose pose *does* have a synthetic
training analogue, not the one that doesn't).

## Artifacts

- Confusion/size-distribution scripts (not committed, scratch):
  `/tmp/claude-1000/-home-saps-projects-rl/9dd864ae-7c39-4467-97cb-e04bf4dd58a9/scratchpad/{confusion_analysis.py,size_dist.py,qualitative_d6.py}`
- Qualitative side-by-side images: `models/eval/d6_regression_analysis/` (12 images, gitignored like other `models/eval/` artifacts)
- Raw per-image prediction dump used for the subgroup/size cuts:
  `/tmp/claude-1000/-home-saps-projects-rl/9dd864ae-7c39-4467-97cb-e04bf4dd58a9/scratchpad/per_image_results.json` (scratch, not committed)
