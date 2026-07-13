<!--
  Ready-to-paste additions for src/content/projects/dice-detection.md.
  Continues "Step 4: Training and Sim-to-Real Results" with the
  datagen-v2 iteration that specifically targeted the d8/d10 collapse
  documented there. I'd suggest adding this as a new "## Step 5" after
  the existing Step 4 content.

  Also includes a new gallery section and an inline SVG bar chart
  (already built and self-contained in site/index.html — copy the
  <svg>...</svg> block from the "detection-fix" section of that file,
  or re-render it as a static image if the Astro Gallery component
  can't host raw inline SVG cleanly).
-->

## Frontmatter gallery additions

```yaml
  - section: "Sim-to-Real Results — datagen-v2"
    file: "/assets/images/projects/dice-detection/d10-before-s-mislabeled-d12.jpg"
    description: "A real d10 photo, boxed correctly but labeled d12 at 0.84 confidence by the original synthetic-only model — localization was never the problem, the class head was reading apparent size as a class cue."
  - section: "Sim-to-Real Results — datagen-v2"
    file: "/assets/images/projects/dice-detection/d10-after-s_v2-correct.jpg"
    description: "The same photo, same die, correctly labeled d10 at 0.86 confidence after adding a close-up training slice that decouples apparent size from class."
```

## Prose section — "Step 5: datagen-v2 — fixing the d8/d10 collapse"

```markdown
## Step 5: datagen-v2 — fixing the d8/d10 collapse

Step 4 found a specific failure: on real photos, d8 and d10 kept getting
called d12 or d20. My working hypothesis was apparent size — the synthetic
scenes render dice at realistic relative sizes in multi-die tabletop shots,
so size correlates with class, and the real test photos are almost all
single-die close-ups where every die looks big. I wrote down two thresholds
before testing it: d8/d10 real mAP50 clearing 0.40 (primary), d12/d20 not
regressing (guard).

I added a 3,000-image close-up slice to training with camera distance
decoupled from class, so every die type shows up at every apparent scale,
not just its physically realistic one. Nothing else in the generator or
training recipe changed.

| class | S (before) | S_v2 (after) |
|---|---|---|
| d4  | 0.695 | 0.784 |
| d6  | 0.519 | **0.275** |
| d8  | 0.090 | 0.442 |
| d10 | 0.097 | 0.534 |
| d12 | 0.936 | 0.946 |
| d20 | 0.855 | 0.907 |

Both thresholds passed: d8 0.090 → 0.442, d10 0.097 → 0.534, both past 0.40;
d12/d20 held or improved slightly. That's real support for the size-confound
hypothesis, not a number that happened to move.

It also cost me d6, which the guard didn't cover: 0.519 → 0.275. I looked
into it instead of writing it off. The confusion matrices show d6's own
false-positive rate actually fell — the problem is recall, not precision.
d10 started absorbing d6's detections: d6→d10 confusion rose from 6% to
50%, while d6's old dominant error (getting called d20) mostly disappeared.
d10 got more aggressive at claiming ambiguous cases once it stopped losing
so many of its own to d12/d20, and d6 lost that fight. That's the open item
for the next data-generation pass — I'm not folding it into a clean win.

Full per-class tables and the confusion-matrix breakdown:
`vision/docs/results/2026-07-dice-detector-v1/eval_s_v2.md`,
`vision/docs/results/2026-07-dice-detector-v1/d6_regression_analysis.md`.
```
