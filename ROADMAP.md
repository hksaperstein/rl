# ROADMAP

Living status doc. Update after each completed plan (per
`.superpowers/sdd/progress.md`): append what shipped, refresh open
follow-ups below.

## Built

- **AR4 pick-and-place** (perception + RL training/eval/interactive demo) —
  working end-to-end.

## Known follow-ups

1. Shape classifier misclassifies cube/rectangular-prism as "sphere" against
   real depth data. Root-caused: `PLANARITY_RESIDUAL_THRESHOLD` (tuned on
   near-noiseless synthetic data) doesn't generalize to real sensor noise.
   Circularity looks more promising as the primary signal, but real
   tilt/plane-fit readings were also noisy on small, low-pixel-count real
   objects — may need more than a threshold nudge.
2. `interactive_demo.py` live GUI drag verification (plan Task 10, Step 4)
   was never performed — needs a human running it without `--headless` to
   confirm the physical drag → settle → pick-and-place → idle-again flow.
3. Minor/cosmetic, non-blocking: `perception/tests/conftest.py`'s
   sys.path-insert comment overstates how many directory levels it climbs;
   `interactive_demo.py` hardcodes `clip_actions=None` instead of reading it
   from agent config; a redundant filter duplicates `find_by_shape`.
4. Final whole-branch review for the perception-integration plan (Task 12)
   was explicitly skipped per user instruction — still pending whenever that
   work resumes.

## Direction

Isaac-Lab-based robotics RL, expanding beyond AR4 manipulation into other
tasks/robots, object detection/perception, and mobility. No committed
roadmap items beyond AR4 yet — this is a stated direction, not a scoped
backlog.
