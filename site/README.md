# site/ — content pack for the portfolio site agent

Demos + data for `hksaperstein.github.io`, to be integrated by that
repo's own agent. (An earlier revision of this bundle included a
standalone dashboard page; removed at Harrison's request — this is
now content only.)

What's here:
- `data.json` — machine-readable manifest: every video/image with
  duration/size/caption/provenance/tag, plus the headline stats with
  their repo sources. Start here.
- `assets/` — 8 processed clips (H.264, yuv420p, +faststart, all
  ffprobe-verified) + poster frames + before/after detection images.
- `content/franka-dice-pick-additions.md`,
  `content/dice-detection-additions.md` — pasteable markdown for the two
  existing project pages, already written first-person in VOICE.md
  register.

Numbers all trace to sources in the public repo
(github.com/hksaperstein/rl, branch `franka-panda-pivot`); every claim's
source file is named in `data.json`.
## Why additions, not a rewrite

Discovered mid-task that a real portfolio already exists at
`hksaperstein.github.io` (Astro + Tailwind) with two project pages that
already cover a meaningful slice of this repo's work — the dice-pick demo
gallery (5 videos, already captioned) and the dice-detector's v1 sim-to-real
results (assets, geometry, YOLO training, the S vs S+R failure). Duplicating
that content here would either go stale immediately or fight the live site
for authority. So this bundle does two things instead:

it carries only what the live pages don't have yet: pasteable markdown
additions (`content/`), the processed media (`assets/`), and a manifest
with every number's source (`data.json`).

## What's genuinely new (not on either live page yet)

Both live pages predate these results:

- **RL joint-space die-lift + the asset-bisect ladder** (2026-07-12/13).
  The Franka lift recipe failed completely on the real-size (30.3mm) d20,
  the DexCube fallback proved the recipe itself was sound, and a 3-variable
  bisect (mass / size / shape, 3 seeds each) found shape is the actual gate
  on grasp discovery, with size as a severity modifier and mass/pipeline
  both cleared. First-ever learned die lift happened en route (asset-bisect
  rung 2, seed 123).
- **datagen-v2** (2026-07-13): the d8/d10 sim-to-real collapse documented
  on the live dice-detection page got a targeted fix (decoupling apparent
  size from class in a new close-up training slice), verified against two
  pre-registered thresholds — both passed — plus an honest d6 regression
  the thresholds didn't cover.
- **Cloud training pipeline proven** (2026-07-13): first real GCP GPU run,
  full lifecycle exercised, under $1 total.

## Integration map

| Content | Target | Action |
|---|---|---|
| `content/franka-dice-pick-additions.md` (3 new RL videos + asset-bisect writeup) | franka-dice-pick.md | **Graft.** New gallery entries ("RL lift experiments") + a new prose section; suggested placement: after "What Isaac Lab taught me the hard way", before "The d4". |
| `content/dice-detection-additions.md` (datagen-v2 results + before/after image pair) | dice-detection.md | **Graft.** New gallery entries ("Sim-to-Real Results — datagen-v2") + a new "## Step 5" prose section. Per-class mAP50 numbers for both variants are in the markdown; re-render a chart from them if wanted (sources in `data.json`). |
| 5 dice-pick clips (`dice-pick-d*.mp4`) | — | **No action needed** — the live page already has all 5 with captions; included here only so the bundle is complete. |
| `data.json` stats | agent's discretion | Headline numbers with sources, e.g. for stat cards or intro copy. |

## Asset paths

Videos and images already use the live site's own path convention
(`/assets/videos/projects/<slug>/...`, `/assets/images/projects/<slug>/...`)
so they can be copied directly into `hksaperstein.github.io/public/assets/`
with the leading `site/assets/` stripped — no rename needed. The 5
dice-pick clips/posters in this bundle are re-encoded from the same source
files (`outputs/dice_demo/gate_v/dice_pick_*.mp4`) already live in that
repo's `public/assets/videos/projects/franka-dice-pick/`; if those are
already current there's no need to copy this bundle's copies over them.

## Video inventory (all verified with `ffprobe` + a frame extract)

| file | source | duration | size | what it shows |
|---|---|---|---|---|
| `dice-pick-d20.mp4` | `outputs/dice_demo/gate_v/dice_pick_d20.mp4` | 18.2s | 656KB | Commanded d20, pass, z-gain 237.1mm — used as the page hero |
| `dice-pick-d10.mp4` | `outputs/dice_demo/gate_v/dice_pick_d10.mp4` | 18.1s | 560KB | Commanded d10, pass, z-gain 239.3mm — used in the gallery |
| `dice-pick-d4.mp4` | `outputs/dice_demo/gate_v/dice_pick_d4.mp4` | 18.1s | 656KB | Commanded d4, **fail** (permitted), z-gain ~0.3mm — used in the gallery |
| `dice-pick-d8.mp4`, `-d12.mp4` | same dir | 17.3–17.7s | 412–504KB | Re-encoded for completeness — all 5 die types available in this bundle |
| `rl-joint-die-d20-falsified.mp4` | `logs/videos/franka_checkpoint_review/franka_checkpoint_review_joint-die_model_1499-step-0.mp4` (checkpoint `logs/train_franka_jointdie/2026-07-12_06-56-02/model_1499.pt`) | 5.0s | 312KB | d20 at real 30.3mm size, 0/8 sustained lifts |
| `rl-first-die-lift-rung2-seed123.mp4` | `..._joint-die-big_model_1499-step-0.mp4` (checkpoint `logs/train_franka_jointdiebig/2026-07-12_15-07-49/model_1499.pt` — confirmed via the matching `heights_joint-die-big_model_1499.json`, this is specifically the seed-123 run) | 10.0s | 384KB | d20 at 48mm, 8/8 sustained lifts — first learned die lift |
| `dice-line-pick-and-place.mp4` (2026-07-21) | `outputs/dice_demo/ik_dice_line/franka_ik_dice_line_demo.mp4` (`scripts/demo_franka_ik_dice_line.py`, run on GCP — desktop was busy with a concurrent workstream) | 243.1s | 8.2MB | Scripted classical-IK demo (no learned policy): all 5 dice picked from a scattered layout and lined up, then re-picked and the whole line relocated (rotated 90°, shifted) to a new spot. 8/10 pick-and-place ops landed within a few mm–~11cm of target across both passes; d4 (this project's own hardest grasp case) failed to be physically grasped in both attempts, so it was sequenced last rather than first. |
| `rl-joint-cube-lift-carry.mp4` | `..._joint-cube_model_1499-step-0.mp4` (checkpoint `logs/train_franka_jointcube/2026-07-12_07-31-58/model_1499.pt`) | 5.0s | 244KB | DexCube fallback, 8/8 sustained lifts, decisive recipe validation |

All three RL clips got a mild `eq` brightness/contrast lift (the source
render is underlit — a dark grid-floor debug scene, not a lighting bug in
the finding itself) purely for on-screen legibility; no content was
altered. Dice-pick clips were re-encoded (`crf 21`) but not otherwise
modified — the on-screen "COMMANDED / detected / confidence" text is the
script's own live overlay, not something added in post.

## Content decisions / left out

- **Only 3 of the asset-bisect ladder's video checkpoints used** (joint-die,
  joint-die-big, joint-cube), skipping `joint-cube-baked` (rung 3, the
  shape-provenance control). It's a real result (3/3, matches DexCube
  reliability) but visually redundant with `joint-cube` — same conclusion
  ("a cube-like object trains reliably"), and the ladder's actual news (die
  fails at 30mm, sometimes succeeds at 48mm, cube always succeeds at 48mm)
  is fully carried by the 3 clips used. Numbers are still in the prose.
- **No d8 before/after detection image pair** — only d10 has a matched
  filename present in both `vision/models/eval/s/overlays/` and
  `vision/models/eval/s_v2/overlays/` with a clean single-die crop; used
  that one pair rather than forcing a second, weaker example.
- **Model-800 / model-4999 checkpoint videos not used** — early Franka Lift
  baseline exploration (2026-07-09/10), superseded by the later joint-space
  work; not part of either milestone this task was asked to cover.
- **No separate d6-regression chart** — folded d6 into the single 6-class
  v1-vs-v2 bar chart rather than adding a second chart; one honest chart
  telling the whole story (2 real wins, 2 unchanged, 1 real regression)
  seemed more useful than a second small chart isolating the bad news.
- **`docker/tensorboard-gcs/`, `scripts/tensorboard_gcs.sh`, and
  `vision/run_render_v2.py`** (untracked files visible in `git status` at
  task start) were not used as content sources — they're infrastructure
  scaffolding, not a documented result with numbers to cite honestly.

