# site/ — portfolio dashboard bundle

What this is: a self-contained, static, dependency-free (plain HTML/CSS/JS,
no build step, no external CDN/font/framework fetches) dashboard for the
`rl` monorepo (Isaac Lab manipulation RL + the Blender-based synthetic dice
detector). It works standalone from `file://` or any static host, and it
is also a source of content blocks meant to be grafted into the existing
Astro portfolio at `hksaperstein.github.io`, which already has two live
project pages covering part of this work
(`src/content/projects/franka-dice-pick.md`,
`src/content/projects/dice-detection.md`).

Total size: **4.1MB** (8 videos, ~3.7MB; 10 images, ~370KB; HTML/CSS/JS,
~40KB). Well under the ~80MB budget in the brief. Every video was
transcoded with `ffmpeg` to H.264 + `yuv420p` + `+faststart`, and every one
was verified afterward with `ffprobe` (duration/codec/size) and a frame
extract — see the video inventory below.

Voice: first person, Harrison, following `hksaperstein.github.io/VOICE.md`
verbatim (state results directly, name failures plainly, no marketing
language, no unsourced numbers, no AI-tell words, no emoji). This
overrides the earlier "recruiter dashboard" framing from the original
brief where the two conflict — the copy in `index.html` was written for
that voice from the start, not converted after the fact.

## Why this shape, not a fresh from-scratch site

Discovered mid-task that a real portfolio already exists at
`hksaperstein.github.io` (Astro + Tailwind) with two project pages that
already cover a meaningful slice of this repo's work — the dice-pick demo
gallery (5 videos, already captioned) and the dice-detector's v1 sim-to-real
results (assets, geometry, YOLO training, the S vs S+R failure). Duplicating
that content here would either go stale immediately or fight the live site
for authority. So this bundle does two things instead:

1. **Standalone dashboard** (`index.html`) that also works as a status page
   in its own right — it includes brief, non-duplicated context for the
   already-shipped material (2 of the 5 dice-pick clips, for orientation)
   plus the full detail on everything that's genuinely new since those
   pages were last written.
2. **Graft-ready content** for the two live pages, marked with matching
   HTML comment pairs in `index.html` (an opening marker naming the target
   page and a closing marker) and mirrored as pasteable markdown in
   `content/`.

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

| Section in `index.html` (marker) | Target | Action |
|---|---|---|
| `hero` | dashboard-only | New. Either keep as the dashboard's own hero, or lift the copy into a new intro paragraph on `franka-dice-pick.md` — controller's call, both read fine. |
| `highlights` (4 stat cards) | dashboard-only | New. Dashboard-specific; not a natural fit inside either project page's own narrative flow. |
| `demo-gallery` (2 dice-pick clips) | franka-dice-pick | **No action needed.** These 2 clips (of the page's existing 5) are shown here only for context before the new RL section; the live page already has the full 5-clip gallery with its own captions. Don't duplicate. |
| `rl-milestones` (3 new RL videos + asset-bisect writeup) | franka-dice-pick | **Graft this.** Matches `content/franka-dice-pick-additions.md` — new gallery entries (section "RL lift experiments") + a new prose section, suggested placement: after "What Isaac Lab taught me the hard way", before "The d4". |
| `detection-fix` (datagen-v2 chart + before/after image pair) | dice-detection | **Graft this.** Matches `content/dice-detection-additions.md` — new gallery entries (section "Sim-to-Real Results — datagen-v2") + a new "## Step 5" prose section. The inline `<svg>` bar chart in this section is copy-pasteable as-is (it's self-contained, no JS dependency) if the Astro Gallery component can host raw SVG; otherwise re-render it as a static image using the same 6-class mAP50 table (both variants' numbers are in the section text and in `content/dice-detection-additions.md`). |
| `research-process` | dashboard-only | New. Spans both projects (and the wider repo's process conventions) — doesn't belong to either single project page. Recommend either keeping it here or, if the live site gets a dedicated "how I work" page later, moving it there. |
| `stack` | dashboard-only | New. Same reasoning as `research-process`. |

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
| `dice-pick-d8.mp4`, `-d12.mp4` | same dir | 17.3–17.7s | 412–504KB | Copied/re-encoded for completeness (all 5 die types available in this bundle); not embedded in `index.html` beyond the 2 gallery clips + hero, since the live franka-dice-pick.md page already has all 5 with their own captions |
| `rl-joint-die-d20-falsified.mp4` | `logs/videos/franka_checkpoint_review/franka_checkpoint_review_joint-die_model_1499-step-0.mp4` (checkpoint `logs/train_franka_jointdie/2026-07-12_06-56-02/model_1499.pt`) | 5.0s | 312KB | d20 at real 30.3mm size, 0/8 sustained lifts |
| `rl-first-die-lift-rung2-seed123.mp4` | `..._joint-die-big_model_1499-step-0.mp4` (checkpoint `logs/train_franka_jointdiebig/2026-07-12_15-07-49/model_1499.pt` — confirmed via the matching `heights_joint-die-big_model_1499.json`, this is specifically the seed-123 run) | 10.0s | 384KB | d20 at 48mm, 8/8 sustained lifts — first learned die lift |
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

## Deploy notes

Any static file server works — this bundle has no server-side dependency.
Examples:

```bash
# quick local check
python3 -m http.server 8000 --directory site

# or, once copied into the Astro site's public/ dir, it's served by
# Astro's own dev/build pipeline — no changes needed to astro.config.mjs
```

If deployed as its own standalone page (rather than grafted into the two
project pages), drop the `site/` directory root onto any static host
(GitHub Pages, Netlify, a plain nginx `root`) — `index.html` resolves all
asset paths relatively (`assets/...`, `css/...`, `js/...`), so it does not
require being served from a particular subpath.
