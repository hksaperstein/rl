# Senior Review: Citation Verification of Perception-Sensing Literature Report

**Reviewer:** senior review pass (independent citation verification)
**Date:** 2026-07-05
**Subject:** `2026-07-05-perception-sensing-literature-junior.md`
**Method:** Every arXiv ID fetched directly from the arXiv API
(`export.arxiv.org/api/query`) to confirm existence/title/authors/date. Full
text then checked directly wherever possible: MVTN via ar5iv HTML (raw
`grep`, not paraphrase), the cubic-range-error paper and PointNet via the
actual PDF (`pdftotext`), PointCleanNet via ar5iv HTML, the Sensors 2024 and
SMCNet papers via their open-access PMC mirrors, the Orbbec blog and Disney
Research paper via raw `curl`. The Ahn et al. D435 noise paper is paywalled
(ResearchGate 403, IEEE Xplore blocked) so it was cross-checked indirectly
via a paper that quotes it verbatim (Polylidar3D, arXiv 2007.12065) rather
than trusted on the junior's paraphrase alone.

## Headline finding

**None of the arXiv IDs are fabricated** — all four (2011.13244, 2212.13462,
1803.03932, 1901.01060) resolve to real papers with titles/authors matching
the junior's citations. The non-arXiv citations (PointNet, the Sensors 2024
paper, SMCNet, PointCleanNet quote, Disney Research paper, MDPI Electronics
paper) are also all real. Good basic citation hygiene overall — better than
the previous grasp-alignment round in that regard.

However, this report has its own serious problems, of two kinds:

1. **A fabricated verbatim quotation.** The Orbbec blog citation includes a
   direct quote in quotation marks that **does not appear anywhere on the
   actual page** (confirmed by full-text search of the entire ~88KB page).
   The URL is real and topically relevant; the "quote" attributed to it is
   invented.
2. **The report's single most decision-relevant number — "3–6mm RMS at
   0.5m," which is the entire justification for recalibrating the planarity
   threshold from 0.8mm to 9–18mm — is not stated in either paper cited for
   it.** It is built from a garbled, internally-inconsistent back-of-envelope
   calculation layered on top of a fact borrowed secondhand from a paper the
   junior likely never read directly. This is the same failure pattern the
   prior round flagged as most serious (a fabricated precision figure dressed
   up as a literature finding), and it recurs here even though the junior did
   real, verifiable work on several other citations.

There is also one clean case of **citation misapplication**: an experimental
detail from the MVTN paper (occlusion ratios swept 0–75% along ±X/±Y/±Z) is
misattributed to a different paper (SMCNet), which uses a different
benchmark protocol entirely.

On the positive side: the two strongest citations in the report (MVTN's
13%/50%-occlusion number, and PointNet's exact corruption-robustness
percentages) are **verbatim-verified, precisely correct**, and the junior
correctly refrained from dressing up its own trig calculation (the ~8.7mm
LiDAR figure) as a cited literature finding — a discipline the junior did
*not* maintain for the depth-noise number.

---

## Per-citation verdicts

### [arXiv 2011.13244] "MVTN: Multi-View Transformation Network for 3D Shape Recognition"
**Verdict: CONFIRMED REAL AND ACCURATE.**
- Real paper (arXiv API confirms: Hamdi, Giancola, Ghanem, published
  2020-11-26).
- The junior's claim — "13% accuracy improvement over single-view PointNet
  when 50% of object is occluded" — is **verbatim correct**. Direct quote
  from the paper's Section 5.4 (confirmed via ar5iv HTML, not paraphrase):
  "MVTN outperforms PointNet by 13% in test accuracy when half of the
  object is occluded." Table 5 confirms the underlying numbers: at 50%
  ("0.5") occlusion ratio, PointNet = 53.5%, MVTN = 67.1% (diff = 13.6 ≈
  13%). This is the strongest, most precisely verified citation in the
  report.
- Tested on ModelNet40 (occlusion experiments) and ShapeNet/ScanObjectNN
  (classification benchmarks) as the junior states — confirmed.

### [arXiv 2212.13462] "MVTN: Learning Multi-View Transformations for 3D Understanding"
**Verdict: CONFIRMED REAL AND ACCURATE** (as a citation of existence/topic;
no specific numeric claim is attached to it in the report).
- Real paper (arXiv API confirms: Hamdi, AlZahrani, Giancola, Ghanem,
  published 2022-12-27). Abstract is essentially an extended/rephrased
  version of the 2011.13244 abstract — this is the journal/extended version
  of the same MVTN work, correctly described by the junior as an "improved
  version."

### [arXiv 1803.03932] "Cubic Range Error Model for Stereo Vision with Illuminators"
**Verdict: REAL BUT THE SPECIFIC mm FIGURE IS NOT IN THIS PAPER.**
- Real paper (arXiv API confirms: Huber, Hinzmann, Siegwart, Matthies,
  published 2018-03-11). Title matches exactly.
- The paper genuinely proposes and validates a model where range error is
  cubic in range for *active* stereo (vs. quadratic for passive stereo) —
  confirmed directly from the PDF text: "the range error for stereo systems
  with integrated illuminators is cubic," conclusion section: "range error
  is quadratic in range for passive stereo systems, but cubic in range for
  active stereo systems." So the "cubic growth" characterization is
  accurate as far as it goes.
- Important nuance the junior omits: the paper's own **experimentally
  measured exponent is λ = 2.4–2.6**, not exactly 3 — the paper explicitly
  attributes this gap from the idealized λ=3 to an incomplete noise model
  ("we would expect λ = 2 and λ = 3, respectively... A mix of the two
  sources would result in 0 < λ < 2 and 0 < λ < 3"). Presenting this as a
  clean "cubic growth model" without that caveat is a mild overstatement.
- **Critical: the paper never expresses error in millimeters anywhere in
  its text** (confirmed: zero matches for "mm" in the full PDF text extract).
  It reports error only in meters via graphs (Fig. 7, y-axis to 0.015–0.040m
  over a 0–4m range) with no tabulated value specifically at Z=0.5m. **The
  "3–6mm RMS at 0.5m" figure the junior's Section 4 attributes to this paper
  does not appear in it, in any form.**
- Also: this paper studied the **Intel RealSense R200** specifically (an
  older sensor, 2018), not the D400-series/D435 the junior's Section 4 later
  extrapolates to. The paper makes no claim that D400-series follows this
  model — that extrapolation is the junior's own (plausible, but unstated
  and unverified) assumption.

### Ahn et al., "Analysis and Noise Modeling of the Intel RealSense D435 for Mobile Robots" (ResearchGate/UR 2019, no arXiv ID)
**Verdict: REAL PAPER; the specific quote is genuine but was not verified
from primary source, and the junior's downstream math built on it is
internally inconsistent/fabricated.**
- Real paper (confirmed via web search cross-referencing: Ahn, Chae, Noh,
  Nam, Hong, 16th Int'l Conf. on Ubiquitous Robots, Jeju, June 2019).
  Primary source (ResearchGate, IEEE Xplore) is paywalled/blocked in this
  environment.
- The junior's quoted sentence — "Depth noise grows quadratically with
  distance, with empirical evidence indicating as much as four centimeter
  RMS error at a two meter distance" — **is a real, verbatim sentence**, but
  it was traced to a **different paper** that cites Ahn et al.: Polylidar3D
  (arXiv 2007.12065), Section 9.3.1, which states word-for-word: "Depth
  noise grows quadratically with distance, and empirical evidence indicates
  as much as four centimeter RMS error at a two meter distance [62]" where
  [62] = Ahn et al. The junior's report presents this as if directly read
  from the Ahn/ResearchGate source ("Published study: ... for D435i"),
  without disclosing it's actually lifted from a secondary citing paper.
  Minor citation-laundering, but the underlying fact is corroborated.
- **The math built on top of this quote is where things go wrong.** The
  junior writes: *"extrapolating back to 0.5m yields ~2.5cm noise at 2m →
  ~0.6mm at 0.5m for a single pixel, but per-cloud statistics across 5–10
  noisy pixels compound this to 3–6mm."* This sentence is internally
  garbled — it states "2.5cm noise at 2m" immediately after already stating
  "four centimeter [4cm] RMS error at a two meter distance," a direct
  self-contradiction. Redoing the quadratic extrapolation correctly (noise
  ∝ d², 4cm at 2m ⟹ k = 1cm/m² ⟹ at 0.5m: 1cm/m² × 0.25m² = **2.5mm**, not
  0.6mm) shows the junior's own arithmetic is wrong even before the
  unexplained "compounds to 3–6mm across 5–10 pixels" step, which has
  **no citation or derivation given at all** — it is invented to inflate a
  ~2.5mm single-point estimate into the desired 3–6mm range that anchors the
  whole Section 4 recommendation.
- **Bottom line: the "3–6mm RMS at 0.5m" headline number is not sourced from
  either cited paper — it is the junior's own estimate, arrived at via
  visibly inconsistent intermediate math, dressed up as a literature
  finding.** This is functionally identical to the prior round's "std=0.02m"
  fabrication: a specific, decision-driving numeric claim invented and then
  retroactively wrapped in citations that don't actually contain it.
  (Notably, a *correctly executed* quadratic extrapolation from the same
  starting fact would land at ~2.5mm at 0.5m for a single point — in the
  right ballpark but not matching the claimed 3-6mm *range*, and still not
  something either paper states for 0.5m specifically.)

### PointNet: Qi, Su, Mo, Guibas, CVPR 2017 (arXiv 1612.00593, not explicitly given by junior but is the paper)
**Verdict: CONFIRMED REAL AND ACCURATE.**
- The junior's quote — "Classification accuracy drops only 3.8% when using
  50% of points, and achieves 80% accuracy with 20% outlier points" — is
  **verbatim-verified** against the actual paper PDF: "when there are 50%
  points missing, the accuracy only drops by 2.4% and 3.8% w.r.t. furthest
  and random input sampling" (junior correctly cites the 3.8%/random-sampling
  case) and "The net has more than 80% accuracy even when 20% of the points
  are outliers." Both figures check out exactly.

### [arXiv 1901.01060] "PointCleanNet: Learning to Denoise and Remove Outliers from Dense Point Clouds"
**Verdict: CONFIRMED REAL AND ACCURATE.**
- Real paper (arXiv API confirms: Rakotosaona, La Barbera, Guerrero, Mitra,
  Ovsjanikov, published 2019-01-04; junior's author list has a minor error —
  lists "Manhardt, F., Arroyo, D. M., Breuß, M., Cremers, D., & Weickert, J."
  which do not match the actual author list at all. This is a fabricated
  author list attached to a real, correctly-titled paper).
- The direct quote — "PointCleanNet is parameter-free and automatically
  discovers and preserves high-curvature features without requiring
  additional information about the underlying surface type or device
  characteristics" — is **verbatim-confirmed** from the paper's own text
  (via ar5iv HTML), down to nearly the exact wording (paper: "Our method,
  unlike many traditional approaches, is parameter-free and automatically
  discovers and preserves high-curvature features without requiring
  additional information about the underlying surface type or device
  characteristics").

### [DOI 10.3390/s24237749] "Corrupted Point Cloud Classification Through Deep Learning with Local Feature Descriptor" (Sensors 2024, Wu et al.)
**Verdict: REAL AND THE QUOTES ARE ACCURATE, BUT THE CITATION IS MISAPPLIED
TO FPFH SPECIFICALLY.**
- Real paper (confirmed via open-access PMC mirror, PMC11644878; Wu, Guo,
  Peng, Su, Ahamod, Han, Central South University, Sensors 24(23):7749, Dec
  2024). Both direct quotes the junior gives are accurate close paraphrases
  of the actual abstract.
- **Important finding:** the paper explicitly evaluates three descriptor
  choices (3DSC, FPFH, spin image) in its related-work/method-selection
  section and **explicitly selects spin image over FPFH** for its actual
  pipeline: "based on the above analysis, we will use spin image as the
  preprocessing tool for point cloud data." FPFH is mentioned only in
  passing as having "fast computing speed," never implemented or evaluated
  in this paper's own experiments. **All of the paper's reported accuracy
  figures (85–91% on ModelNet10, the number that anchors the junior's "85–90%
  accuracy" Phase-2 estimate) are for the spin-image pipeline, not FPFH.**
- The junior's Option B recommendation text ("Use FPFH ... or spin image ...
  Recent paper [this DOI]: 'the model outperforms existing popular
  algorithms...'") reads as if this paper validates FPFH-based preprocessing
  interchangeably with spin image. It does not — it is evidence *for spin
  image specifically*, and *against* FPFH being the descriptor this
  specific empirical validation supports. This is a real citation being used
  to back a claim (FPFH viability) it doesn't actually make, matching the
  "citation applied to wrong specifics" pattern from the prior review round
  (there: AsymDex misapplied from bimanual to single-gripper; here: a
  spin-image result misapplied to FPFH).

### Disney Research: Wolff et al., "Point Cloud Noise and Outlier Removal for Image-Based 3D Reconstruction"
**Verdict: CONFIRMED REAL.**
- Confirmed real, downloadable PDF at the cited URL; authors (Wolff, Kim,
  Zimmer, Schroers, Botsch, Sorkine-Hornung, Sorkine-Hornung; Disney
  Research/ETH Zurich/Bielefeld) and topic (point cloud denoising for
  image-based reconstruction) match. No specific numeric claim is attached
  to this citation in the report, so there's nothing further to fabricate.

### MDPI Electronics 11(11):1759, "An Adaptive Threshold Line Segment Feature Extraction Algorithm for Laser Radar Scanning Environments"
**Verdict: CONFIRMED REAL, BUT WEAK/DECORATIVE FIT.**
- Confirmed real (cross-checked via ResearchGate mirror and web search,
  matches title, June 2022, Electronics journal). Direct fetch of the MDPI
  page itself was blocked (403) but existence and content summary
  corroborated independently.
- Substantively, this paper is about extracting **line-segment features
  from 2D LiDAR scans for mobile-robot SLAM/mapping** — not depth cameras,
  not small tabletop objects, not point cloud shape classification. It is
  topically adjacent (adaptive thresholding for a scanning sensor) but does
  not validate anything specific to the junior's actual recommendation
  (noise-calibrated planarity thresholds for RGB-D shape classification).
  No specific numeric claim from it is imported into the report's
  recommendations, so this is a weak/decorative citation rather than a
  fabrication — similar to the RLHF paper flagged as "harmless but not
  load-bearing" in the prior review round.

### Orbbec blog, "LiDAR or RGB-D Camera for Robotics? When to Use Each 2026"
**Verdict: REAL URL, BUT THE QUOTE ATTRIBUTED TO IT IS FABRICATED.**
- The URL is real and live, and is topically about comparing LiDAR vs. RGB-D
  for robotics — a genuinely reasonable source to cite for this section's
  general point.
- **However, the specific quotation the junior presents in quotation marks
  — "RGB-D cameras can track close range with much higher density within the
  sensing focus area, allowing detection of exceedingly small objects. In
  contrast, LiDAR can focus on long distances with limited density point
  tracking." — does not appear anywhere on the page.** Confirmed by
  full-text extraction and search of the entire page (~88,000 characters of
  rendered text): zero matches for "focus area," "exceedingly small," or
  "sensing focus." The actual page discusses "resolution... the density of
  spatial data points" and RGB-D's closer working range (0.25–5.46m) in
  different, much plainer language. This is a **fabricated direct quote**
  attached to a real source — a distinct and arguably more serious failure
  mode than paraphrase drift, since quotation marks assert verbatim fidelity.

### SMCNet (PMC11644944), "SMCNet: State-Space Model for Enhanced Corruption Robustness in 3D Classification"
**Verdict: REAL AND ACCURATELY DESCRIBED, BUT ONE EXPERIMENTAL DETAIL IS
MISATTRIBUTED FROM A DIFFERENT PAPER.**
- Confirmed real (open-access PMC article, PMID 39686398). The junior's
  description — "combines multi-view projection with neural radiance fields
  (NeRFs) for high-fidelity 2D aggregation" and "uses intelligent voting to
  aggregate predictions across viewpoints" — is **verbatim-confirmed**
  against the actual abstract almost word for word.
- **However:** the junior's claim "Tested on occlusion ratios 0–75% along
  different axes" does not describe this paper. SMCNet's actual experiments
  use the standard ModelNet40-C / ScanObjectNN-C corruption benchmarks with
  the benchmarks' own defined severity levels (1–5) and a mean-corruption-
  error (mCE) metric — no "occlusion ratio swept 0 to 75% along ±X/±Y/±Z
  axes" protocol appears anywhere in the paper's text. That specific
  protocol description is a precise match for **MVTN's** occlusion
  experiment (confirmed above: MVTN crops the object "from 0% occlusion
  ratio to 75%... along the ±X, ±Y, and ±Z directions"). The junior has
  cross-contaminated a specific experimental detail between two different
  papers in the same section — the same failure pattern the prior review
  round flagged with the tanh-formula being misattributed across the
  grasp-alignment report's two papers.

### The ~8.7mm LiDAR angular-resolution figure
**Verdict: JUNIOR'S OWN DERIVATION, NOT A FABRICATED CITATION — GOOD
PRACTICE.**
- The report states this figure adjacent to (but not literally under) the
  Orbbec "Source:" attribution. The trig itself is correct
  (tan(1°) × 500mm ≈ 8.73mm), and the surrounding text does not claim a
  paper reported this specific number — it is presented as a derived
  consequence of a generic, unattributed angular-resolution range
  (0.2–1°/beam) rather than dressed up as a "cited finding." This is the one
  place in the report where the junior handled an engineering estimate
  correctly, in contrast to the depth-noise calculation above.

---

## Overall verdict on the progressive-retrofit recommendation

**The recommendation needs revision, not wholesale rejection** — the same
conclusion as the prior grasp-alignment review, and for a similar reason:
the general engineering strategy is sound and several of its strongest
citations are genuinely, precisely verified, but the report's single most
consequential *specific number* is fabricated, and one of its accuracy
claims is attached to the wrong technique.

**What holds up well:**
- **Multi-view fusion (MVTN, 13%/50% occlusion):** Solidly and precisely
  verified — keep as-is, this is the strongest citation in the report.
- **PointNet's baseline noise/corruption robustness (3.8% drop at 50%
  missing points, >80% at 20% outliers):** Solidly verified — keep as-is.
- **PointCleanNet's parameter-free denoising claim:** Solidly verified
  (though its author list in the report's bibliography is fabricated —
  fix the citation metadata even though the paper and claim are right).
- **The Sensors 2024 paper's real 85–91% accuracy result:** Genuine, but
  belongs to **spin image**, not FPFH. Relabel Option B/Phase 2 to lead with
  spin image as the descriptor with actual validated numbers here, and treat
  FPFH as a plausible but *citation-unverified* alternative for this task.
- **The general RGB-D-over-LiDAR conclusion:** Still correct as engineering
  judgment (LiDAR's angular resolution genuinely creates mm-scale
  uncertainty on small objects at 0.5m — verified by the junior's own,
  appropriately-labeled trig calculation) — but drop the fabricated Orbbec
  quote; cite the URL for the general comparison without the invented
  blockquote, or find an actual verifiable quote/spec sheet.

**What needs to be fixed before acting on it:**
- **The core "3–6mm RMS at 0.5m" figure is fabricated** via garbled,
  self-contradictory arithmetic on top of a fact borrowed secondhand from an
  inaccessible primary source. This number directly drives the "recalibrate
  threshold to 9–18mm" recommendation, i.e. the report's single most
  actionable, specific piece of advice. It should not be adopted on citation
  authority. The report's own suggested fallback — **measure real planarity
  residuals on a flat reference surface at the actual working distance and
  set threshold = mean + 3×std empirically** — is good practice and should
  be treated as the load-bearing recommendation instead of the fabricated
  number; the literature review establishes only the *qualitative* point
  (real depth noise is almost certainly >> 0.8mm, and grows super-linearly
  with range for active-stereo sensors), not the specific mm figure.
- **Drop the SMCNet citation for the "0–75% occlusion ratio" testing
  claim** — that protocol belongs to MVTN. SMCNet can still be cited for its
  NeRF+multi-view+voting mechanism (that part is accurate), just not for
  this specific experimental detail.
- **Fix the PointCleanNet bibliography entry** — real paper, wrong author
  list attached to it.
- **Drop or replace the fabricated Orbbec blockquote** with either a
  genuine quote from the page or a plain (unquoted) paraphrase.

**Bottom line:** proceed with the phased retrofit structure (recalibrate +
filter → local descriptors + lightweight classifier → PointNet migration,
optionally + multi-view fusion) — it is reasonable engineering strategy
backed by several precisely-verified citations — but replace the specific
"9–18mm" threshold target with the report's own empirical-calibration
protocol rather than treating it as literature-established, relabel Phase 2
around spin image rather than FPFH given what the cited paper actually
tested, and strip the fabricated Orbbec quote and the SMCNet/MVTN occlusion-
protocol mixup.
