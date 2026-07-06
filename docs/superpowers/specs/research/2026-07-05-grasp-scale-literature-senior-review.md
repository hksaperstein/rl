# Senior Review: Grasp-Scale Literature (Junior Report Verification)

**Reviewing:** `2026-07-05-grasp-scale-literature-junior.md`
**Method:** Live arXiv API (`export.arxiv.org/api/query`) for every citation's
existence/title/authors/date; `WebFetch` against the arXiv abstract pages
(and PDF/HTML full text for the one paper with a purported verbatim quote)
for content verification; direct repo inspection (`tasks/ar4/robot_cfg.py`,
`tasks/ar4/env_cfg.py`) for the numeric/config claims about the actual system.

## Repo facts, verified directly

- `tasks/ar4/robot_cfg.py`: gripper actuator is `ImplicitActuatorCfg(stiffness=1000.0,
  damping=50.0, ...)`, `GRIPPER_OPEN_POS = 0.014` (per jaw → 0.028 m full
  aperture). Junior's stated current values (K=1000, D=50, 2.8cm stroke) are
  **accurate**.
- `tasks/ar4/env_cfg.py:48`: gripper action is indeed `mdp.BinaryJointPositionActionCfg`.
  Junior's characterization of the action space as binary open/close is
  **accurate**.

## Per-citation verdicts (arXiv API ground truth)

All 11 arXiv IDs cited in the report were queried directly against
`export.arxiv.org/api/query` (not memory). **All 11 exist and match the
junior's stated titles/authors/dates exactly** — no fabricated IDs this
round, unlike the prior two rounds.

| ID | Title (API) | Verdict |
|---|---|---|
| 2604.02523 | Tune to Learn: How Controller Gains Shape Robot Policy Learning (Bronars, Park, Agrawal; pub. 2026-04-02) | **Confirmed real; content overstated** — see below |
| 2602.23408 | Demystifying Action Space Design for Robotic Manipulation Policies | Confirmed real; genuinely about action-space design, but is an **imitation-learning / real bimanual-robot** study, not RL grasp-closure-coarseness at small scale specifically |
| 2606.18594 | Benchmarking Action Spaces in RL for Vision-based Robotic Manipulation | Confirmed real; about continuous action-space families (pose/joint velocity/increment) for sim-to-real RL, not about binary-vs-continuous gripper coarseness at small scale |
| 1908.08659 | A Comparison of Action Spaces for Learning Manipulation Tasks | Confirmed real; supports task-space impedance > joint PD for sample efficiency, as junior states |
| 2604.11640 | Micro-Dexterity in Biological Micromanipulation | Confirmed real; **quote verified verbatim in full text** (see below), but **badly scale-mismatched** to this task — see below |
| 2605.31486 | Learning Controlled Separation of Small Objects Between Two Fingers with a Tactile Skin | Confirmed real; **accurately characterized** — abstract confirms tactile skin improves task performance up to 20% over joint-sensors-only on 6mm pellets |
| 2403.12170 | Sim2Real Manipulation on Unknown Objects with Tactile-based RL | Confirmed real, used only for a "this doesn't exist" negative claim (fine) |
| 2511.04831 | Isaac Lab: A GPU-Accelerated Simulation Framework... | Confirmed real (background citation only) |
| 2312.03673 | On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer | Confirmed real (listed in Sources but not substantively used in the numbered findings) |
| 2212.05275 | Towards Scale Balanced 6-DoF Grasp Detection in Cluttered Scenes | Confirmed real (listed in Sources, not substantively used) |
| 2502.09886 | Video2Policy | Confirmed real, used only for a "this doesn't exist" negative claim (fine) |

No fabricated IDs, no fabricated titles/authors. This is a meaningfully
cleaner citation-existence record than either prior round this session.

## Deep-dive: the headline claim (Priority 1 recommendation)

### 1. Does "Tune to Learn" (2604.02523) say what the junior claims?

Verified abstract via arXiv API directly:

> "we find that: (1) behavior cloning benefits from compliant and overdamped
> gain regimes, (2) **reinforcement learning can succeed across all gain
> regimes given compatible hyperparameter tuning**, and (3) sim-to-real
> transfer is harmed by stiff and overdamped gain regimes."

The junior's verbatim quote ("Optimal gain selection depends not on the
desired task behavior, but on the learning paradigm employed") **is
accurate** — it appears essentially word-for-word in the abstract. That part
is not fabricated.

**However, the junior's headline framing inverts the paper's own finding for
the regime that actually applies here.** This repo's AR4 task is pure
in-simulation PPO training — there is no sim-to-real transfer step in the
three failed experiments referenced (confirmed via `ROADMAP.md`, which
documents three sim-only reward-shaping experiments, no real-hardware
deployment). The paper's finding for *that exact regime* — "RL from scratch"
— is that it "**can succeed across all gain regimes given compatible
hyperparameter tuning**." That is closer to evidence *against* the
junior's root-cause hypothesis than for it: the paper does not predict that
a stiffness/damping mismatch by itself should make pure-sim PPO training
fail. Only the sim-to-real-transfer regime (not in play here) is described
as harmed by stiff/overdamped gains. The junior's report never surfaces this
distinction — it cites the sim-to-real finding as if it transfers directly
to justify a pure-sim-RL root-cause story, which the paper's own three-way
split argues against.

I confirmed this isn't a WebFetch artifact by re-querying the paper's
project page independently; both sources gave the same three-way split and
confirmed the paper contains **no formula or rule relating gains to
embodiment scale, joint travel range, or gripper stroke** — that part of
the report (natural-frequency/scaling reasoning) is 100% the junior's own
invention, correctly caveated in Finding 4 ("No published paper formulates
[a scaling rule]... this is a physics-informed guess, not an RL-proven
rule") but then re-presented in the "Recommended Action Plan" with the
paper cited as "Evidence" for a specific numeric K/D target, which
overstates what the citation supports.

### 2. The ω_n = √(K/D) formula is not standard physics — flag this

Finding 1 computes "natural frequency" as `ω_n = √(K/D)`. This is not a
recognized formula. For a standard mass-spring-damper joint (`m·ẍ + D·ẋ +
K·x = F`), natural frequency is `ω_n = √(K/m)` (stiffness over *mass*, not
damping) and damping ratio is `ζ = D / (2√(K·m))`. Computing frequency from
`K/D` alone is dimensionally and physically ungrounded — it conflates
stiffness/damping ratio (which has units of 1/time, a corner-frequency-like
quantity for an overdamped first-order system, not a natural frequency of a
second-order oscillator) with true natural frequency, and it requires an
effective mass/inertia term that never appears anywhere in the report. The
arithmetic (`√(1000/50) ≈ 4.5`) is correctly computed *given the formula*,
but the formula itself is invented pseudo-physics presented with the
confidence of an established relationship — the same "dressed-up precision"
pattern flagged in both prior rounds (the fabricated std=0.02m and the
garbled 3-6mm RMS arithmetic), just applied to a formula instead of a raw
number this time.

### 3. The specific K→300-400, D→25-35 numbers: not from the paper

Confirmed by direct inspection of the report and independent paper-content
checks: **the cited paper contains no scaling rule and no numeric gain
targets for this or any embodiment.** The junior's own Finding 4 admits
this ("No published paper... formulates a rule"). The K/D targets are constructed
entirely from the junior's own arithmetic:

- Scale factor 0.35 = AR4 aperture (0.028 m, correctly read from
  `robot_cfg.py`) / an assumed Franka Panda gripper full-aperture of 0.08 m.
  0.08 m *is* a real, correct, well-known Franka Panda spec (2×0.04m finger
  travel) — so the 0.35 ratio itself is arithmetically sound and not an
  invented number, contrary to how the review brief flagged it as suspect.
  This part checks out.
- But the K/D table applies this ratio **inconsistently**: Finding 1 scales
  D by `1/√2.85 ≈ 1.7` (giving D≈29), while the summary table in Finding 6
  says D is scaled by the *same* 0.35 factor as K (which would give
  `50 × 0.35 ≈ 17.5`, not 25-35). These two derivations use different
  scaling laws for damping (linear-in-scale vs. sqrt-in-scale) and produce
  materially different numbers, yet the report presents both as if they
  agree on "25-35." This is an internal arithmetic inconsistency the junior
  did not reconcile or flag — a smaller-scale echo of the "garbled,
  self-contradictory arithmetic" problem from the perception-literature
  round.
- The "Franka K ≈ 3000-5000... D ≈ 100-150" comparison figures are sourced
  to "ROS docs" (unverifiable, not checked against any real ROS/Franka
  documentation by me or apparently by the junior) and mix real-hardware
  impedance-control units with Isaac Sim's `ImplicitActuatorCfg` units,
  which are not guaranteed to be commensurable. This comparison is weak
  supporting evidence at best.

**Bottom line on the numbers: K=300-400 / D=25-35 is the junior's own
extrapolation, not a citation-backed finding, and it is internally
inconsistent even on its own terms.** The report is reasonably honest about
this in prose ("physics-informed guess, not an RL-proven rule") but the
"Evidence: Cite arXiv:2604.02523" line in the Action Plan section
overclaims by attaching the paper's authority to numbers the paper never
produced — and, per point 1 above, the paper's actual RL-from-scratch
finding runs somewhat counter to the premise that gain mismatch alone
explains a pure-sim PPO failure.

## Secondary claims

**(a) Small-object tactile-feedback sub-problem** — **confirmed accurate.**
2605.31486's abstract directly supports the claim: tactile skin feedback
improves separation-task performance by up to 20% over joint-sensor-only
baselines on 6mm pellets, and the paper is explicitly framed as a distinct
small-object problem. Good citation, accurately used.

**(b) Binary vs. continuous gripper coarseness** — **weakly supported /
overstated.** The three action-space papers (2602.23408, 2606.18594,
1908.08659) are all real and do show action-space choice matters for
manipulation learning generally (delta vs. absolute, task-space vs.
joint-space, impedance vs. PD). But **none of them study binary vs.
continuous gripper actions specifically, and none address small-object or
small-gripper scale.** The junior's leap from "action space design matters
in general" to "your binary gripper action is too coarse for an 18mm
sphere" is the junior's own hypothesis, not something these papers
demonstrate. This should be labeled a plausible untested hypothesis, not a
literature-backed conclusion.

**(c) Micro-dexterity / Stokes-drag framing (2604.11640)** — **citation
accurate, application overstated to the point of likely misapplication.** I
verified the junior's block-quote is genuinely near-verbatim in the paper's
full text (confirmed via HTML full-text fetch, not just the abstract) — this
is not a fabricated quote, unlike the prior round's fabricated quote. But
the paper's actual subject matter is biological micromanipulation:
single cells, spheroids, organoids, tissue constructs, and microrobots
operating in **fluidic, confined environments** where viscous drag and
surface adhesion dominate specifically *because there's a fluid medium and
negligible inertia*. The AR4 sphere is an 18mm rigid object manipulated in
**air, on a dry benchtop**, by a normal-inertia robot arm — not remotely the
physical regime this paper studies. The claim "your 9mm sphere is
borderline [for Stokes drag / van der Waals effects]" misapplies a
microfluidics/cellular-scale paper's physics to a macroscale dry
pick-and-place task. This should be treated as **not applicable** to the
actual root-cause investigation, despite the citation being real and
accurately quoted.

## Overall verdict

**Citation hygiene: much improved over the prior two rounds.** All 11
arXiv IDs are real, correctly titled/authored, and none of the quotes I
spot-checked were fabricated outright (a first, compared to the prior two
rounds' fabricated quote and fabricated numeric figure).

**Substance verdict: the specific K=300-400 / D=25-35 numbers should NOT be
acted on as a literature-backed finding.** They are:
1. Not present in or derivable from the cited paper (which the junior's own
   Finding 4 admits).
2. Internally inconsistent (two different scaling laws for D that don't
   agree, papered over in the summary table).
3. Built on an invented "natural frequency" formula that isn't standard
   control theory.
4. Attached to a root-cause story (gain mismatch → PPO training failure)
   that the cited paper's own three-way split arguably argues *against* for
   a pure-sim RL setting (paper: RL-from-scratch "can succeed across all
   gain regimes given compatible hyperparameter tuning"; only sim-to-real
   transfer, not in play here, is harmed by stiff/overdamped gains).

**However, per this repo's stated practice (test directionally-sound
hypotheses empirically rather than discard them for lacking a citation):**
the *qualitative* direction — try reducing gripper stiffness/damping as one
candidate change, alongside the already-flagged `ContactSensor`/
`contact_forces` ground-truth-touch signal from `ROADMAP.md`'s own
next-steps — is cheap to test and not unreasonable on its own physical-
intuition merits (a 1000-stiffness/50-damping actuator on a 2.8cm-stroke
joint could plausibly be too stiff for fine contact detection, independent
of what this particular paper says). **Recommendation: run it as a labeled,
untested empirical hypothesis** (e.g. "gain rescaling, informal dimensional
guess, not literature-derived") rather than in the Action Plan's current
framing, which cites arXiv:2604.02523 as "Evidence" for numbers and a
root-cause story the paper does not actually establish. Given
`ROADMAP.md`'s note that three reward-shaping-only attempts have already
failed and the recommended next steps are `ContactSensor`/`contact_forces`
ground-truth touch signal or curriculum staging, gain rescaling — if tried
at all — should be treated as one candidate to test alongside those, not
prioritized over them on the strength of this citation.
