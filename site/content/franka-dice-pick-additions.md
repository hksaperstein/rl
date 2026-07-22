<!--
  Ready-to-paste additions for src/content/projects/franka-dice-pick.md.
  Everything below happened after that page's 2026-07-11 date and isn't on
  it yet. Two pieces:

  1. New `gallery` array entries — insert into the existing frontmatter
     `gallery:` list (a new "RL lift experiments" section).
  2. A new prose section — insert into the markdown body, I'd suggest
     right after "## What Isaac Lab taught me the hard way" and before
     "## The d4", since it's a second engineering thread on the same
     platform, not a continuation of the d4 discussion.

  Voice: first person, technical register, per VOICE.md. Written fresh in
  that voice, not lifted from ROADMAP.md's third-person research-log style.
  Video/image assets referenced here are already at the paths below —
  see site/README-deploy.md for the copy-in instructions.
-->

## Frontmatter gallery additions

```yaml
  - section: "RL lift experiments"
    file: "/assets/videos/projects/franka-dice-pick/rl-joint-die-d20-falsified.mp4"
    description: "The d20 at its real 30.3mm size, direct joint-position actions, 1500 iterations. Reach-then-settle every episode, 0/8 sustained lifts. Training itself was stable — the policy just never discovered a grasp at this size."
  - section: "RL lift experiments"
    file: "/assets/videos/projects/franka-dice-pick/rl-first-die-lift-rung2-seed123.mp4"
    description: "Same d20, scaled to 48mm (DexCube's size), mass pinned at 0.216kg. Seed 123 of three found the grasp — position error 0.0956, better than the DexCube reference run's own 0.105, all 8 eval envs sustaining the lift. Seeds 42 and 7 at the same size still failed. First checkpoint on this platform that ever picked up a die."
  - section: "RL lift experiments"
    file: "/assets/videos/projects/franka-dice-pick/rl-joint-cube-lift-carry.mp4"
    description: "The DexCube fallback that told me the joint-space action recipe itself was sound before I started bisecting the die asset: position error 0.105, mean episode reward 138 versus 2 for the failed d20 run."
```

## Prose section — "RL: getting a policy to lift a die"

```markdown
## RL: getting a policy to lift a die

The pick above is scripted — Phase I of this project is a trained policy that
can grasp and lift on its own, and that turned out to be a project in itself.

I took Isaac Lab's validated Franka lift recipe, swapped it to direct
joint-position actions (no IK), and pointed it at the physics-baked d20. It
failed completely: 1500 iterations, `lifting_object` never moved off its
0.12 spawn-artifact floor, position error never beat the do-nothing
baseline. Before concluding the action space didn't work, I ran the
pre-authorized fallback — same config, DexCube swapped back in for the die —
and it trained a clean lift and carry in the same 1500 iterations: position
error 0.105, half the baseline. So the recipe was fine. The die asset was
the variable.

I bisected it three variables at a time, three seeds each, against a
0.216kg/48mm baseline:

- **Mass** — d20 at its real 30.3mm size, mass raised 21.6x: 0/3, nearly
  identical failure curves across seeds. Mass isn't the gate.
- **Size** — d20 scaled to 48mm, mass pinned: 1/3. One seed (123) found the
  grasp outright — position error better than the DexCube reference. The
  other two failed the same way as the 30.3mm runs. That's a real split, not
  a clean pass, but it's the finding: 30.3mm is a deterministic 0/4 across
  every full run I have at that size; 48mm is "sometimes discoverable."
- **Shape** — a cube from my own bake pipeline at the same 48mm/0.216kg:
  3/3, matching DexCube's own reliability. This also clears the bake
  pipeline itself of producing broken assets.

Shape gates whether the policy ever discovers a grasp; size modulates how
often. My read: a flat-faced object gives a clumsy early contact something
to land on — a wide antipodal-grasp basin. A near-spherical polyhedron rolls
away from a clumsy contact and never offers parallel faces, so the first
rewarding grasp is rare, and at 30mm, in three seeds of evidence, never
found. That's grasp-affordance scarcity as an exploration failure, not a
physics bug in this project's own pipeline — and it lines up with what I
found in the literature on shape-conditioned grasp discovery (Zhou & Held
2022; Danielczuk et al.) before I ran the ladder.

Next step, not yet built: an object-size curriculum, training where
discovery is reliable (48mm and/or cube-like) and annealing toward the
30mm d20.
```

## Frontmatter gallery addition — IK dice-line demo (2026-07-21)

```yaml
  - section: "Scripted IK demo"
    file: "/assets/videos/projects/franka-dice-pick/dice-line-pick-and-place.mp4"
    description: "Classical differential IK, no learned policy: the arm picks all 5 dice from a scattered layout and lines them up, then re-picks each one and relocates the whole line to a rotated, shifted position. 8/10 pick-and-place operations landed within a few mm to ~11cm of target across both passes; d4 — this project's own hardest grasp case — never got a clean grip in either attempt, exactly the failure mode documented earlier on this project."
```

## Prose section — "A scripted choreography demo, and d4's grip problem again"

```markdown
## A scripted choreography demo, and d4's grip problem again

Not every result here is from a trained policy. This one is a straight
Cartesian-waypoint controller — approach, descend, close, lift, carry,
descend, release, retract — driving the same Franka+dice scene through two
acts: line the 5 dice up, then re-pick the whole line and move it somewhere
else, rotated 90 degrees. No vision, no RL — ground-truth poses read
straight out of the simulator, reusing the exact staged-IK mechanism
(joint-space prep before any Cartesian move, canonical straight-down
orientation, the V-notch fingertip fixture) already proved out on the
single-die pick demo above.

8 of 10 pick-and-place operations across both passes landed within a few
millimeters of their target, occasionally worse (~11cm on one d20 attempt —
Isaac's physics isn't perfectly deterministic run to run, and this project
hasn't chased that down). d4 is the outlier again: every IK waypoint
converged cleanly, but the gripper never actually captured the die — it
just stayed where it started, both times I asked for it. This is the same
grasp-precision problem the V-notch fixture above was built to fix, still
showing up here. I moved d4 to the last pick of the sequence once I saw
this live, so the video doesn't open on a stall — a sequencing choice, not
a fix.
```
