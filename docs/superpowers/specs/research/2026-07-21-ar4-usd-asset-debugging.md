# AR4 USD asset debugging — direct prim-level inspection of the three unresolved defects

**Date:** 2026-07-21
**Author:** Senior engineering thread (delegated by Principal)
**Purpose:** `docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`
established, from repo history/code reading alone, that Hypothesis 2
(jaw-mimic constraint) was "confirmed never correctly enforced" but never
isolated the physical mechanism, and Hypothesis 3 (jaw collision-geometry
approximation) was "genuinely inconclusive... never independently
inspected on either the built AR4 USD or Isaac Lab's own shipped Franka
asset." This task opens the actual built `ar4_mk5.usd` directly (via
`pxr.Usd`/`UsdPhysics`/`PhysxSchema`, this project's own established
direct-USD-introspection pattern from `scripts/bake_die_asset.py`) and
answers all three with real, dumped prim properties rather than inference.

**Method:** No local Isaac Sim/GPU on this machine (`assets/` is
gitignored, no `pxr`, no AR4 URDF checkout locally) — dispatched to a GCP
SPOT `g2-standard-4`+`nvidia-l4` instance (`us-central1-c`, after a
stockout on `us-central1-a/b`), Isaac Sim 5.1.0 + Isaac Lab v2.3.1 via the
proven pip recipe (`docs/cloud/franka-cloud-shakedown.md`). Built the AR4
asset from `https://github.com/Annin-Robotics/ar4_ros_driver.git` at
commit `6a3ebb11cedab12cd1d29b41d63aa270a008ab8b` (the same commit the
concurrent `2026-07-21-ar4-franka-fixes-transfer-design.md` task pinned),
using the same `xacro`/`ament_index_python`-shim environment fixes that
task documented. One genuine SPOT preemption occurred mid-task (confirmed
via `gcloud compute operations list` as a real `compute.instances.preempted`
event, not a stockout/manual stop); the instance was restarted and the
already-built asset survived on the boot disk, so no work was lost. GPU
quota (`GPUS_ALL_REGIONS`, project-wide cap of 1) was confirmed free before
provisioning and the instance was fully torn down at the end (verified via
`scripts/check_cloud_state.sh`: zero instances/disks/snapshots belonging to
this task).

---

## 1. Jaw-mimic constraint — a real, correctly-referenced `PhysxMimicJointAPI` exists, but its own joint-limit import was mathematically inconsistent with it

**Verdict: this project's 2026-07-09-era belief ("Isaac Sim's USD import of
this asset appears not to enforce that [mimic] constraint," restated as
"confirmed" in the 2026-07-20 root-cause doc) is directly contradicted by
today's build.** `parse_mimic=True` (`scripts/build_asset.py`, unchanged)
DOES produce a real, live `PhysxMimicJointAPI:rotX` instance on
`gripper_jaw2_joint`, with:

```
rel physxMimicJoint:rotX:referenceJoint -> /mk5/root_joint/joints/gripper_jaw1_joint
attr physxMimicJoint:rotX:gearing = -1.0
attr physxMimicJoint:rotX:offset = 0.0
attr physxMimicJoint:rotX:dampingRatio = 0.005
attr physxMimicJoint:rotX:naturalFrequency = 25.0
```

This is a spring-based constraint (`dampingRatio`/`naturalFrequency`, not a
hard kinematic tie) targeting `jaw2_position = gearing * jaw1_position +
offset = -jaw1_position`. Whether the earlier finding was simply testing an
older Isaac Sim/importer version, or was never actually checked at the
prim level at all (the 2026-07-20 doc's own §2b flags an unresolved,
contradictory ROADMAP claim on exactly this point) is not resolved here —
what matters is that **today's pinned stack (Isaac Lab v2.3.1 / Isaac Sim
5.1.0) does author a real mimic constraint with the correct reference
joint.**

### The actual bug: jaw2's own hard `physics:lowerLimit`/`upperLimit` don't fit the mimic formula's own mapped range

Directly dumped, before any fix:

| Joint | `lowerLimit` | `upperLimit` |
|---|---|---|
| `gripper_jaw1_joint` | `0.0` | `0.014` |
| `gripper_jaw2_joint` (as imported) | `-0.0028` | `0.0168` |

The mimic formula (`q2 = gearing*q1 + offset = -q1`), applied to jaw1's own
real range `[0, 0.014]`, maps to `q2 ∈ [-0.014, 0]`. This does **not** fit
inside jaw2's own imported hard limits `[-0.0028, 0.0168]` — specifically,
`-0.014 < -0.0028`. PhysX's hard joint-limit clamp takes priority over the
spring-based mimic constraint, so **as soon as jaw1 moves past
`q1 = 0.0028m` (only 20% of its full 14mm stroke), the mimic constraint's
target for jaw2 falls below jaw2's own allowed minimum and jaw2 gets
physically clamped at its own limit — unable to continue tracking jaw1 for
the remaining 80% of the gripper's real closing motion.**

This is a concrete, directly-measurable, self-consistent explanation for
this project's long-standing, three-times-unresolved jaw asymmetry defect
(URDF-native mimic reliance, a manually-authored PhysX mimic API in
Experiment 19, software leader-follower mirroring in Experiment 22 — see
`2026-07-20-ar4-vs-franka-root-cause-comparison.md` §2a) — **none of the
three prior attempts diagnosed this specific limit/gearing mismatch**,
because none of them opened the built USD to check jaw2's own authored
limits against the mimic formula's mapped range.

The source URDF (`ar_gripper_macro.xacro:75-89`) itself is internally
consistent — both joints declare identical `<limit lower="0" upper="0.014"
.../>` and the `<mimic multiplier="1" offset="0"/>` tag — so this specific
limit mismatch (and the importer's own sign flip from `multiplier=1` to
`gearing=-1.0`, plausibly a legitimate axis-convention conversion given the
gripper macro's 180°-about-Y jaw2 frame flip) is introduced by Isaac Sim's
URDF→USD import step itself, not by anything authored in this repo or the
upstream URDF.

### Fix applied

`scripts/build_asset.py`: added `_fix_gripper_jaw2_mimic_limits()`, run
right after import. It reads jaw1's own limits and jaw2's already-authored
`gearing`/`offset` directly off the just-imported stage (no hardcoded
constants), computes the mimic-consistent range, and overwrites jaw2's
`physics:lowerLimit`/`physics:upperLimit` to match. Re-verified by
re-opening the rebuilt USD fresh: `gripper_jaw2_joint` now shows
`lowerLimit = -0.014, upperLimit = 0.0`, exactly the mapped range of
jaw1's own `[0, 0.014]` under the unchanged `gearing=-1.0, offset=0.0`.

### Verification status: static fix confirmed; live dynamic confirmation inconclusive

Direct, static re-inspection of the rebuilt USD confirms the fix is
correctly authored (above). A live dynamic test was also attempted: a bare
`isaacsim.core.api.World` scene (no IsaacLab task pipeline — deliberately
minimal, to isolate the asset itself) with the fixed asset referenced in,
driving `gripper_jaw1_joint`'s position target through a 300-step ramp from
0 to 0.014 and reading back both jaws' actual positions via
`isaacsim.core.prims.Articulation`. **jaw1 tracked its ramping target with
a normal (if laggy, due to a continuously-moving setpoint) PD response,
reaching 0.00854m of real motion — but jaw2 read back as exactly `0.00000`
at every single logged step, with zero response to jaw1's substantial real
motion.** This was reproduced identically with the physics scene's solver
explicitly set to `TGS` (PhysX's documented requirement for mimic-joint
support, in case the bare `World()` scene defaulted to the older `PGS`
solver) — no change in result, suggesting the scene was likely already TGS
by default and solver type is not the explanation.

This is reported as a genuine, unresolved discrepancy, not swept under the
static verification: either (a) the mimic constraint genuinely isn't
engaging in this specific bare test-rig scene for a reason not yet
isolated (a PhysX scene setting beyond solver type, an articulation-root
property IsaacLab's own env cfgs set that a bare `World()` doesn't, etc.),
or (b) `isaacsim.core.prims.Articulation`'s tensor-based joint-position
readback doesn't correctly reflect a mimic-constrained (driveless) joint's
real simulated state — a readback bug distinct from the physics itself.
**Distinguishing these requires testing inside the actual IsaacLab task
env cfg pipeline** (which already exercises this exact asset via
`tasks/ar4/robot_cfg.py`, with IsaacLab's own `SimulationCfg`/articulation
setup this bare test rig did not replicate) — out of scope to build fresh
here given this task's time budget, and not attempted against the
concurrent `ar4-franka-fixes-transfer` workstream's own env cfgs per this
task's explicit instruction not to touch that workstream's files. Flagged
as the concrete, well-defined next step for whoever picks this back up.

---

## 2. Jaw collision geometry — `convexHull` approximation directly confirmed real (previously "unverified")

Direct traversal of the built stage **with instance-proxy traversal**
(`stage.Traverse(Usd.TraverseInstanceProxies())` — the default
`Usd.PrimRange` predicate does *not* descend into the instanceable
prototype references the URDF importer uses for every mesh, which is why
an initial pass using the default predicate found zero mesh prims under
the jaw paths at all) confirms:

| Prim | `UsdPhysics.CollisionAPI` | `MeshCollisionAPI.approximation` |
|---|---|---|
| `gripper_jaw1_link/.../node_STL_BINARY_` | `True` | `convexHull` |
| `gripper_jaw2_link/.../node_STL_BINARY_` | `True` | `convexHull` |
| `gripper_base_link/.../node_STL_BINARY_` | `True` | `convexHull` |

This directly resolves the "unverified convex-hull approximation" item
named in `CLAUDE.md`'s Platform-pivot rationale and in Hypothesis 3 of the
2026-07-20 root-cause doc — it is real, not assumed.

**What this does *not* resolve**: the authored `approximation="convexHull"`
attribute tells PhysX to compute a convex hull from the referenced
triangle mesh at simulation start — the hull itself is not pre-baked into
the USD file, so its actual vertex/face count (and thus how much it
distorts the jaw's real, possibly-non-convex fingertip surface) cannot be
read from static USD inspection alone; it would require either a live
PhysX cook-and-introspect pass or an offline convex-hull computation
against the raw mesh points (e.g. `scipy.spatial.ConvexHull`) as an
independent check. **Not performed in this task** given the time budget —
flagged as a well-defined, cheap (CPU-only, no GPU needed) follow-up: pull
`gripper_jaw1_link`'s render-mesh points (already confirmed accessible:
1866 points, 622 faces) and compare its real convex hull's face count
against the original mesh's face count. A large reduction would indicate
the fingertip has real concave features (a grip notch/texture) that
`convexHull` washes out, which would matter for `antipodal_grasp_bonus`'s
contact-normal-direction read; a small reduction would suggest the
fingertip is already close to convex and the approximation is largely
harmless.

Franka's own shipped asset was not re-checked here (the 2026-07-20 doc
already established this side is equally unexamined and out of reach
without a separate Isaac Lab installation check) — this task was scoped to
the AR4 asset specifically, per the task instructions.

---

## 3. Link_5/Link_6 missing collision — confirmed at the prim level (not just the build log), fixed with a substitute box

Direct traversal confirms, unambiguously: `link_5/collisions/Link_5_Col`
and `link_6/collisions/Link_6_Col` are **empty instanceable prototype
prims with zero mesh children** — not merely an unresolved-reference
warning at build time (which was already known from today's earlier build
log), but a genuine, permanent absence of any collision geometry at the
USD level for these two links. This matches (and directly confirms at the
USD-prim level, not just the importer log) the finding the concurrent
`2026-07-21-ar4-franka-fixes-transfer-design.md` task flagged as its "Gap
4" during its own build/compatibility smoke test earlier the same day.

Both links' **visual** sub-meshes *did* resolve correctly
(`Link_5_Aluminum`/`Link_5_Motor`/`Link_6_Aluminum`, 12612/2076/2526
points respectively) — only the dedicated `*_Col.STL` collision meshes are
missing from the upstream `annin_ar4_description` checkout at the pinned
commit.

### Fix applied

`scripts/build_asset.py`: added `_add_substitute_link_collision()`, run
for both `link_5` and `link_6` right after import. It computes each
link's combined visual-sub-mesh bounding box **dynamically, in the link's
own local frame** (via `UsdGeom.XformCache.ComputeRelativeTransform`, not
a hardcoded constant — so this stays correct if the upstream mesh checkout
ever changes), and authors a `UsdGeom.Cube` collider (an analytic PhysX
primitive shape, no mesh-approximation risk) sized/positioned to that box,
with `UsdPhysics.CollisionAPI` applied and `purpose="guide"` (a
collision-only proxy, not meant to render — the links' own real visual
meshes are untouched).

Measured boxes (link-local frame, for reference — not hardcoded in the
fix itself):
- `link_5`: center `(0.0012, 0.0035, -0.0073)`, size `(0.089, 0.053,
  0.097)` m
- `link_6`: center `(0.0, 0.0, -0.0085)`, size `(0.048, 0.032, 0.017)` m

Re-verified by re-opening the rebuilt USD: both
`/mk5/root_joint/link_5/substitute_collision_box` and
`.../link_6/substitute_collision_box` exist as `Cube`-typed prims with
`CollisionAPI=True`. This closes the concrete blind spot flagged by the
concurrent transfer-fixes task: `arm_ground_contact_penalty`'s
ground-collision safety sensor previously had zero collision geometry to
detect a strike at exactly these two wrist links; a live behavioral
"arm no longer clips through its own body at these links" check was not
performed (would require a full scene + contact sensor setup beyond this
task's time budget) — the fix is verified at the asset/schema level
(collider present, correctly sized/positioned, correctly typed) but not
yet exercised in a live rollout.

---

## Summary table

| Defect | Prior status | Now | Fix applied? |
|---|---|---|---|
| 1. Jaw-mimic constraint | "Confirmed never enforced" (3/3 fixes failed, mechanism never isolated) | A real, correctly-referenced `PhysxMimicJointAPI` exists; the bug is a joint-limit/gearing mismatch, not absence of the constraint | Yes — jaw2 limits recomputed from jaw1's limits under the existing gearing/offset; statically verified, live dynamic confirmation inconclusive (see §1) |
| 2. Jaw collision approximation | "Genuinely inconclusive, never inspected" | Confirmed real: `convexHull` on jaw1/jaw2/gripper_base | No fix — convexHull is standard practice (matches this project's own die-asset convention); hull-distortion severity itself not yet quantified (follow-up identified) |
| 3. Link_5/Link_6 missing collision | Found in today's earlier build log, "non-blocking," not yet fixed | Confirmed at the USD prim level: genuinely zero collision geometry | Yes — substitute box collider added, dynamically sized from each link's own visual bounding box; verified present and correctly typed in the rebuilt USD |

---

## Sources

`scripts/build_asset.py` (this task's own edits — `_fix_gripper_jaw2_mimic_limits`,
`_add_substitute_link_collision`), `scripts/bake_die_asset.py` (the
established direct-USD-introspection pattern this task followed),
`docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`
(Hypotheses 2/3, the questions this task answers directly),
`docs/superpowers/specs/2026-07-21-ar4-franka-fixes-transfer-design.md`
(Gap 4 - Link_5/6, the same-day independent confirmation this task
verified at the prim level), `docs/cloud/franka-cloud-shakedown.md` (the
cloud provisioning/install recipe used), live-built
`assets/ar4_mk5/ar4_mk5.usd` on a GCP `g2-standard-4`+`nvidia-l4` SPOT
instance (`us-central1-c`, 2026-07-21, torn down at task end — not
committed, `assets/` is gitignored project-wide), the
`annin_ar4_description` URDF checkout at
`https://github.com/Annin-Robotics/ar4_ros_driver.git@6a3ebb11cedab12cd1d29b41d63aa270a008ab8b`.
