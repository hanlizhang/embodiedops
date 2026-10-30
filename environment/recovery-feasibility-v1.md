# Failure-state recovery feasibility v1 — stopped, incomplete

2026-10-23. **Four valid VLA recovery branches were executed on one development
state; none recovered or contacted the fridge. The second scene failed the
registered camera restoration gate before inference. The expert attempt is
invalid for feasibility inference because its approach direction was reversed.**
This does not resolve whether successful corrective actions exist. No learning
method is yet justified by this pilot, and no training or retry was launched.

## Registered benchmark and preservation

Two authentic, single-door, failed pretrained task-ID rollouts were registered
before branch outcomes were available. Both are development scenes already used
in prior diagnostics; neither is genuinely held out. They were selected for a
stalled partial closure (48) and late incomplete closure (43), not for recovery
outcomes. Seed 43 is a difficult, substantially open state, not near completion.

| State | Original rollout | After action | Opening | Prior 50-action context |
|---|---|---:|---:|---|
| 48 | instruction-control-v1 / A-ID-48 | 500 | 34.391894° | Door stalled |
| 43 | instruction-control-v1 / A-ID-43 | 550 | 79.306390° | Late incomplete closure |

The machine-readable registration contains exact qpos/qvel, base/EEF poses,
returned 16D state, preceding-window progress and original artifact paths. Source
rollouts remain in the closed instruction study; that study was not resumed.
Pretrained checkpoint SHA256:
`96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b`.
A/B/C model hashes and original processors were checked before execution.
All 147 inventoried source-rollout/model/processor files remained unchanged.
This is an enumerated preservation check, not a claim to have rehashed every
historical file in the repository.

New artifacts are exclusively in
`D:/EmbodiedOpsOutputs/recovery-feasibility-v1`, through verified junction
`outputs/agent-workflow-v1/recovery-feasibility-v1`. Initial free storage was about
C:30.19 GiB / D:167.25 GiB. Each physical action checked the existing 2 GiB reserve
via Win32 and filesystem readings. No dependencies, original checkpoints,
README, demo, CV or website were changed. Reserved seeds 120–123 were unused.

## Restoration audit

Historical per-action records are **not full simulator snapshots**. They do not
contain all MuJoCo integration/warm-start/applied-force state, all mutable
controller internals, observable caches, renderer lifecycle state or RNG state.
The existing display-only qpos/qvel replay is therefore not a causal restoration
mechanism. This pilot never assigned recorded door angles or robot joint states.

Instead, each process reconstructed the seeded scene and physically executed
the entire recorded action prefix through the original controller. Reset
qpos/qvel/state, scene metadata, controller configuration and cameras were checked.
Every prefix action checked qpos/qvel and returned 16D observations. The complete
prefix regenerates controller targets, gripper accumulation, simulator history
and observation lifecycle; it is more expensive than a snapshot. NumPy reset RNG
was controlled; no policy sampling occurred during reconstruction. Recovery
explicitly reset policy queues and replanned at the registered boundary.

| Audit | Prefix + control actions | Maximum qpos / qvel / state error | Measured worker section |
|---|---:|---|---:|
| 48 | 500 + 50 | 0 / 0 / 0 | 77.703 s |
| 43 | 550 + 50 | 0 / 0 / 0 | 63.015 s |

Both matched controls passed: **1,150 actions with exact recorded physical and
16D state equality**. This establishes measured replay fidelity under this path,
not a universal bitwise-reproducibility guarantee or a portable full snapshot.
Each later branch also reconstructed and checked its prefix independently.

The inherited camera tolerance was max pixel difference ≤1 and mean absolute
difference ≤0.001 on each uint8 256×256×3 image. Sparse one-level differences were
observed. The first seed-43 candidate's right-camera mean was **0.0037231445**,
exceeding that limit; its max difference was still 1. Its physical prefix passed,
but the camera gate stopped it **before inference and before recovery actions**.
No threshold relaxation, image substitution on a failed gate, or retry occurred.

For the four valid seed-48 policy calls, identical audited RGB/16D arrays were
used after independently checking each branch's actual observation against that
reference. Some valid branch images differed sparsely within tolerance; the
policy input itself was identical. Rendering determinism remains unresolved.

## Frozen action comparison and actual results

Four independent recorded Flow Matching noise seeds: 930001–930004. Each native
noise tensor is retained, with a distinct byte hash. Same pretrained checkpoint,
`CloseFridge` instruction, saved preprocessing/normalization, 10 flow steps,
50×32 internal noise and 50×12 public actions. Three RGB cameras and 16D robot
state are the only observation inputs. No privileged door/contact data entered
the policy. An oracle best-of-four is only an offline diagnostic.

Each branch had a **50-action / 2.5-second** recovery budget, equal to the matched
control. It is a short feasibility horizon, not a test of eventual recoverability.
Inherited finite-state, action-conversion, camera, ≥5 cm one-action motion and
MuJoCo-warning guards remained enabled. An added 1,500 N normal-contact ceiling
was fixed in advance. Useful progress required ≥5° closure and peak force within
`1.25 × control + 25 N`. True success used the actual all-door task predicate.

| State / branch | Actions | True success | Final opening | Closure | Peak fridge / self force N | EEF path m |
|---|---:|---|---:|---:|---:|---:|
| 48 control | 50 | No | 34.389184° | 0.002710° | 0 / 0 | 0.3246 |
| 48 VLA 0 | 50 | No | 34.389184° | 0.002710° | 0 / 130.137 | 0.1538 |
| 48 VLA 1 | 50 | No | 34.389184° | 0.002710° | 0 / 0 | 0.1415 |
| 48 VLA 2 | 50 | No | 34.389184° | 0.002710° | 0 / 0 | 0.2698 |
| 48 VLA 3 | 50 | No | 34.389184° | 0.002710° | 0 / 0 | 0.1112 |
| 48 expert — invalid direction | 50 | No | 34.389184° | 0.002710° | 0 / 0 | 0.1245 |
| 43 control | 50 | No | 76.254831° | 3.051559° | 130.172 / 0 | 0.1793 |
| 43 VLA 0 | 0 recovery | Invalid camera gate | — | — | — | — |
| 43 VLA 1–3 / expert | Unrun | Unrun | — | — | — | — |

Seed 48: zero loaded fridge-contact actions in every executed branch. The tiny
door change exactly matches the no-contact control, so it is not evidence of
useful manipulation. VLA 0 had 15 actions with ≥1 N self-contact and exceeded the
25 N relative force ceiling, despite remaining below the absolute hard stop.
No candidate meets the useful-progress criterion, and oracle best-of-four task
success is 0 on this **one** evaluated state. Four unsuccessful draws do not prove
the model distribution contains no recovery action.

Seed 43 control had 34 loaded fridge-contact actions and best closure 3.720967°;
it did not succeed. No candidate outcome exists for comparison on this scene.
Invalid/unrun branches must not be counted as robot failures.

## Expert implementation error and repair

The pilot used actual collision geometry and hinge axis/anchor, a nearest-surface
approach, then a tangential closing push within 12 mm clearance. Commands were
base-frame OSC translation increments of at most 1 cm, divided by the installed
0.05 m scale; base/torso and rotation deltas were zero, gripper open, achieved
goal mode. Actions used the normal 12D conversion and physics controller.

The executed expert wrongly assumed `mj_geomDistance(hand, door)` always returned
the hand point first. In this pinned MuJoCo 3.3.1 mesh/box case, native CCD returns
points in geometry-type-sorted order. The resulting approach vector was reversed.
Its clearance increased **0.250725 → 0.370098 m** and it never entered the push
stage. This is an implementation failure, **not evidence of unreachable geometry,
insufficient robot authority, or failure of a correctly directed expert**.

A static mesh/box fixture, with **zero integration steps**, reproduced the issue
in both query orders. The [pinned upstream source](https://raw.githubusercontent.com/google-deepmind/mujoco/3.3.1/src/engine/engine_support.c)
sorts geometry types and returns early from native CCD before restoring endpoint
order. The local repair queries in canonical type order and explicitly restores
caller order. Tests pass for both query orders with native CCD and its fallback.
No simulator flag or dependency was changed in the experiment. The test-only
fallback switch is confined to the static fixture.

The repaired expert has **not** been physically rerun. Executed source snapshots
and original raw result files are preserved. Raw `valid=true` for the expert
means it completed the execution guards; `comparison.json` explicitly marks it
invalid for expert-feasibility inference. The repair was made after the frozen
run ended and is not retroactively attributed to that run.

## Decision and one conditional learning proposal

The requested A/B/C distinction is **unresolved**. There is no sampled useful
candidate on the only evaluated state, no valid successful expert correction,
and no valid second-scene branch comparison. Neither learned outcome selection,
corrective imitation nor residual RL is currently supported as the next actual
learning run. In particular, do not label this Case C: the expert was invalid.

**One conditional follow-up proposal: corrective imitation learning**, only if
a separately approved corrected expert pilot demonstrates safe true recovery.
This is a proposal, not an outcome justified by the present expert attempt:

- First establish trustworthy cameras and successful corrected physical expert
  actions on policy-induced failures. Preserve failed attempts; collect only
  verified successful corrections as positive imitation targets. Stop if that
  prerequisite fails; do not train on the invalid expert recording.
- Proposed data budget: at most 20 successful recovery trajectories from at
  least 10 development scenes, capped at 40 collection attempts. Keep whole
  scenes and related failure prefixes disjoint from held-out evaluation. Store
  genuine RGB/16D/12D sequences, instructions, timestamps and action masks.
- Proposed learning budget: one 300-update action-expert/state-projection run,
  original normalization and frozen VLM, 50% recovery / 50% ordinary episode
  samples. Compare with a matched 300-update ordinary-data-only baseline and
  the frozen original policy. Fixed validation chooses checkpoints before robot
  outcomes; no 5,000-step run and no hyperparameter sweep.
- Before collection, freeze 10 genuinely held-out recovery states from at least
  five new scenes and 10 independent full-task scenes. Do not reuse these two
  development states or reserved 120–123 as held-out evidence. Match observation,
  instruction, noise and controller budgets across models; report paired true
  success, per-door progress, fridge/self forces, guard rates and latency.
- Accept only paired task-success improvement without lost baseline successes,
  new guard failures, >1° per-door regression or contact-force regression under
  the existing gate. This small pilot still would not establish broad reliability.
- Expected demo: authentic matched before/after rollouts in which the **updated
  learned policy**, using RGB/proprioception/language only, completes recovery
  where its frozen baseline fails, plus full-task held-out verification. A
  privileged expert or oracle video alone is not learned-policy improvement.

Call a later experiment DAgger-style only after policy-state collection, expert
correction, dataset aggregation and policy updating have actually all occurred.
None of those learning claims is established by this pilot.

## Execution accounting and artifacts

Two control-audit processes plus six branch processes were attempted. Four VLA
branches are valid; one expert recording is interpretation-invalid; one branch
stopped at its camera gate. Four planned branches remain unrun. Total commanded
robot actions: **4,450**, including reconstruction prefixes, 100 matched-control
actions and 250 attempted recovery actions. There were only **four policy
inferences**, totaling 6.438 s; maximum Torch CUDA allocation was 971,354,112 bytes
(renderer/other allocations excluded). Branch-driver wall time was 624.157 s;
the two audit measured sections totaled 140.718 s. Active GPU time, energy and
complete process RAM were not measured. No timeout or disk-floor breach occurred.

Five recovery tests and seven existing contact/replanning tests passed. The
static geometry test added after the pilot is distinct from preflight validation;
the missing preflight endpoint-order test allowed the expert defect.

- [Frozen benchmark](../outputs/agent-workflow-v1/recovery-feasibility-v1/registered-plan.json)
- [Restoration audit](../outputs/agent-workflow-v1/recovery-feasibility-v1/audit-summary.json)
- [Comparison CSV](../outputs/agent-workflow-v1/recovery-feasibility-v1/comparison.csv)
- [Full comparison and validity](../outputs/agent-workflow-v1/recovery-feasibility-v1/comparison.json)
- [Actual recordings](../outputs/agent-workflow-v1/recovery-feasibility-v1/recordings.html)
- [Static endpoint audit](../outputs/agent-workflow-v1/recovery-feasibility-v1/geometry-endpoint-audit.json)
- [Execution/preservation status](../outputs/agent-workflow-v1/recovery-feasibility-v1/branch-status.json)

These local large artifacts are ignored by Git. The study remains stopped and
incomplete; the definition of done is not met. Further physical attempts require
a new bounded authorization rather than silently replacing failed attempts.
