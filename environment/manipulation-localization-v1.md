# CloseFridge: focused failure localization

**Decision:** do not select or pilot a manipulation repair yet. Valid recordings
localize ineffective door interaction, but do not distinguish contact geometry
from controller tracking. A newly discovered RGB blackout also invalidates
seed-42 H20 as a clean horizon comparison. No policy inference, dynamic replay,
training, new rollout, dependency change or commit was performed for this report.
Historical reports/checkpoints remain intact; this is additional evidence.

## Synchronized visual evidence

All artifacts below are in `outputs/manipulation-localization-v1/final/`:

- [30-second synchronized six-panel video](../outputs/manipulation-localization-v1/final/synchronized.mp4):
  rows are seeds 42/43/44, columns H50/H20; each panel retains all three RGB views.
  Every panel uses the same simulated-time index (20 Hz), with explicit terminal
  freezing and black-RGB labels. This is time synchronization, not phase alignment.
- Event sheets: [42](../outputs/manipulation-localization-v1/final/seed42-events.jpg),
  [43](../outputs/manipulation-localization-v1/final/seed43-events.jpg),
  [44](../outputs/manipulation-localization-v1/final/seed44-events.jpg).
  Columns show reset, a fixed 1-second early-approach sample, first recorded loaded
  contact, best summed absolute door opening, and final state. Event columns have
  explicit indices/times; their timestamps differ between runs. H50 contact is
  unavailable, not inferred. Seed42's best opening sums both doors.
- [Selected demonstration phases](../outputs/manipulation-localization-v1/final/demo-phases.jpg)
  and [seed42 display-only replay](../outputs/manipulation-localization-v1/final/seed42-display-replay.jpg).

### Material new finding: seed42 H20 lost RGB

[Original frames 44 and 45](../outputs/manipulation-localization-v1/final/blackout-seed42-H20.jpg)
show a transition from normal images to **three entirely black camera views**.
Frames 45–285 are black: 241 consecutive endpoints, starting at 2.25 s. The saved
final raw camera PNG is also black, ruling out video compression/compositing as
the sole cause. The first replan consuming a black observation is action 61;
actions 46–60 consume the existing queue. The runner uses the same `obs` for its
video and the following policy input; it does not regenerate a separate image.
The first observation-contract file covers only reset, so the actual later
preprocessed input tensors were not independently archived.

No all-black frame was found in the other five rollout videos. The physical
failure and collision remain real recorded outcomes, but attributing this H20
trajectory solely to execution horizon or manipulation competence is unjustified.
The previous finite/range checks accepted zero images, and decoding a video
successfully did not establish valid visual observations. Renderer/driver/context
failure versus another image-production fault remains unresolved. Do not relabel
or silently replace the historical episode.

## Door progress and physical interaction

| Seed / H | Initial → best → final opening (degrees) | Contact evidence / outcome |
|---|---|---|
| 42 / 50 | L 87.38 → 87.25 → 87.87; R 82.56 → ~0 → ~0 | Right first meets closure threshold at action 284. Left remains open. H50 forces unavailable. |
| 42 / 20 | L 87.38 → 86.71 → 86.71; R 82.56 → 82.50 → 82.50 | Arm links 3/4 hit fridge at 281–282; no recorded hand/finger contact. Guard stop at 285, after RGB loss. |
| 43 / 50 | 85.52 → 80.86 → 81.69 | Failed at 600; contact history unavailable. |
| 43 / 20 | 85.52 → 77.81 → 83.35 | Loaded hand/arm–door contact in 345 actions; failed at 600. |
| 44 / 50 | 83.32 → −0.08 → −0.08 | Success at 342. |
| 44 / 20 | 83.32 → 0.22 → 0.22 | Success at 337; hand/fingers interact with door/handle. |

H20 first loaded contact endpoints: 42 at 281 (14.05 s), 43 at 77 (3.85 s),
44 at 210 (10.50 s). These labels locate the enclosing 50-ms action; contacts
are actually sampled every 2 ms. Counted loaded contacts require an active
constraint address and positive normal force, not proximity or overlap.

Seed43's contacts involve `gripper0_right_hand_collision` and arm links 5/6/7,
including door geoms g47/g33/g58. Saved body ancestry identifies those as moving
door geometry. No finger contact is recorded. Peak single-contact normal force
is 324.24 N. Door-angle changes sum to −7.22 degrees during contact-containing
actions, versus only −2.17 degrees overall: progress is partly lost. Its best
opening occurs at action 486; it subsequently reopens by 5.54 degrees.

Seed44 has 120 contact-containing actions, including hand–handle followed by
finger–door/handle contacts, with peak single-contact normal force 78.57 N.
Door-angle changes sum to −76.85 degrees on those actions. Open-gripper commands
persist; contact with fingers is not proof of a commanded grasp.

Seed42 H20 has a 2.27 kN peak link4–fridge constraint force and 5.22 cm EEF
displacement at the later guard stop. Display-only replay places its hand outside
the opening near the oven while the arm contacts the fixture. It is not evidence
that an EEF-in-fridge mechanism explains all failures.

## Coordinate-frame and tracking checks

Only seed42 has an existing validated compiled scene. Its model/name/hash checks,
exact reset qpos/qvel, and every recorded hinge angle pass for both new replay
trajectories. EEF replay-to-cached-observation differences peak at **0.524 mm H50**
and **2.003 mm H20**; endpoint kinematics and cached observations are not exactly
simultaneous. A declared 5-mm acceptance bound was used, and fine geometric
inferences must respect those errors. `mj_step`, `mj_step1`, and `mj_step2` were
replaced with rejecting stubs. `mj_forward` only updates recorded-pose kinematics;
its reconstructed constraint forces are never interpreted as historical forces.

For each door, recorded-pose measurements use:
`p_door = R_door^T (p_EEF_world − p_door_world)`, the model's actual world hinge
anchor/axis, and gripper-to-door collision-geometry signed clearance. Body-frame
coordinates are not camera axes or an automatically inferred "inside" label.
Full values are in `seed42-geometry.json`.

| Seed42 H50 phase | Gripper surface clearance | EEF radial distance to hinge |
|---|---:|---:|
| Right sweep, action 200 (53.86° open) | −0.274 mm | 0.386 m |
| Right near closure, action 281 (1.61°) | −0.095 mm | 0.382 m |
| Left late approach, action 578 (87.25°) | −0.755 mm | 0.394 m |

These small overlaps are geometric measurements only. H50 does transfer toward
the left door: a complete absence of stage transition is not supported. It
arrives within 1 cm of the open left door only for actions 576–600.

For that late interval, project the clipped/scaled translation command, rotated
from the recorded base frame, onto the closing tangent
`−axis × (p_EEF − hinge)` (decreasing joint angle). Mean requested projection is
**+5.09 mm/action**, versus **−0.159 mm/action achieved**; 88% of requested
projections point toward closure. This is not a simple consistently reversed
closing command. The tangent at the EEF is a geometric reference, not a measured
contact force or torque at the actual contact patch.

For achieved-mode actions in seed43 H20 with loaded contacts, mean requested
translation norm is **20.80 mm**, versus **2.19 mm** achieved relative displacement.
Without recorded loaded contact: 17.75 vs 4.37 mm. Successful seed44 H20 also
shows a gap during contact: 18.12 vs 3.72 mm. Commands specify controller goals,
not guaranteed displacement over 50 ms; the gap alone does not diagnose bad gains.
The saved controller snapshots validate the 0.05 m scaling/base-frame calculation
to <1e−6 m. Full desired-goal history, servo errors and actuator saturation are
not saved; desired-mode actions are excluded from these approximations.

Seeds43/44 lack validated fixture models/transforms, so moving-door/hinge-relative
EEF trajectories and surface clearance cannot be recovered reliably from their
16D state alone. Contact forces are in contact-local axes; missing per-substep
contact-frame orientation prevents world force and effective hinge-torque
calculation even for seed42. No H50 contact history was reconstructed.

## Successful demonstrations and decision

Existing episodes **0, 437 and 5** were selected previously, not newly cherry-picked
for this result. Reward-positive tails and visible closure support their use as
examples, with the existing dataset-provenance caveat. The phase sheet shows:
0/437 route around a door, sweep it closed, transfer, then close the other;
5 repositions the base before the closing sweep. All three command open fingers.
The visual difference is sustained productive closing motion, not merely whether
the EEF ever crosses the opening. Their fixture poses, door angles and forces are
absent from the export; no world-coordinate overlay or exact contact alignment
with policy scenes is justified.

**Localized failure:** valid runs reach/approach the target door but fail to
convert commanded motion into sustained closure (43, and late left-door phase
of 42/H50). **Unresolved cause:** incorrect contact side/leverage, geometric
obstruction, insufficient controller tracking, or some combination. Successful
44 demonstrates that contact and the same controller can support closing; it
does not isolate which difference is causal. Seed42/H20 is separately compromised
by RGB loss.

An action limiter would not establish restored manipulation, gain changes might
increase load, and targeted imitation would assume a policy cause not yet
isolated. Therefore **no repair or pilot was implemented**, as permitted by the
task's evidence-insufficiency stop condition. The implemented code is the bounded
offline analysis, not a diagnosis framework or a claimed repair. No final
evaluation seed set is selected without a supported pilot.

## Minimal acquisition plan and reproducibility

Before another policy trial, test the seed42 image-production path using saved
states only, recording per-camera nonzero/variance counters and renderer errors.
Never replace missing observations with display-only renders and call that a
deployable policy result.

Then acquire **one seed43 H20 baseline**, capped at 600 actions, unchanged pinned
weights/processors/instruction/controller/10 flow steps, with read-only logging:
compiled model + geometry inventory; every substep contact frame/position/wrench;
hinge axis/anchor; commanded EEF target and achieved pose in the same frame;
controller mode, joint/actuator limit and torque saturation; observation-image
integrity. This is an evaluation-only oracle measurement plan, not an oracle
policy input. It should distinguish ineffective closing moment from tracking
failure before choosing one intervention. It has **not been run**.

Commands (existing `embodiedops-robocasa` environment, repository root):

```powershell
python -B -m unittest discover -s tests -p test_manipulation_localization.py -v
python -B scripts/localize_manipulation.py --output outputs/manipulation-localization-reproduction
```

Output directory must not exist. Tests reject proximity/inactive constraints as
loaded contacts and detect black cameras despite a bright video title. No policy
is imported. Source: `scripts/localize_manipulation.py`; tests:
`tests/test_manipulation_localization.py`. Numeric evidence, source hash and
preservation checks are in the final artifact folder. The prior two diagnostics
physics tests were not rerun. All historical source artifacts are preserved.
