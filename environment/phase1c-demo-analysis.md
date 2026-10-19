# CloseFridge demonstration versus policy analysis — 2026-10-18

The recorded failure differs from the selected demonstrations primarily in its
approach geometry and sustained lateral door sweep. Open-gripper commands are
normal in these demonstrations. The current evidence does **not** establish a
need for fine-tuning or identify a single root cause.

All analysis used cached recordings and numeric data. No simulator, policy
inference, additional episode, training, dependency change or commit was run.
Only small public provenance files were fetched; the checkpoint and its settings
were not changed. Existing evidence and `RESUME_EVIDENCE.md` remain unchanged.

## Dataset audit and completion evidence

Dataset: `pepijn223/robocasa_CloseFridge`, revision
`b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4`.

- All **513 episodes / 155,443 rows** are present in the cached Parquet file.
  All three cached H.264 streams also contain 155,443 frames at 20 Hz, 256×256 RGB.
- Every episode's index range, length, zero-based frame indices, 0.05-s timestamps,
  annotation consistency and three camera intervals were checked. No missing or
  nonfinite 16D states / 12D actions were found. The only split is `train: 0:513`.
- Task description and `task_index` agree on every row: 168 episodes say
  **“Close the fridge doors.”**, 345 say **“Close the fridge door.”**.
  `annotation.human.task_name=2` always resolves to `CloseFridge`.
  `total_tasks=2` counts instructions; the three-row lookup also includes the
  task-name annotation. These IDs are local to this dataset.
- All 513 episodes contain binary reward 1 and end with reward 1. Each has exactly
  one `next.done=true`, at its final row. There are 8,312 reward-positive frames;
  positive-frame counts per episode range from 16 to 35.
- **There is no separate success label, door-angle trace, model XML, fixture pose,
  original simulator state or contact record in this pinned export.** The Hub file
  inventory contains only numeric data, metadata and videos; no `extras/` files.
  A README is absent at this revision (HTTP 404).

Thus all episodes are **reward-labeled as completed**, but this analysis does not
independently certify all 513 simulator success predicates. The pinned RoboCasa
source defines sparse reward as `float(_check_success())`; `CloseFridge` checks
the fixture's closed state. Its conversion utilities also distinguish end-of-file
`done` from task success. The exact historical conversion run is not documented
by this dataset, so current source semantics alone do not prove label provenance.
For the three selections below, visible closure corroborates the positive labels.
No precise demonstration door angle is inferred from the images.

## Selected complete demonstrations and visible strategy

Selection was purposive, not a random performance sample: episode 0 preserves the
previously inspected reference; 437 is the closest initial relative EEF position
among plural-instruction episodes; 5 is the first episode with appreciable base
movement in episode order and illustrates the single-door variant.

| Episode | Instruction / visible fixture | Frames / last timestamp | First reward-positive row | Visible sequence |
|---|---|---:|---:|---|
| 0 | Plural / two doors | 429 / 21.40 s | 413 / 20.65 s | Approach right door's outside around 3–6 s; right door swings closed around 6–11 s; transfer left around 13–16 s; left door closes around 18–21 s |
| 437 | Plural / two doors | 484 / 24.15 s | 468 / 23.40 s | Move around right door around 3–9 s; sweep it closed around 9–15 s; transfer toward left around 15–21 s; both visibly closed by the reward-positive tail |
| 5 | Singular / one door | 300 / 14.95 s | 284 / 14.20 s | Reposition base early; route arm around the outside of the open door around 5–9 s; swing it shut around 9–14 s |

“Right” and “left” here mean the robot's sides while facing the refrigerator,
identified visually; demonstration body names were not saved. Times are visual
intervals, not force/contact timestamps. In all three recordings the open fingers
approach the outer face/edge, sometimes near the handle, and follow the closing
door. This is consistent with **pushing**, not a commanded handle grasp: all three
command opening throughout. Images cannot establish the exact contact patch or
exclude incidental contact with a handle. The arm routes around the door perimeter
and crosses the opening during repositioning; successful behavior does not require
the arm to remain outside the opening at every instant. The distinction is the
sustained closing sweep versus the policy's prolonged interior-directed motion.

All three camera views were inspected at ten timestamps per demonstration,
including the first positive-reward row and final frame. Full continuous exports
are available for closer inspection; these observations do not claim a frame-by-
frame contact annotation.

## Action and motion comparison

The existing controller contract is unchanged. Actions are controller commands:
`[base vx, base vy, base yaw, torso, mode, EEF dx/dy/dz, EEF rot xyz, gripper]`.
Base commands are local-frame velocity inputs; EEF translation/axis-angle commands
are base-frame deltas. The verified controller scales translation by 0.05 m and
rotation by 0.5 rad after clipping. Mode `<0.5` updates relative to achieved pose;
mode `>=0.5` updates retained desired goals. Gripper `<0.5` opens; `>=0.5` closes.
Command magnitude is **not** achieved displacement, especially against obstacles.

| Measurement | Demo 0 | Demo 437 | Demo 5 | Policy seed 42 |
|---|---:|---:|---:|---:|
| Recorded span, seconds | 21.40 | 24.15 | 14.95 | 30.00 |
| Base net displacement, m | 0.00014 | 0.00280 | 0.39340 | 0.14053 |
| Sampled world EEF path length, m | 3.262 | 3.736 | 2.942 | 1.657 |
| Base-relative EEF lateral range, m | 0.889 | 0.982 | 0.729 | 0.193 |
| Mean EEF translation command norm | 0.651 | 0.645 | 0.635 | 0.207 |
| Mean EEF rotation command norm | 0.112 | 0.200 | 0.150 | 0.034 |
| Desired-mode commands | 0% | 0% | 17.0% | 9.17% |
| Close-gripper commands | 0% | 0% | 0% | 0% |

Both two-door demonstrations command zero base/torso motion; small observed base
drift is physical response. Demo 5 uses desired mode on frames 32–82 (1.60–4.10 s)
while repositioning, then achieved mode. All demonstrations have zero torso commands.
Across the whole dataset, 57/513 episodes contain a base-command norm above 0.01;
desired mode accounts for 2.72% of frames. Only four episodes (77, 179, 373, 512)
ever command gripper closing: 319/155,443 actions, or 0.205%. Consequently neither
base movement nor an open gripper is itself a fault.

The policy's translation/rotation commands are smaller on average than these
examples (whole-dataset means: 0.697 / 0.177), and it covers much less lateral
range despite running longer. This describes behavior; it is not evidence of a
controller scaling error. Episode duration, scene geometry, motion phase and
idle frames affect the means. Full per-dimension min/quantiles/mean/std for all
12 commands and 16 states, plus 50-frame demonstration summaries, are saved.

The policy's twelve 50-action chunks show where closing progress was lost:

| Chunk / action endpoints | Base net, cm | EEF net, cm | Right door angle at end, degrees |
|---|---:|---:|---:|
| 1 / 1–50 | 0.02 | 11.73 | 82.54 |
| 2 / 51–100 | 8.57 | 11.04 | 82.53 |
| 3 / 101–150 | 2.53 | 6.85 | 82.52 |
| 4 / 151–200 | 0.02 | 6.80 | 82.51 |
| 5 / 201–250 | 2.07 | 8.49 | 82.50 |
| 6 / 251–300 | 1.01 | 9.23 | 79.44 |
| 7 / 301–350 | 0.01 | 8.89 | 80.03 |
| 8 / 351–400 | 0.01 | 12.50 | 80.02 |
| 9 / 401–450 | 0.03 | 12.65 | 80.21 |
| 10 / 451–500 | 0.08 | 5.30 | 80.20 |
| 11 / 501–550 | <0.01 | 8.16 | 80.19 |
| 12 / 551–600 | 0.01 | 11.61 | 80.18 |

Door angles come from the existing logged joints, not newly simulated states.
The left door remains approximately 87.38→87.27°. The right moves only
82.56→80.18° overall and partially reopens after chunk 6. Continued EEF movement
in chunks 7–12 does not produce sustained closing or a left-door phase.
Previously computed geometry shows near-zero right-door clearance around
steps 250–272, followed by increasing separation; it does not prove contact force.

## Reset state, images and coordinate frames

The initial images share camera roles, robot embodiment and a fridge-facing start,
but differ in door style, cabinet/wall textures, clutter and opening geometry.
The seed-42 dark doors and objects inside the fridge differ from the selected
gray-door demonstrations. No visual novelty score or claim that these appearances
are absent from training is justified by three examples.

Every seed-42 reset component lies within the **513-episode initial-state range**.
The table reports marginal percentiles, not a joint distribution/OOD test:

| State component | Seed-42 reset | Percentile among episode starts |
|---|---:|---:|
| base x, world m | 3.027995 | 47.4 |
| base y, world m | -2.792414 | 36.3 |
| base z, world m | 0.700000 | 99.8 |
| base quaternion x | 0 | constant zero |
| base quaternion y | 0 | constant zero |
| base quaternion z | -0.000002 | 23.8 |
| base quaternion w | 1 | 100.0 |
| relative EEF x, m | 0.257288 | 95.5 |
| relative EEF y, m | -0.014912 | 14.0 |
| relative EEF z, m | 0.592192 | 36.3 |
| relative EEF quaternion x | -0.990781 | 25.7 |
| relative EEF quaternion y | -0.031070 | 51.9 |
| relative EEF quaternion z | -0.129879 | 66.3 |
| relative EEF quaternion w | 0.022775 | 55.4 |
| finger 1, m | 0.020535 | 23.4 |
| finger 2, m | -0.020532 | 23.4 |

Near-constant z has only numerical-scale variation; its high percentile is not a
meaningful height mismatch. Raw quaternion component percentiles are descriptive
only: q and −q represent the same orientation. Orientation distances use normalized
quaternions and `2*acos(abs(dot(q1,q2)))`, so sign flips do not become false errors.
With the checkpoint's unchanged saved normalizer, reset values span only
**−0.9444 to +1.1128**. That does not rule out visual or joint-distribution shift.

Demo 437 starts 5.28 mm / 1.37° from the evaluation's relative EEF pose; episode 51
(single-door instruction, not selected for video) is 2.00 mm / 0.93° away. Base yaw
is roughly +90° in demo 0, −180° in demo 437 and 0° in the evaluation. Absolute
world-coordinate trajectories must therefore **not** be overlaid as if scenes
were aligned. This report uses base-relative EEF ranges and scalar world path/net
lengths; world EEF positions are reconstructed as `base_pos + R(base_quat)*eef_rel`.
No demonstration fixture transforms are available, so fridge-frame alignment,
door/handle distances and exact door trajectories cannot be recovered offline.
The policy's robot-only 16D state likewise does not itself encode fridge geometry.

## Checkpoint coverage and a concrete conditioning hypothesis

The pinned [model card](https://huggingface.co/lerobot/smolvla_robocasa/blob/f7af588d11e30b0069f200792fb285904b74c80b/README.md)
and saved `train_config.json` identify **`lerobot/robocasa_target_human_unified`**
as the training dataset; base model is `lerobot/smolvla_base`. The training config
sets `episodes=null`, `revision=null`, 20,000 steps and batch size 32. It does not
identify the exact training-data commit or establish that these 513 standalone
episodes were included. No training-subset equality or held-out status is claimed.

The training repository's metadata fetched for this review was pinned on retrieval
to `a1c7cba1a128f5dd8d3084012ba366d10ce174d9`: 25,307 episodes, 14,957,899 frames,
50 task names, including **task index 1 → `CloseFridge`**. Its schema lacks the
standalone export's human-description fields. The installed LeRobot 0.6.1 reader
resolves `item['task']` directly from that task table; the saved tokenizer reads
`task` and the saved newline step appends a newline.

This gives a specific **possible language-conditioning discrepancy**:
`CloseFridge\n` in the currently published training representation versus
`Close the fridge doors.\n` in our rollout. It is not a proven historical mismatch:
the training revision is unspecified and the current table could differ from the
one used then. CloseFridge's presence supports stated task coverage, not competence
on this scene. The model card supplies no task-specific performance evidence.

## Evidence versus hypotheses; one next experiment

**Observed:** complete, reward-positive demonstrations visually close doors with
open-gripper sweeps; policy execution remains finite and responsive but emphasizes
interior reach, limited lateral travel and small door motion. Initial proprioception
is not marginally out of range. Training-source identity differs from the comparison
dataset, and current task strings differ from the rollout's natural-language string.

**Unresolved hypotheses:** language/task conditioning, visual/fixture distribution
shift, insufficient learned door-sweep behavior, or loss of useful contact during
50-action open-loop chunks. Prior probes and logged physical response argue against
a completely dead or grossly disconnected action path, but do not exclude subtler
execution/temporal issues. Open fingers, desired-mode use or base motion alone do
not diagnose failure. No evidence establishes confusion with a particular other task.

**Recommend one controlled offline prompt A/B experiment, not executed here:**
use the saved seed-42 reset plus a fixed pre-push observation from each of demos
0 and 437. Keep images/state, checkpoint, normalization, camera mapping, 50-action
chunk settings and the identical sampler-noise tensor fixed. Change only the task
string between `Close the fridge doors.` and `CloseFridge` through the existing
newline/tokenizer path. Compare the resulting chunks; on demonstration observations,
report normalized per-dimension error to the next 50 recorded actions, translation/
rotation direction, and mode/gripper agreement. The reset has no ground-truth action
target. Do not execute actions. A consistent reduction in demonstration error would
support investigating conditioning; static agreement would still not prove task
success. If there is no useful difference, do not infer that fine-tuning is required.

## Deliverables and reproducibility

Run the offline exporter with the existing environment:

```powershell
conda activate embodiedops-robocasa
cd C:\Users\Alan\Projects\embodiedops
python scripts\phase1c_demo_analysis.py
```

Outputs: `outputs/phase1c/demo-analysis/`.

- `demo-000-three-camera.mp4`, `demo-437-three-camera.mp4`,
  `demo-005-three-camera.mp4`: complete three-view videos, episode/frame timestamps,
  reward labels; separate `*-timestamps.json` files map back to source stream PTS.
- `demonstrations-vs-policy.mp4`: four columns, each with left/right/wrist stacked;
  601 frames at 20 Hz. Comparison is by **elapsed time**, not matched task progress.
  Completed demonstrations freeze with an explicit label and their final timestamp.
- `comparison-000.png`, `comparison-285.png`, `comparison-600.png` and demonstration
  contact sheets support reset, intermediate and outcome inspection.
- `episode-audit.json`, `episode-metrics.json`, `analysis.json`, `demo-chunks.json`,
  `policy-chunks.json`, `reset-normalization.json`: full measured numeric evidence.
- `manifest.json`: original-data hashes, script hash and decoded export frame counts.
  All exported videos were decoded again to verify frame counts; selected camera
  source PTS were checked to within 0.2 ms of metadata and 20 Hz spacing.
- Saved model card, retrieved training metadata/task table and provenance checks.

Demo row/frame numbers are zero-based. Policy action endpoint 0 is reset; endpoint
600 is after the 600th command. Demo final observation does not include a separately
saved extra endpoint after its final action. `next.reward` is shown as stored, not
silently shifted or recomputed. No cross-scene action MSE or temporal warping is used
for the existing rollout comparison.
