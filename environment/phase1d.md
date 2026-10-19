# Phase 1D — real SmolVLA training and paired CloseFridge evaluation

2026-10-19. **300 optimizer updates completed on the local RTX 3060 Laptop.** Paired task success:
pretrained **0/10**, trained **3/10** using the actual RoboCasa predicate.
These results distinguish successful training execution from successful robot control.
Evaluation status: **10/10 planned seed pairs completed**.
One task-relevant fridge door: pretrained **0/7**, trained **3/7**.
Two task-relevant fridge doors: pretrained **0/3**, trained **0/3**.
Trained successes: seed 44 at 267 actions, seed 47 at 422, seed 49 at 413.
Seed 48 ended at 5.298 degrees and **failed** the actual predicate.
The observed sample improvement is 30 percentage points. Ten paired seeds and one
training run do not establish statistically reliable generalization. The three
discordant pairs favor training; an exact two-sided paired sign test is p=0.25.
Overlap between the checkpoint's original pretraining data and our CloseFridge
training data or evaluation scenes is unknown. Episode-disjoint validation only
establishes separation within this local dataset split, not pretraining independence.
The fixed held-out loss worsened from 0.531797 to 0.594187; the fixed training
probe worsened from 0.301324 to 0.352550. This run does not demonstrate lower loss.
No dependencies changed, no original checkpoint or Phase 1A/1B evidence changed,
no RESUME_EVIDENCE update, and no commit.

## Language control (one additional seed-42 episode)

`CloseFridge` closed the right door, but not the left: final angles **87.867° / ~0°**;
actual task success remained false after 600 actions. The prior natural-language
run ended at **87.265° / 80.183°**. Both reset qpos/qvel hashes match exactly.
Base net displacement changed from 0.1405 m to 0.0040 m; EEF world path length
changed from 1.658 m to 3.158 m. The language control used open-gripper commands
and achieved-mode commands throughout (the original used 55 desired-mode commands).
This is one observed language sensitivity result, not a success-rate estimate.
Video: `outputs/phase1d/language-control-native/rollout.mp4`.

An earlier sandbox initialization was interrupted before reset or any action;
its artifacts remain in `outputs/phase1d/language-control`. Native execution
completed normally. No additional prompt sweep was performed.

## Exact training contract

- LeRobot 0.6.1; Torch 2.11.0+cu128; existing `embodiedops-robocasa` environment.
- Policy: `lerobot/smolvla_robocasa` at `f7af588d11e30b0069f200792fb285904b74c80b`.
- Dataset: `pepijn223/robocasa_CloseFridge` at `b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4`.
- RoboCasa 1.0.0 source: `56e355ccc64389dfc1b8a61a33b9127b975ba681`.
- robosuite 1.5.2 source: `aaa8b9b214ce8e77e82926d677b4d61d55e577ab`; MuJoCo 3.3.1.
- Backbone: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Native supported switches: `freeze_vision_encoder=true`, `train_expert_only=true`,
  `train_state_proj=true`. **99,880,992 / 450,046,176** parameters trainable,
  in 155 tensors. No custom architecture, PEFT installation, or gradient checkpointing.
  The entire VLM is frozen; the action expert, state projection and action/time
  projections are trainable.
  SmolVLA's installed configuration exposes no gradient-checkpointing switch;
  PEFT is optional and absent. Freezing fit comfortably without either.
- Batch 1, workers 0, seed 42, native BF16/FP32, `use_amp=false`, no augmentation.
- Checkpoint's AdamW preset: peak LR 1e-4, betas (0.9, 0.95), eps 1e-8,
  weight decay 1e-10, gradient clipping 10. The native scheduler scales its
  1000/20000 warmup/decay prescription to 15/300 steps; initial LR 6.25e-6,
  final decay target 2.5e-6. No loss-driven hyperparameter changes.
- Episode split: 461 training episodes (139724 frames),
  52 validation episodes (15719 frames),
  disjoint. Native split takes the last ceil(10%) episodes per instruction.
  Exact episode lists and 300 deterministic shuffled training sample indices are saved.
  The updates touched 221 training episodes and no validation episodes.
  This is only 300 sampled transitions, not a full epoch. Pretraining overlap is unknown.
- Left camera → camera1; right → camera2; wrist → camera3. Original processor JSONs
  and state/action normalization tensors are byte-identical in saved checkpoints.
  The stock CLI replaces normalizer statistics from the selected dataset;
  this runner instead reuses the saved processors while calling LeRobot's actual
  dataset, policy, optimizer, update and checkpoint functions.

Each training sample supplies three RGB images: collated `[1,3,256,256]`, uint8
converted to float [0,1]; SmolVLA internally resizes to `[1,3,512,512]` and rescales
to [-1,1]. State is `[1,1,16]`, internally last-time-selected and padded to 32.
Language tokens/mask are length 48. Actions are `[1,50,12]` with `[1,50]` padding
mask; the model pads actions to 32. Native flow matching samples noise and time,
forms `x_t = t*noise + (1-t)*actions`, predicts velocity, and minimizes squared
error against `noise-actions`, over the 12 real action dimensions and valid
episode positions. `update_policy` runs forward, backward, gradient clipping,
AdamW, zero_grad and scheduler.step. No forward-only substitute was used.

The initial smoke attempt stopped before forward because this runner incorrectly
expected a camera time axis. Actual decoded BCHW images match the installed
`prepare_images` contract; only that assertion was corrected. The zero-update
attempt remains in `outputs/phase1d/training`; the successful run is `training-v2`.

## Measured training and checkpoint evidence

Smoke loss **0.175941**, gradient norm **5.194081**;
all 155 intended tensors had finite gradients, with no missing gradients.
The action output projection changed in **8,640** elements,
max absolute change 6.25103712e-06; the sampled frozen parameter stayed equal.
All 300 updates passed gradient checks. Peak allocated/reserved CUDA memory:
**1.685 / 1.801 GiB** (Torch allocator, not whole-device use).
Training loops took **188.70 s**, including probes/checkpoint writes but excluding
imports, dataset setup, initial loss probes and separate reload processes.
`outputs/phase1d/training-timing.json` records UTC starts from the original run markers and
completion-summary write timestamps from filesystem metadata, including setup time.
Mean synchronized optimizer update: **0.343 s**.
First/last 50 stochastic update loss means: **0.486479 / 0.490504**;
these differ in samples/noise and are not a paired loss comparison.

An exact final-vs-pretrained tensor audit found 122 changed tensors,
all in the intended trainable set; every frozen tensor remained equal.
The 33 trainable expert normalization-weight tensors remained byte-identical
despite receiving gradients under the retained native precision. No precision
or architecture change was made to force additional updates.

The following probes use eight fixed training frames and eight fixed held-out
frames, with fixed flow-noise seeds. They are small diagnostic probes, not a
full validation-set score or evidence of task success.

| Step | Fixed training probe loss | Fixed validation probe loss |
|---:|---:|---:|
| 0 | 0.301324 | 0.531797 |
| 50 | 0.542885 | 0.637462 |
| 100 | 0.439539 | 0.557320 |
| 150 | 0.443148 | 0.579879 |
| 200 | 0.357384 | 0.579646 |
| 250 | 0.358370 | 0.610496 |
| 300 | 0.352550 | 0.594187 |

Full checkpoints: `outputs/phase1d/training-v2/checkpoints/000001`, `000150`, `000300`.
Steps 1 and 300 were reloaded in separate processes: all **500 state tensors**
matched SHA256 hashes exactly, processors matched byte-for-byte, and fixed
validation losses matched within rtol 1e-5 / atol 1e-6. Optimizer, scheduler and RNG
state are saved; continuation loaded the verified step-1 training state.
Final model SHA256: `0e011aa494e0d4e370b11feb5eec6e0c49af2fadd1f3f97ee03056909de0fe6d`.
Local sustained training succeeded, so a Nebius fallback was not required.

## Paired robot evaluation

Both checkpoints use each scene's native instruction (singular `Close the fridge door.`
or plural `Close the fridge doors.`); only weights differ within each pair.
The separate `CloseFridge` language control is excluded from this comparison.
The original seed-42 natural-language baseline was reused. Other episodes use
the same pinned scene generator, 600-action horizon, 20 Hz control, 25 physics
substeps/action, 50-action queue and 10 flow steps. Before each trained run,
reset qpos/qvel and state must match its paired baseline exactly. All three camera
views must match within Phase 1C's existing sparse one-level rasterization tolerance
(maximum 1/255, mean at most 0.001/255). New scene metadata also matches when present;
seed 42 additionally matches the original Phase 1B scene metadata. An initial
zero-action evaluation attempt rejected four one-level RGB differences in the
right view under an overly strict exact-pixel check; it is preserved in
`evaluation/trained-seed42-reset-rounding`. The first learned action
and all subsequent actions retain the Phase 1C guards. Success is RoboCasa's
`_check_success()`, requiring all relevant fridge doors closed. Some seeded scenes
have one task-relevant fridge joint; its angle is in the Other/single column,
with left/right marked N/A (a separate freezer door is not relabeled as a fridge door).
No success surrogate from loss.

Another zero-action attempt, `evaluation/trained-seed45-metadata-order`, differed
only in the order of `receptacle` / `stackable` category labels. RoboCasa constructs
`groups_containing_sampled_obj` by iterating a group dictionary populated from a
Python set (`kitchen_objects.py:2962`, `kitchen_object_utils.py:488`). The comparator
now sorts only this membership-label list; it preserves all poses, asset paths,
controller values and other list ordering. Raw scene metadata remains recorded.
The evaluation aggregator also accepts each fixture's actual singular/plural
instruction instead of assuming every fixture has two fridge doors; completed
episodes were reused after this reporting assertion was corrected.

| Seed | Checkpoint | Success | Left ° | Right ° | Other/single ° | Actions | Chunk inference s | Episode wall s |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 42 | pretrained | False | 87.265 | 80.183 | N/A | 600 | 0.714 | 83.36 |
| 42 | trained | False | 87.265 | 69.088 | N/A | 600 | 0.715 | 83.29 |
| 43 | pretrained | False | N/A | N/A | 89.988 | 600 | 0.665 | 75.77 |
| 43 | trained | False | N/A | N/A | 89.995 | 600 | 0.765 | 78.81 |
| 44 | pretrained | False | N/A | N/A | 89.963 | 600 | 0.647 | 65.89 |
| 44 | trained | True | N/A | N/A | 0.173 | 267 | 0.821 | 32.20 |
| 45 | pretrained | False | 85.456 | 82.212 | N/A | 600 | 0.622 | 80.52 |
| 45 | trained | False | 85.456 | 75.147 | N/A | 600 | 0.599 | 83.10 |
| 46 | pretrained | False | N/A | N/A | 86.399 | 600 | 0.650 | 74.56 |
| 46 | trained | False | N/A | N/A | 86.285 | 600 | 0.641 | 80.16 |
| 47 | pretrained | False | N/A | N/A | 87.538 | 600 | 0.589 | 79.13 |
| 47 | trained | True | N/A | N/A | -0.366 | 422 | 0.682 | 55.71 |
| 48 | pretrained | False | N/A | N/A | 87.521 | 600 | 0.769 | 120.92 |
| 48 | trained | False | N/A | N/A | 5.298 | 600 | 0.651 | 87.59 |
| 49 | pretrained | False | N/A | N/A | 89.985 | 600 | 0.653 | 91.85 |
| 49 | trained | True | N/A | N/A | 0.397 | 413 | 0.666 | 64.19 |
| 50 | pretrained | False | 83.143 | 87.778 | N/A | 600 | 0.700 | 116.59 |
| 50 | trained | False | 83.143 | 86.217 | N/A | 600 | 0.670 | 117.57 |
| 51 | pretrained | False | N/A | N/A | 82.444 | 600 | 0.681 | 73.09 |
| 51 | trained | False | N/A | N/A | 82.444 | 600 | 0.807 | 95.25 |

Episode wall times exclude initialization; chunk latency includes instrumentation
and CPU copies, so these are diagnostic timings, not throughput benchmarks.
Videos include reset plus every action endpoint. Exact paths, inference calls,
physics steps, simulated duration and exceptions are in `evaluation/results.json`.
See `outputs/phase1d/comparison.html` for paired before/after videos.
This small seeded evaluation does not establish generalization or causality.

## Final offline audit

`outputs/phase1d/final-audit.json` verifies exactly 20 unique complete episodes,
with portable summaries in `environment/phase1d-audit.json` and `phase1d-results.json`.
seeds 42–51 paired by checkpoint. Summaries, sequential action logs, stop reasons,
model identities, native language, reset guards and final door values agree.
The saved actual `_check_success()` flags also agree with the pinned source:
`ManipulateDoor._check_success` calls `Fridge.is_closed`, which requires every
task-relevant normalized fridge joint to be at most 0.005. Visual near-closure
is not success. No policy or simulator was executed during this review.

All 20 MP4s decoded completely with strictly increasing timestamps: reset plus
every executed action, 768x288 three-camera frames. All 11 previously completed
episode directories (487 files) match pre-resume inventories including file bytes,
sizes and modification times. The interrupted seed-47 directory/log and prior
aggregate results/config remain preserved with matching inventories:
`trained-seed47-interrupted-before-reset`, its `.log`,
`results-before-seed47-resume.json`, `evaluation-config-before-seed47-resume.json`.
The interruption occurred before reset/inference and is excluded from success rates.
Step-300 weight SHA256 and all 500 tensor hashes were rechecked offline against
the saved reload target; 122 intended tensors changed, all frozen tensors agree
with the original. The 461/52 episode split is disjoint and covers all 513 episodes;
all 300 logged updates use training episodes and have finite loss/gradients.

## Reproduction and review

Exact code: `scripts/phase1d_train.py`, `scripts/phase1d_rollout.py`,
`scripts/phase1d_evaluate.py`, `scripts/phase1d_report.py`.
Exact resolved training config: `environment/phase1d-training-config.json`.
This provenance copy retains the original machine's absolute paths; it is not
a portable CLI configuration. Scripts resolve caches relative to the checkout.
Raw manifests, step logs, split, losses, gradient/weight checks and reload evidence
are under `outputs/phase1d/training-v2`.

```powershell
conda activate embodiedops-robocasa
# Run from the repository root. Historical commands below are not executed by reporting.
python scripts/phase1d_train.py --mode smoke --output outputs/phase1d/training-v2
python scripts/phase1d_train.py --mode verify --output outputs/phase1d/training-v2 --checkpoint outputs/phase1d/training-v2/checkpoints/000001
python scripts/phase1d_train.py --mode continue --output outputs/phase1d/training-v2 --checkpoint outputs/phase1d/training-v2/checkpoints/000001
python scripts/phase1d_train.py --mode verify --output outputs/phase1d/training-v2 --checkpoint outputs/phase1d/training-v2/checkpoints/000300
python scripts/phase1d_evaluate.py --checkpoint outputs/phase1d/training-v2/checkpoints/000300 --seeds 42 43 44 45 46 47 48 49 50 51
python scripts/phase1d_report.py
```

These record the executed workflow. Exclusive run markers prevent overwriting
existing experiments; choose a new output directory for any separately authorized
repeat. Native simulator execution was needed because sandbox initialization stalled.
