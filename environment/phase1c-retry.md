# Phase 1C — one guarded learned-policy rollout

2026-10-18, seed 42. **The integration ran for one complete bounded episode,
but CloseFridge was not solved.** The process exited normally at the registered
600-action horizon. No second episode, training, dependency change, or automatic
commit occurred. `RESUME_EVIDENCE.md` is unchanged.

## Result

| Measurement | Observed value |
|---|---:|
| Policy inference calls / generated action chunks | 12 / 12 |
| Executed learned actions | 600 |
| Physics substeps | 15,000 (25 per action) |
| Simulated rollout time | 30.000 s |
| Simulator clock, reset → final | 0.500 → 30.500 s |
| Wall-clock rollout time | 83.355 s |
| First chunk latency | 1.791 s |
| Subsequent chunk latency, mean | 0.616 s |
| Subsequent chunk latency, min–max | 0.496–0.716 s |
| Peak Torch CUDA allocated / reserved | 971,354,112 / 1,008,730,112 bytes |
| Actual final task success | false |
| Runtime exceptions / guard failures | none / none |
| Stop reason | registered 600-action horizon |
| Continuous video | 601 frames, 20 FPS, 768×288, 30.05 s |

The 30.05 s video includes one reset frame plus the 600 post-action frames.
The 30 s simulated duration excludes upstream reset settling. Wall time includes
preprocessing, inference, instrumentation, rendering and logging, but excludes
model loading/environment initialization and final video flush. Inference timings
are synchronized and include the instrumentation's internal output CPU copy;
they are diagnostic measurements, not a throughput benchmark. CUDA figures are
Torch allocator peaks with the model resident, not total rendering/GPU memory.

## Repository, environment and pins

HEAD and local `origin/main` remained
`a1d8a6f8fa5fd64a241024bb69681fd0fd858910`, the committed Phase 1B baseline.
The only initial working-tree additions were the prior Phase 1C runner/report.
No unrelated tracked changes were present. The prior turn had verified this
commit on the live remote; this turn did not fetch or change Git history.

| Component | Exact pin |
|---|---|
| Environment | `D:\Apps\miniconda\envs\embodiedops-robocasa` |
| Policy | `lerobot/smolvla_robocasa` |
| Policy revision | `f7af588d11e30b0069f200792fb285904b74c80b` |
| Backbone revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| LeRobot | `0.6.1` |
| RoboCasa | `1.0.0`, revision `56e355ccc64389dfc1b8a61a33b9127b975ba681` |
| robosuite | `1.5.2`, revision `aaa8b9b214ce8e77e82926d677b4d61d55e577ab` |
| MuJoCo / NumPy | `3.3.1` / `2.2.5` |
| Torch | `2.11.0+cu128` |

The runner checked versions against Phase 1B, reused the offline Phase 1A HF
cache and Phase 1B Numba cache, and used the normal pretrained loading path.
Saved checkpoint processor JSONs, camera renaming, state/action dimensions,
chunk size 50, n_action_steps 50, 10 flow-integration steps and native BF16/FP32
parameters (`use_amp=false`) were retained. No demonstration actions were loaded.
The manifest records config/processor/normalization SHA256 hashes and the executed
runner SHA256 `fb63a7a4f64e8ec7dbaefe21c46d2a1017f93b2643f29335d5256552b2258bb7`.

## Guard review and actual observation contract

The installed LeRobot `processor/converters.py::batch_to_transition` maps a
missing action to `None`; `transition_to_batch` explicitly emits `action: None`.
The corrected `batch.get("action") is None` accepts only absence/None and rejects
every supplied non-None action. It does not skip validation of other fields.

Before this rollout, the actual runner's `prepare` function was checked using
the saved real reset cameras/state and the pinned processors, without creating
a simulator or loading a policy model. Valid input passed. Each of these was
rejected: supplied action, missing camera, missing state, missing language tokens,
wrong task text, wrong state shape, NaN state, image values outside `[0,1]`.
Results: `outputs/phase1c/guard-review-20261018.json`.

Existing camera/state/finite checks remain. Explicit task-text, token/mask dtype
and shape, nonempty attention, padded state/image shape and chunk checks were
added before execution. These assertions validate rather than transform inputs.

The runtime follows installed `lerobot_eval.py`'s processor sequence:

```text
RoboCasaEnv observation
→ lerobot.envs.utils.preprocess_observation
→ task = [env.task_description]
→ make_env_pre_post_processors environment preprocessor (identity for RoboCasa)
→ pinned saved policy preprocessor
→ official policy.select_action
→ pinned saved action postprocessor
→ environment postprocessor (identity)
→ CPU NumPy action, batch dimension removed for the single environment
```

The official helper performs HWC uint8 → batched CHW float32 /255. No custom
transpose, scaling or normalization bridge was introduced. Measured boundaries:

| Field | Actual shape/type/range |
|---|---|
| Three raw policy cameras | `(256,256,3)` RGB uint8 |
| Saved LEFT/right/wrist map | raw left → camera1; raw right → camera2; eye_in_hand → camera3 |
| Processed camera1 | `[1,3,256,256]` float32 CUDA, `[0.031373,1]` |
| Processed camera2 | `[1,3,256,256]` float32 CUDA, `[0.086275,1]` |
| Processed camera3 | `[1,3,256,256]` float32 CUDA, `[0.054902,1]` |
| State | raw `(16,)` → normalized `[1,16]` float32, `[-0.944401,1.112839]` |
| Language | `Close the fridge doors.` plus the saved newline processor |
| Tokens / attention mask | `[1,48]` int64 / `[1,48]` bool, nonempty attention |
| Model images | three `[1,3,512,512]`, finite, within `[-1,1]` |
| Model state | `[1,32]`, finite |
| Internal generated chunk | `[1,50,32]`, finite |
| Public normalized / unnormalized chunk | `[1,50,12]` / `[1,50,12]`, finite |

Camera semantics, 16D state order, instruction and saved normalization matched
Phase 1A. The seed-42 reset matched Phase 1B full qpos/qvel within 1e-9 and state
within 1e-7. Controller classes/configuration/scaling and action splits matched.
Reset qpos/qvel byte hashes are retained in `reset.json` and match the prior attempt.

## First generated chunk and first action

First normalized public chunk range: `[-2.726168, 1.818236]`.
First final unnormalized chunk range: `[-1.028321, 0.547045]`.
The first five 12D learned actions, rounded for readability:

```text
[ .001837,-.002004,-.000095,0,-.990862, .001898, .001496, .002283, .001234, .001080,-.008198,-.996398]
[ .001985,-.001946, .000021,0,-.994670,-.000169, .005245, .006209,-.000344, .002333, .000733,-.994838]
[ .000218,-.000301,-.000403,0,-.995714,-.001744,-.001781,-.002814,-.001423, .006005, .001403,-.991040]
[ .000212,-.000474,-.000021,0,-.993689, .000090,-.002952,-.004892,-.000270, .001350, .000427,-.987291]
[-.000839,-.001438,-.000802,0,-.989069,-.417126,-.116665, .140704, .098055,-.068606, .015530,-.972741]
```

Full precision, per-dimension extrema, padded/normalized/final tensors and
latency/memory are in `chunk-01.json` and `first-chunk.pt`.

The first action was inside the nominal Box, selected achieved-pose mode and
open gripper. The traced reordered controller vector was:

```text
[.001898,.001496,.002283,.001234,.001080,-.008198,
 -1,.001837,-.002004,-.000095,0,-1]
```

EEF goal position in the base frame was
`[0.257382572,-0.014837104,0.592306218]` m. Base velocity goal was approximately
`[0.001836692,-0.002004011,-0.000094581]`. Gripper accumulator became `[0.2,-0.2]`.
The first action advanced simulator time from 0.50 to 0.55 s through exactly
25 physics substeps with 13-dimensional control vectors. The full target matrices
and all 25 control vectors are in `first-action-trace.json`.

Measured first-action world displacement:

- Base: `[0.001513,-0.003114,0.000005]` mm.
- EEF: `[0.005865,-0.008424,-0.003625]` mm.
- Gripper joint changes: `[0.858922,-0.833170]` mm.
- Left/right fridge joint changes: `[-3.473452e-6,-3.721972e-6]` rad.
- Reward 0, success false, terminated/truncated false; finite state,
  zero MuJoCo warning counters and no guard failures. Only then did the loop continue.

Across the full episode, action extrema were `[-1.034794,1.022227]`. The only
out-of-Box dimensions were mode (4) and gripper (11), on 344 steps; their validated
threshold-at-0.5 interpretation handled them. No ad-hoc clipping was added.
545 actions used achieved mode and 55 desired mode. All 600 commanded open gripper.

## Queue behavior, physical response and outcome

The official queue generated a chunk at actions
`1,51,101,151,201,251,301,351,401,451,501,551`. It consumed 50 selected actions per
chunk with queue lengths 49 down to 0. All 600 step records were checked against
this schedule; the model was not called separately to produce inspection chunks.
The sampler instrumentation returned its original tensors unchanged.

Decoded video frames at reset and actions 50,100,200,275,300,450,600 were visually
inspected. The robot changes arm posture, moves the base toward the fridge, and
extends the open gripper into the vicinity of the open right-side door/interior.
It later makes further arm movements there without closing the fridge. The
measured base net displacement is 0.1405 m and EEF net displacement is 0.5125 m
(world delta `[0.49605,-0.09363,-0.08823]` m). Fingers open from approximately
`[0.02053,-0.02053]` to `[0.04001,-0.03985]` m.

The right door moves during the middle of the episode. Video proximity and door
motion are consistent with interaction, but contact forces/pairs were not logged,
so this is not a verified handle grasp or contact-event diagnosis. The left door
remains almost unchanged. Maximum per-action base/EEF translations were
12.665/12.898 mm, below the retained 50 mm diagnostic guard. All observations and
physical states remained finite; no guard failed. This single failure does not
establish its cause or a success rate.

| Door | Initial qpos → final qpos (rad) | Final normalized opening |
|---|---|---:|
| Left | 1.525142851 → 1.523060200 | 0.969610238 |
| Right | 1.440874970 → 1.399462603 | 0.890925564 |

The pinned success predicate requires both normalized openings ≤0.005.
Neither door met that condition. The local horizon guard stopped the single
episode; the simulator itself did not emit terminal/truncated on the final step.
The local wrapper omits LeRobot's terminal auto-reset and rejects a second reset.

## Artifacts and exact execution

Continuous learned-policy video:

`C:\Users\Alan\Projects\embodiedops\outputs\phase1c\retry-20261018\rollout.mp4`

It decoded to 601 frames; all camera regions (excluding text labels) had distinct
decoded frame hashes. File size 1,091,877 bytes; SHA256:
`1cf5686cfd5d9ccc1033a5d22f7a6dd62eb12390f91631c083102c706307bb31`.

Alongside the video: `manifest.json`, `policy.json`, `reset.json`,
`observation-contract.json`, `chunk-01.json` through `chunk-12.json`,
`first-chunk.pt`, `first-action-gate.json`, `first-action-trace.json`,
`first-action-result.json`, `steps.jsonl`, `summary.json`, `verification.json`,
`video-review-contact-sheet.png`, and camera PNGs at reset, first action,
first replan input and final state. `review.diff` contains the complete review diff.
The old failed attempt and its marker remain in the parent directory untouched.

Actual command, from the repository root:

```powershell
& D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe -u scripts\phase1c_rollout.py --output outputs\phase1c\retry-20261018
```

The new `--output` option permits a separately authorized attempt without erasing
old artifacts. Exclusive run markers still prevent overwriting/repeating a run
in the same directory. Do not relaunch this completed attempt for review.

## Separate viewer proposal — not implemented or executed

The Phase 1C runner does not support a live MuJoCo viewer; none was added.
The existing Phase 1B interactive viewer replays its manual probes, not this
learned rollout, so it should not be presented as a learned-policy visualization.

The smallest next option is a video replay UI that pauses after each 50-action
chunk (2.5 simulated seconds), displays the corresponding action/state metadata,
and advances on Enter. It uses this exact recorded episode with no inference,
controller change or physics rerun.

If a rotatable MuJoCo view is desired, propose a separate recorded-state viewer:
reconstruct the matching seed-42 model, verify reset dimensions/hash, load each
logged qpos/qvel into display-only MjData, call `mj_forward` for visualization,
and pause at the recorded chunk boundaries. This requires a model-reconstruction
check and is not yet verified. It would replay measured states rather than call
`select_action` or `env.step`. A new live learned rollout with chunk pauses would
be a separate experiment requiring approval.

Source changes are limited to the uncommitted Phase 1C runner and reports.
No backward pass, optimizer, fine-tuning, benchmark, extra episode or viewer was run.
