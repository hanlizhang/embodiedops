# Phase 1C attempt — stopped before learned action generation

**Historical attempt.** The separately approved 2026-10-18 retry completed one
600-action learned-policy rollout. See [phase1c-retry.md](phase1c-retry.md) for
the current result and continuous video. The original artifacts below are preserved.

Date: 2026-10-17. This attempt did **not** produce a learned-policy rollout.
One seed-42 CloseFridge environment was initialized and explicitly reset once.
The pinned model and saved processors loaded successfully, but a guard in the
new runner raised before `select_action`. There were **zero inference calls,
zero environment steps, and zero learned actions executed**.

The failed guard was `assert "action" not in batch`. LeRobot 0.6.1's
`processor/converters.py::transition_to_batch` always emits an `action` key;
its value is `None` for observation-only input. This was an error in our guard,
not evidence of demonstration-action leakage or a model/simulator mismatch.
The guard now checks `batch.get("action") is None`. The official converter
was checked independently without creating an environment or loading a policy.
The corrected runner has **not** been relaunched, in accordance with the
requested stop on runtime exception. The existing run marker prevents an
accidental second invocation over these results.

## Repository and exact pins

Before changes, `git status` reported a clean `main`, up to date with
`origin/main`. `git rev-parse HEAD origin/main` and a live
`git ls-remote origin refs/heads/main` agreed on
`a1d8a6f8fa5fd64a241024bb69681fd0fd858910`:
`Verify RoboCasa controller and MuJoCo contract`.
Phase 1A commit `14c62a8` was also present. Phase 1B was already pushed.

| Component | Exact version/revision |
|---|---|
| Policy | `lerobot/smolvla_robocasa` |
| Policy revision | `f7af588d11e30b0069f200792fb285904b74c80b` |
| SmolVLM backbone | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| LeRobot | `0.6.1` |
| RoboCasa | `1.0.0`, `56e355ccc64389dfc1b8a61a33b9127b975ba681` |
| robosuite | `1.5.2`, `aaa8b9b214ce8e77e82926d677b4d61d55e577ab` |
| MuJoCo | `3.3.1` |
| Torch | `2.11.0+cu128` |
| NumPy | `2.2.5` |

The 494 installed/editable RoboCasa Python files and 204 installed robosuite
Python files were byte-compared against the existing pinned source archives;
all compared files matched. No packages were installed or changed.
Only `D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe` was used.
The run reused `outputs/phase1a/hf-cache` offline and
`outputs/phase1b/numba-cache`. No repeating faulthandler timeout was scheduled.

## Official observation and action path inspected

The installed `lerobot/scripts/lerobot_eval.py` rollout, lines 274–305, calls:

1. `lerobot.envs.utils.preprocess_observation(observation)`.
2. Attach the environment's natural-language task as a one-element list.
3. `make_env_pre_post_processors(RoboCasaConfig(), policy.config)` environment
   preprocessor. RoboCasa inherits the empty/identity pipeline.
4. Saved policy preprocessor from `make_pre_post_processors`, using the exact
   pretrained model path and revision above.
5. `policy.select_action(batch)` under inference mode.
6. Saved policy postprocessor, followed by the empty environment postprocessor.
7. Convert the selected CPU tensor to NumPy and step the environment.

The runner follows this sequence for one non-vectorized environment; the
official helper already supports adding the batch dimension in this case.
There is no custom image transpose, image normalization, or state reordering.

The intended tensor boundaries, established from the inspected official code
and Phase 1A, are:

| Boundary | Contract |
|---|---|
| Simulator cameras | Three RGB HWC `(256,256,3)` uint8 arrays under `pixels` |
| Official helper | `observation.images.<raw_camera>`: `[1,3,256,256]` float32, divided by 255 |
| Simulator state/helper | `agent_pos (16,)` → `observation.state [1,16]` float32 |
| Saved rename | left → camera1; right → camera2; eye_in_hand → camera3 |
| Saved processing | Add batch if needed; task newline; tokenizer length 48; CUDA; saved STATE MEAN_STD, VISUAL IDENTITY |
| Model preparation | Three `[1,3,512,512]` images in `[-1,1]`; padded `[1,32]` state; `[1,48]` tokens/mask |
| Sampler/public chunk | Internal `[1,50,32]` → normalized `[1,50,12]` |
| Saved postprocessor | ACTION MEAN_STD inverse, then CPU; final `[1,50,12]` |

The policy loaded with native BF16/FP32 parameters, `use_amp=false`, chunk size
50, `n_action_steps=50`, and 10 flow integration steps. The saved camera map,
16D/12D configuration and processor JSONs matched Phase 1A. Checkpoint JSON and
normalization-file SHA256 hashes are in `outputs/phase1c/manifest.json`.
Both normalization safetensor files have SHA256
`fa18666b4621709db49c7d29108801e4ddd11c666609323413148a6c2328a620`.

The live observation passed through the saved processor before the faulty
assertion. However, the complete postprocessor-output/model-input shape gate
and its tensor report were **not reached**. The table above is the expected
contract, not a claim that all those tensors were measured in this attempt.

## Actual reset evidence

The reset state, controller configuration/classes/scaling parameters, action
splits, and full qpos/qvel matched the Phase 1B evidence (state tolerance 1e-7;
qpos/qvel tolerance 1e-9, no relative tolerance). The three saved camera images
were visually inspected: left/right show the open fridge and robot; the wrist
view looks toward the lower fridge/front and floor. These retain the established
LEFT / RIGHT / WRIST semantics.

Instruction: `Close the fridge doors.` State:

```text
[3.027994871, -2.792414188, 0.700000048,
 0, 0, -0.000001949, 1,
 0.257287681, -0.014911915, 0.592192054,
 -0.990781128, -0.031070113, -0.129879445, 0.022775136,
 0.020534920, -0.020531641]
```

Initial simulator time: 0.5000000000000003 seconds after upstream reset settling.
The factory performs its own initialization/reset work; this is separate from
the single explicitly seeded episode reset and is not learned-policy execution.

| Fridge joint | Initial and final qpos (rad) | Normalized opening |
|---|---:|---:|
| `fridge_right_group_fridge_left_door_joint` | 1.5251428505026177 | 0.9709360943150206 |
| `fridge_right_group_fridge_right_door_joint` | 1.4408749700006889 | 0.9172894954119842 |

The actual pinned `_check_success()` returned **false**, both initially and
finally. There was no intervening physics step. Initial array byte hashes:

- qpos: `c97454489d119ba49c96acda5f06b15f2c74f1e8e31107ae82b59d23edcb89be`
- qvel: `5721022093504d6ee5a8b4cb0562ea38ca3745ab2facbc7aae1fb62fe72844ee`

## Requested rollout measurements and their limits

| Measurement | Actual result |
|---|---|
| First generated chunk shape/range | Unavailable: sampler never called |
| First five learned actions | Unavailable: no learned actions generated |
| First-action controller/MuJoCo trace | Unavailable: `env.step` never called |
| Official queue semantics | Source confirms queue of 50, pop one per call; replan when empty |
| Observed queue/replanning behavior | Not exercised |
| Policy inference calls | 0 |
| Simulator action steps | 0 |
| Simulated episode duration | 0 seconds beyond reset |
| Wall-clock attempted loop | 0.1113587 seconds; excludes model loading/environment initialization |
| Inference latency | No samples |
| Peak Torch CUDA allocated | 928,648,704 bytes (about 885.6 MiB) |
| Peak Torch CUDA reserved | 952,107,008 bytes (908 MiB) |
| Gradients present | false |
| Final actual success | false |
| Termination reason | Local observation guard AssertionError, before inference |

CUDA numbers are process allocator peaks recorded after the rollout peak reset,
with the loaded model resident; they are **not rollout inference memory** and do
not include all non-Torch rendering allocations. No approach, handle movement,
gripper operation, contact, or other learned behavior can be reported: none ran.

## Runner safeguards prepared but not exercised

`scripts/phase1c_rollout.py` instruments the same `set_goal` and `_pre_action`
boundaries as Phase 1B. It retains controller input/targets and all 25 MuJoCo
control vectors for the first action. It adds no clipping to the validated
12D controller path, logs out-of-Box values and mode/gripper interpretation,
and checks finite state/targets/controls, substep count/time, warning counters,
and the Phase 1B conservative 5 cm one-step translation limit.

A local `SingleEpisodeEnv.step` uses the existing official conversion and
formatting while omitting LeRobot's terminal auto-reset; its reset method also
rejects a second explicit reset. A local loop limit uses the verified registered
600-step horizon. Upstream package code and controller logic are unchanged.
Success/termination/guard failure/exception ends the attempt. The official
`select_action` queue is used without changing its checkpoint settings.

## Artifacts and review boundary

All runtime artifacts are under `outputs/phase1c/`:

- `manifest.json`, `policy.json`: exact run pins, hashes, loaded configuration.
- `reset.json`, `reset-cameras.png`, three `reset-robot0_*.png` images.
- `final-cameras.png`, three `final-robot0_*.png`: same unchanged reset scene.
- `summary.json`: exception, zero action/inference counts, state, memory.
- `steps.jsonl`: empty because no step was executed.
- `rollout.mp4`: reset-only video, **not a learned rollout video**.
- `run-started.json`: retained single-attempt marker.
- `postmortem-checks.json`: offline converter/video/syntax checks; no relaunch.
- `review.diff`: complete diff for the two new review files.

Execution used `& D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe -u scripts\phase1c_rollout.py`
from `C:\Users\Alan\Projects\embodiedops`.
The manifest hashes the executed script version with the erroneous guard;
the postmortem file hashes the corrected, unexecuted version.

Only the runner and this report are new source-controlled candidates. Phase 1A,
Phase 1B, dependencies and `RESUME_EVIDENCE.md` are unchanged. No training,
backward, optimizer, benchmark, second episode, or automatic commit occurred.
Phase 1C remains incomplete; retrying the corrected runner is left for review.
