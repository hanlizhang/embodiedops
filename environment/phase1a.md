# Phase 1A: SmolVLA / CloseFridge static contract verification

Verified locally on 2026-10-16. Repository baseline: `8cbde4a` on `main`.

The real dataset -> saved preprocessing -> forward-only flow-matching loss ->
static action-chunk pipeline passed. This establishes no task success, rollout
performance, backward-pass feasibility, or fine-tuning result.

No RoboCasa, robosuite, simulator assets, environment rollout, backward pass,
optimizer, or training job was installed/executed. `RESUME_EVIDENCE.md` was not
modified. Both existing environments were left intact. No commit was made.

## Environment and reproducibility

New Conda environment: `D:\Apps\miniconda\envs\embodiedops-robocasa`.

| Component | Actual version |
|---|---|
| Python | 3.12.15 |
| PyTorch / CUDA runtime | 2.11.0+cu128 / 12.8 |
| torchvision | 0.26.0+cu128 |
| LeRobot | 0.6.1 |
| Transformers / tokenizers | 5.5.4 / 0.22.2 |
| Accelerate | 1.15.0 |
| NumPy | 2.2.6 |
| OpenCV provider | opencv-python-headless 4.13.0.92 only |
| datasets / pandas / pyarrow | 4.8.5 / 2.3.3 / 26.0.0 |
| PyAV / torchcodec | 15.1.0 / 0.11.1 |
| huggingface-hub / safetensors | 1.33.0 / 0.8.0 |
| GPU / driver | RTX 3060 Laptop GPU, 6 GB / 610.47 |

All installed distributions are recorded in [phase1a-packages.txt](phase1a-packages.txt).
The Conda-managed binaries are recorded in [phase1a-conda-explicit.txt](phase1a-conda-explicit.txt).
The direct installation constraints are in [phase1a-requirements.txt](phase1a-requirements.txt).
`pip check` passed. Video decoding explicitly used PyAV, not torchcodec.
No system FFmpeg or additional OpenCV provider was installed.

Conda initially could not load its SSL DLLs from the unactivated base executable.
Adding the base Conda DLL directories to that command's process-local PATH fixed
environment creation; no base/environment package repair or upgrade was done.

The exact successful creation command sequence was:

```powershell
$env:PATH = 'D:\Apps\miniconda;D:\Apps\miniconda\Library\mingw-w64\bin;D:\Apps\miniconda\Library\usr\bin;D:\Apps\miniconda\Library\bin;D:\Apps\miniconda\Scripts;' + $env:PATH
& D:\Apps\miniconda\python.exe -B -m conda create --name embodiedops-robocasa python=3.12 pip --yes
```

This used the configured `defaults` channel, with no channel override. The
successful transaction's packages came from `https://repo.anaconda.com/pkgs/main`
(`win-64` and `noarch`), matching every package URL in
`phase1a-conda-explicit.txt`. The earlier conda-forge retry failed before creation;
conda-forge was not the source of the successful environment.

The exact subsequent installation sequence, run from the repository root, was:

```powershell
& D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe -B -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
& D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe -B -m pip install -r environment/phase1a-requirements.txt
```

The first command installed the CUDA wheels `torch==2.11.0+cu128` and
`torchvision==0.26.0+cu128`; the requirements command retained them and installed
`lerobot[smolvla,dataset,evaluation]==0.6.1` with the remaining policy/data stack.
Both pip commands targeted only the new environment.

The explicit Conda export records Conda's package records, not subsequent pip
replacements. In particular, it lists the original `setuptools=84.0.0` and
`packaging=26.3`, while the pip-resolved final distributions are
`setuptools==78.1.0` and `packaging==25.0`, as recorded in
`phase1a-packages.txt`. Thus the explicit export describes the Conda scaffold;
it must be followed by the CUDA/pip sequence above. It alone does not reproduce
the final mixed Conda/pip environment. The complete final Python distribution
snapshot is `phase1a-packages.txt`.

| Artifact | Exact revision |
|---|---|
| `lerobot/smolvla_robocasa` | `f7af588d11e30b0069f200792fb285904b74c80b` |
| `pepijn223/robocasa_CloseFridge` | `b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4` |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |

Normal LeRobot checkpoint loading was used, with no feature, processor-statistic,
precision, or model-configuration overrides. The checkpoint's `load_vlm_weights`
setting requires loading the backbone before the policy weights. Actual parameter
dtypes were BF16 and FP32; `use_amp=false`. This native mixture was preserved.
The forward/inference invocation used `HF_HUB_OFFLINE=1` after caching the files.
The script checks the loaded backbone commit and refuses revision drift.

Only episode 0 was selected. Its 429 frames share storage containers with other
episodes. Required data/video files downloaded were:

| Container | Bytes |
|---|---:|
| `data/chunk-000/file-000.parquet` | 17,388,973 |
| Left camera MP4, chunk-000/file-000 | 105,004,466 |
| Right camera MP4, chunk-000/file-000 | 127,745,455 |
| Wrist camera MP4, chunk-000/file-000 | 159,578,181 |

Small metadata files were also downloaded. These are dataset recordings, not
simulator assets. Full dataset/video download was not requested.

## Loaded checkpoint contract

The actual loaded `input_features` were state `(16,)` and three camera features
`(3,256,256)`. The output action feature was `(12,)`.

| Setting | Loaded value |
|---|---:|
| `chunk_size` | 50 |
| `n_action_steps` | 50 |
| `max_state_dim` / `max_action_dim` | 32 / 32 |
| Flow-integration `num_steps` | 10 |
| Image resize/padding target | 512 x 512 |

The saved preprocessor loads these steps in order:

1. RenameObservationsProcessorStep: left -> camera1, right -> camera2, wrist -> camera3.
2. AddBatchDimensionProcessorStep.
3. NewLineTaskProcessorStep.
4. TokenizerProcessorStep: right padding to 48 tokens, truncation enabled.
5. DeviceProcessorStep: CUDA, no float-dtype override.
6. NormalizerProcessorStep: visual IDENTITY; state and action MEAN_STD; epsilon 1e-8.

Normalizer feature shapes were state `(16,)`, action `(12,)`, and three
`(3,256,256)` images. Actual mean/std vectors had 16 and 12 entries. The saved
postprocessor is UnnormalizerProcessorStep (ACTION MEAN_STD, action `(12,)`,
epsilon 1e-8), then DeviceProcessorStep to CPU. Its saved statistics were retained.
All loaded processor configurations/statistic summaries are in the trace.

The reported historical 6D state inconsistency was **not reproduced at this pinned
revision**: policy config, training config, normalization features/statistics,
dataset metadata, and the measured sample all agree on 16D. No state was cropped,
patched, or overwritten. The checker's initial scalar-count false positive was
corrected: sample counts are scalar metadata, not feature-wise normalization
vectors. This did not change upstream files or tensors.

## Measured sample and camera semantics

Episode 0, frame 0, global index 0, timestamp 0; dataset FPS 20.
Dataset totals: 513 episodes and 155,443 frames. Numeric storage declares float64
state/actions; the normal LeRobot loader returned float32 tensors.

Relevant sample keys: three `observation.images.robot0_*` cameras,
`observation.state`, `action`, `action_is_pad`, both `annotation.human.*` fields,
`task`, `task_index`, `episode_index`, `frame_index`, `index`, `timestamp`,
`next.reward`, and `next.done`.

| Field | Measured value / decoded meaning |
|---|---|
| `task_index` | 0 -> `Close the fridge doors.` |
| `annotation.human.task_description` | 0 -> `Close the fridge doors.` |
| `annotation.human.task_name` | 2 -> `CloseFridge` |

The task table also contains index 1 -> `Close the fridge door.`. Metadata says
`total_tasks=2`, but the table has three entries. The human description was selected
explicitly; for this sample it agrees with the loader's default `task` string.

Measured raw state `(16,)`, float32, range `[-1.443588, 4.103148]`:

```text
[4.103148, -1.443588, 0.700000, 0, 0, 0.707107, 0.707107,
 0.246891, -0.019631, 0.593560, 0.990985, 0.029300, 0.130431,
 0.008829, 0.020537, -0.020529]
```

The raw demonstration chunk is `(50,12)`, float32, range `[-1,0.507143]`.
Its first action is:

```text
[0, 0, 0, 0, -1, 0.057143, 0.171429, 0, 0, 0, -0.037143, -1]
```

The saved contact sheet was visually inspected before enabling model forward:
left/right are two distinct external views of the robot and open fridge; wrist
is the close hand-mounted view. Labels bind actual raw tensors to saved processor
keys: camera1 LEFT, camera2 RIGHT, camera3 WRIST. This is consistent with the
checkpoint's saved training mapping. It is not independent camera calibration.
The conflicting documentation mapping was not used.

### Source-derived semantics, separate from sample measurements

LeRobot 0.6.1's `envs/robocasa.py` constructs state as base position `[0:3]`,
base quaternion `[3:7]`, relative end-effector position `[7:10]`, relative
end-effector quaternion `[10:14]`, and gripper qpos `[14:16]`.
Its action conversion uses base/torso `[0:4]`, control mode `[4:5]`, end-effector
position command `[5:8]`, rotation command `[8:11]`, and gripper `[11:12]`.
These are controller inputs, not twelve joint angles or torques. RoboCasa's
wrapper maps the fourth base-motion entry to torso and thresholds mode/gripper
at 0.5. No simulator controller or physical displacement was measured in Phase 1A.
Dataset metadata does not label individual vector dimensions; matching these
semantics to simulator observations remains a Phase 1B check.

Sources: [LeRobot wrapper](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/envs/robocasa.py),
[RoboCasa wrapper](https://github.com/robocasa/robocasa/blob/56e355ccc64389dfc1b8a61a33b9127b975ba681/robocasa/wrappers/gym_wrapper.py).

## Measured preprocessing boundaries

| Boundary | Shape / dtype | Actual range or detail |
|---|---|---|
| Decoded LEFT | `[3,256,256]` float32 | `[0.043137,1]` |
| Decoded RIGHT | `[3,256,256]` float32 | `[0.078431,1]` |
| Decoded WRIST | `[3,256,256]` float32 | `[0.011765,1]` |
| Collation and renamed images | each `[1,3,256,256]` float32 | Same values/ranges |
| State before normalization | `[1,16]` float32 | `[-1.443588,4.103148]` |
| State after normalization | `[1,16]` float32 | `[-0.794007,1.455431]` |
| Instruction after newline step | one string | `Close the fridge doors.\n` |
| Token IDs | `[1,48]` int64 | First six: `28948,260,28867,9160,30,198`; padding ID 2 |
| Attention mask | `[1,48]` bool | Six valid tokens |
| Resized LEFT | `[1,3,512,512]` float32 | `[0.083333,1]` |
| Resized RIGHT | `[1,3,512,512]` float32 | `[0.099265,1]` |
| Resized WRIST | `[1,3,512,512]` float32 | `[0.046569,1]` |
| Model-scaled LEFT | `[1,3,512,512]` float32 | `[-0.833333,1]` |
| Model-scaled RIGHT | `[1,3,512,512]` float32 | `[-0.801471,1]` |
| Model-scaled WRIST | `[1,3,512,512]` float32 | `[-0.906863,1]` |
| Normalized demonstration chunk | `[1,50,12]` float32 | `[-3.743813,3.184251]` |
| Internal padded state | `[1,32]` float32 | Original 16 normalized values, then zeros |
| Internal padded actions | `[1,50,32]` float32 | Original 12 normalized entries, then zeros |
| Initial padding mask | `[1,50]` bool | All false; all 50 actions valid |
| Last-frame padding mask | `[50]` bool | Frame 428: first false, remaining 49 true |

Every processor boundary was traced using the pipeline's `step_through` API.
The resize helper was evaluated on the actual image tensors, then equality with
the model's `prepare_images` result after `2*x-1` scaling was checked. No model
method was replaced. Forward hooks captured actual action-projection inputs and
velocity outputs without changing them.

## Forward-only flow matching

Fixed seed 42. Demonstration actions `[1,50,12]` are normalized and padded to
`[1,50,32]`. Actual noise, interpolated input, target velocity, and predicted
velocity all have shape `[1,50,32]` and dtype float32.

```text
t = 0.6690568327903748
x_t = t * noise + (1 - t) * action
u_t = noise - action

noise first 4 = [0.194019, 2.161374, -0.172050, 0.849060]
x_t   first 4 = [0.086871, 1.442478, -0.113568, 0.568069]
u_t   first 4 = [0.323765, 2.172264, -0.176714, 0.849060]
v_t   first 4 = [0.299483, 2.177656, -0.050348, 0.837213]
```

`x_t` range `[-2.551562,2.030988]`; `u_t` range `[-3.430533,5.841636]`;
predicted velocity range `[-3.823583,4.868744]`.

The real action-projection input equals the reconstructed `x_t`. Independently
recomputing masked MSE over the 12 real dimensions and 50 valid timesteps agrees
with the actual policy loss: **0.39290663599967957**, finite. There were no parameter
gradients. Execution used `torch.no_grad()`; no backward or optimizer existed.

Training parameterizes the path from the data/action endpoint at t=0 toward noise
at t=1. Inference begins at noise and integrates in the reverse direction toward
t=0. This loss value is one observation/noise draw, not a validation benchmark.

## Static action generation and memory

The public `predict_action_chunk` path and saved postprocessor were called on
the real observation, with fixed noise and no demonstration action passed in.

| Output / measurement | Actual result |
|---|---|
| Internal sampler output | `[1,50,32]` float32 |
| Public normalized chunk | `[1,50,12]` float32 |
| Unpadded, unnormalized final chunk | `[1,50,12]` float32 on CPU; all finite |
| Final action range | `[-1.0274199247,0.7214528322]` |
| First timed chunk call including postprocessing | 0.5618446000 seconds |
| Flow-forward peak allocated / reserved | 942.078 / 982 MiB |
| Static chunk peak allocated / reserved | 940.641 / 984 MiB |

First three generated 12D actions, rounded for readability:

```text
[-0.001137, -0.002643,  0.000004, 0, -1.000330, -0.088317,
 -0.096725, -0.051076,  0.003404, -0.081943, -0.045283, -1.003169]
[ 0.000444, -0.001755, -0.000061, 0, -1.002711, -0.020649,
  0.001457, -0.016083, -0.000435, -0.015333, -0.001706, -0.998448]
[-0.000059,  0.000002, -0.000431, 0, -0.999678, -0.005721,
  0.000554, -0.000606, -0.001373,  0.002872,  0.000703, -0.994569]
```

An additional internal sampler call with identical observation/noise exposed the
padded output and verified that its first 12 dimensions match the public chunk.
The timer covers the first public action-generation call after the preceding
flow forward warmed the runtime; it excludes loading, data decoding, and initial
preprocessing. It is one smoke measurement, not a latency distribution or a
real-time guarantee. CUDA synchronization brackets the timer.
Allocator memory excludes driver/context/display allocations. At the inference
snapshot, CUDA reported 4,309,647,360 free bytes out of 6,441,926,656 total bytes.
No OOM occurred and no precision/configuration workaround was applied.

The final values slightly exceed the nominal -1 lower action bound. They were
not clipped. Simulator/controller boundary behavior must be inspected before
executing generated actions in Phase 1B.

`chunk_size=50` is the generated trajectory length. `n_action_steps=50` is how
many queued actions the public selection path would execute before replanning.
`num_steps=10` is the number of flow-integration iterations for generating a
chunk; it is not ten simulator steps. No actions were executed in a simulator.

## Artifacts and repeat command

All generated artifacts remain under ignored `outputs/phase1a/`:

- `trace.json`: real keys, shapes, dtypes, ranges, first values, configurations,
  decoded annotation mapping, loss, timing, and memory.
- `cameras.png`: labeled actual three-camera contact sheet.
- `flow_tensors.pt`: complete measured noise/time/x_t/u_t/velocity tensors.
- `inference_tensors.pt`: complete internal/public/final action chunks.
- `inspection-console.log`, `forward-console.log`, pinned checkpoint JSON files,
  `packages.txt`, and the required Hub cache.

The verified forward invocation was:

```powershell
$env:HF_HOME = 'C:\Users\Alan\Projects\embodiedops\outputs\phase1a\hf-cache'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_OFFLINE = '1'
& D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe -u -B scripts\phase1a_contract.py --execute-forward
```

Without `--execute-forward`, the script stops after the contract/data trace and
camera sheet. Contract mismatch or CUDA OOM stops execution; no fallback is tried.

## Proposed Phase 1B, not executed or approved

After separate approval, install only the pinned RoboCasa/robosuite sources and
minimal fixture/texture/lightwheel assets in the new environment. First reset one
CloseFridge environment with seed 42 and synchronous rendering, inspect actual
controller configuration, state ordering, camera keys/orientation and task text,
and compare these with the Phase 1A contract. Inspect how action bounds and
mode/gripper thresholds are handled. Then execute five validated neutral-control
steps and verify finite observations/rendered frames. Stop and report.

This minimal simulator smoke test would not include learned-policy rollout,
`lerobot-eval`, training, or a success-rate claim. A bounded learned-policy rollout
and baseline evaluation would be a subsequent explicitly approved step.
