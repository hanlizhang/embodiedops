# Phase 1B: simulator and controller contract

Repository starting point: `14c62a8` on `main`, initially clean.
Scope: simulator-only inspection, five neutral-command steps, and bounded manual
probes. No learned policy, Phase 1A rerun, evaluation, backward pass, optimizer,
or training. `RESUME_EVIDENCE.md` is intentionally unchanged.

## Installation

Only `D:\Apps\miniconda\envs\embodiedops-robocasa` was modified.

| Component | Pin |
|---|---|
| RoboCasa | 1.0.0, `56e355ccc64389dfc1b8a61a33b9127b975ba681` |
| robosuite | 1.5.2, `aaa8b9b214ce8e77e82926d677b4d61d55e577ab` |
| MuJoCo | 3.3.1 |
| NumPy | 2.2.5 |
| Numba / llvmlite | 0.61.2 / 0.44.0 |
| SciPy | 1.15.3 |
| LeRobot | retained 0.6.1 |
| PyTorch / torchvision | retained 2.11.0+cu128 / 0.26.0+cu128 |
| OpenCV | retained opencv-python-headless 4.13.0.92; sole provider |

The exact final distribution versions are in `phase1b-packages.txt`. The source
ZIPs were downloaded from the commit-specific GitHub archive URLs and extracted
under `outputs/phase1b/sources/`. Archive hashes are recorded in
`outputs/phase1b/archive-hashes.json` and `phase1b-results.json`. The 21 exact
simulator package pins (20 additions plus NumPy) are in `phase1b-requirements.txt`;
the two upstream sources must still be installed separately with `--no-deps`.
All 494 installed RoboCasa Python files and 204 installed robosuite Python files
were byte-compared with their respective pinned source archives and matched.
The robosuite wheel omits nine optional example/USD files, listed in the results.

The source download commands were:

```powershell
Invoke-WebRequest 'https://github.com/robocasa/robocasa/archive/56e355ccc64389dfc1b8a61a33b9127b975ba681.zip' -OutFile outputs/phase1b/sources/robocasa.zip
Invoke-WebRequest 'https://github.com/ARISE-Initiative/robosuite/archive/aaa8b9b214ce8e77e82926d677b4d61d55e577ab.zip' -OutFile outputs/phase1b/sources/robosuite.zip
Expand-Archive outputs/phase1b/sources/robocasa.zip outputs/phase1b/sources
Expand-Archive outputs/phase1b/sources/robosuite.zip outputs/phase1b/sources
```

RoboCasa's setup.py requests LeRobot 0.3.3, NumPy 2.2.5, opencv-python, and
tianshou 0.4.10. Both simulator sources were installed with `--no-deps`.
LeRobot's obsolete pin was not applied; headless OpenCV supplies cv2. Tianshou
is not needed by this simulator path and was not installed. These declarations
remain visible as `pip check` metadata conflicts; this is not a clean dependency
resolution claim. RoboCasa also asserts NumPy 2.2.5 at import, so the initial
attempt with 2.2.6 stopped at that assertion. NumPy alone was changed to 2.2.5,
which satisfies LeRobot, OpenCV, SciPy, and Numba requirements. A metadata-based
comparison verified that every other pre-existing package version was retained.

Exact command sequence from the repository root (PowerShell):

```powershell
$py = 'D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe'
& $py -B -m pip freeze | Set-Content outputs/phase1b/before-packages.txt
& $py -B -m pip install --dry-run --report outputs/phase1b/dependency-plan.json -c outputs/phase1b/before-packages.txt mujoco==3.3.1 numba==0.61.2 scipy==1.15.3 'qpsolvers[quadprog]>=4.3.1' pygame pynput imageio h5py lxml hidapi pytest
```

The dry-run report was converted into `outputs/phase1b/simulator-dependencies.txt`
containing exact `name==version` entries for its 20 additions. Then:

```powershell
& $py -B -m pip install -c outputs/phase1b/before-packages.txt -r outputs/phase1b/simulator-dependencies.txt --report outputs/phase1b/install-report.json
& $py -B -m pip install --no-deps --no-build-isolation outputs/phase1b/sources/robosuite-aaa8b9b214ce8e77e82926d677b4d61d55e577ab outputs/phase1b/sources/robocasa-56e355ccc64389dfc1b8a61a33b9127b975ba681
& $py -B -m pip install --no-deps --no-build-isolation -e outputs/phase1b/sources/robocasa-56e355ccc64389dfc1b8a61a33b9127b975ba681
Get-Content outputs/phase1b/before-packages.txt | Where-Object { $_ -notmatch '^numpy==' } | Set-Content outputs/phase1b/protected-packages.txt
& $py -B -m pip install --dry-run --report outputs/phase1b/numpy-plan.json -c outputs/phase1b/protected-packages.txt numpy==2.2.5
& $py -B -m pip install -c outputs/phase1b/protected-packages.txt numpy==2.2.5 --report outputs/phase1b/numpy-install.json
```

The editable reinstall is necessary because RoboCasa's wheel omits shipped
non-Python assets/configurations. Its source directory must remain present.
`pip freeze` incorrectly infers the enclosing EmbodiedOps repository as the
editable package's Git provenance; use the explicit upstream SHA above, not
that inferred Git line. Upstream Python source was not patched.

Numba cache creation inside site-packages stalled under the filesystem sandbox.
The diagnostic uses process-local `NUMBA_CACHE_DIR=outputs/phase1b/numba-cache`
and runs with approved filesystem access. This does not alter simulator math.

Official asset registry URLs at the pinned source:

| Pack | Box shared-static ZIP ID |
|---|---|
| fixtures_lw | idbncsadpnaz1jfl4i6m8qejawk7p9pi |
| tex | 4i85ileasdvstmlln5sbvzptz7keuoy1 |
| objs_lw | vckqvvkh1z8t69k8qcpcmee6k66stii4 |
| tex_generative | ebaad09k82tmfmlq6ohdkmrh8izl9vn5 |

Each URL is `https://utexas.box.com/shared/static/<ID>.zip`, downloaded with
`Invoke-WebRequest -OutFile outputs/phase1b/sources/<pack>.zip`. Fixtures and
textures extract under RoboCasa's `robocasa/models/assets/`; the object archive
extracts under `assets/objects/`. Only the Lightwheel object registry is used.
HEAD returned 404 for fixture/texture archives, but GET succeeded. The first
model-load attempt identified a required generative wall texture; that pack is
needed even with generative texture randomization disabled.

## Verification results

Verified on 2026-10-17 (Europe/Berlin): reset, rendering, exactly five public
neutral-command steps and four one-step probes completed. All nine returned
finite observations, reward 0, success false, terminated false, truncated false.
There is no task-success or learned-policy performance claim.

### Reset and observation contract

Runtime task: `robocasa.environments.kitchen.atomic.kitchen_doors.CloseFridge`.
Layers: LeRobot `RoboCasaEnv` -> `RoboCasaGymEnv` -> `CloseFridge`; robot
`WheeledRobot` / `PandaOmron`; fixture `FridgeFrenchDoor`. Split `all`, episode
index 0, seed 42, only the Lightwheel object registry. The actual instruction is
`Close the fridge doors.`. Full scene/fixture/object metadata is in `reset.json`.

Public observation keys are `pixels` and `agent_pos`. The raw image keys are
`robot0_agentview_left_image`, `robot0_agentview_right_image`, and
`robot0_eye_in_hand_image`. RoboCasa flips the OpenGL image vertically, exposes
`video.<camera>`, and LeRobot exposes `pixels[<camera>]`: each `(256,256,3)` uint8
RGB, declared range [0,255]. Phase 1A's checkpoint processor mapping remains
left -> camera1, right -> camera2, wrist -> camera3. The reset contact sheet was
visually compared with the Phase 1A dataset contact sheet: matching external
left/right viewpoints and hand-mounted wrist view. The kitchen scene differs;
this does not claim identical images or independent extrinsic calibration.

Raw observations also include robot joint position/cos/sin/velocity/acceleration,
EEF position and quaternion (including site quaternion), gripper qpos/qvel,
base pose, relative EEF pose, object poses and object-to-EEF poses, and
`robot0_proprio-state` / `object-state`. Every exact key is saved in `reset.json`.

`agent_pos` is `(16,)` float32 with declared unbounded range:

| Slice | Meaning |
|---|---|
| 0:3 | Robot root/base position in world coordinates, metres; z is ~0.7, not ground height |
| 3:7 | Base quaternion, xyzw |
| 7:10 | EEF position relative to base, metres |
| 10:14 | EEF quaternion relative to base, xyzw; the wrapper uses the non-`_site` observation |
| 14:16 | Two gripper joint positions, metres, opposite joint-axis signs |

Measured reset state:

```text
[3.027994871, -2.792414188, 0.700000048,
 0, 0, -0.000001949, 1,
 0.257287681, -0.014911915, 0.592192054,
 -0.990781128, -0.031070113, -0.129879445, 0.022775136,
 0.020534920, -0.020531641]
```

The public action space is `Box(-1,1,(12,),float32)`. RoboCasa creates its
reordered controller vector with `np.zeros`, so that boundary is float64.
Neither Gym Box nor either wrapper clips or checks bounds on these values.

### Action contract and controller configuration

The exact path is LeRobot `convert_action` / `RoboCasaEnv.step` ->
`PandaOmronKeyConverter.unmap_action` -> `RoboCasaGymEnv.step` ->
robosuite `MujocoEnv.step` / robot `WheeledRobot.control` ->
`HybridMobileBase.set_goal` -> per-part `set_goal` and `run_controller` ->
actuator-range clipping -> `sim.data.ctrl` -> `MjSim.step1/step2` -> MuJoCo.

All public dimensions declare [-1,1]. The table describes the actual transforms:

| Public index | Command | Controller handling |
|---|---|---|
| 0 | Base local x translation velocity command | Legacy base frame transform; final velocity actuator range ±1 m/s |
| 1 | Base local y translation velocity command | Same; final velocity actuator range ±1 m/s |
| 2 | Base yaw velocity command | Normalized command scaled by actuator half-range 1.5 rad/s |
| 3 | Torso height delta | JOINT_POSITION: clip [-1,1], multiply by 0.05 m, add to achieved torso joint position |
| 4 | Control mode | `<0.5` -> -1 -> arm goals relative to achieved pose; `>=0.5` -> +1 -> update retained desired goals |
| 5,6,7 | EEF x,y,z translation delta | OSC_POSE: clip each [-1,1], multiply by 0.05 m in base frame |
| 8,9,10 | EEF axis-angle rotation delta x,y,z | OSC_POSE: clip each [-1,1], multiply by 0.5 rad; construct delta rotation and left-multiply base-frame orientation |
| 11 | Gripper direction | `<0.5` -> -1 opening; `>=0.5` -> +1 closing; integrated gripper command described below |

The **runtime** composite configuration is HYBRID_MOBILE_BASE, with part ordering
`right`, `right_gripper`, `base`, `torso`. RoboCasa Kitchen modifies the default
robot config to select `JOINT_VELOCITY_LEGACY`. The converted 12D order is:

```text
[a5,a6,a7,a8,a9,a10, threshold(a11), a0,a1,a2,a3, threshold(a4)]
```

OSC uses fixed impedance, kp=150, damping ratio=1, kd=24.494897, uncoupled
position/orientation, delta input, base reference frame, no interpolation or
position/orientation limits. Torso kp=2000, kd=89.442719. The base has velocity
actuators (kv=1000,1000,1500 in XML) and significant joint frictionloss=250;
small commands need not produce the kinematic displacement `command * dt`.

The legacy base controller measures yaw relative to its reset orientation.
For this runtime, forward joint axis is [1,0,0], side [0,1,0], yaw [0,0,1], and
`_check_forward_joint_reversed()` is false. It applies
`[x*cos(theta)-y*sin(theta), x*sin(theta)+y*cos(theta), yaw]`.
On older models with forward axis [0,1,0], it swaps x/y and uses the other
rotation branch. The controller computes `scaled_delta = scale_action(...)`
but does not use it: the original action forms the velocity goal. Its output
then maps through actuator ranges; `WheeledRobot.control` clips each final
actuator command before assigning `ctrl`. Mode does **not** gate either the
base or arm commands; it changes the arm goal-update reference.

PandaGripper keeps a two-element accumulator, initially [0,0]:
`current_action = clip(current_action + [-1,+1] * 0.2 * sign(direction), -1,1)`.
SimpleGripController maps that to position-actuator targets [0,0.04] and
[-0.04,0] metres. Its `goal_qvel` field name does not make these physical
velocity actuators. One opening call produces accumulator [0.2,-0.2] and
targets [0.024,-0.024]; one closing call from reset produces [-0.2,0.2] and
targets [0.016,-0.016].

For an example value **-1.02742**, the outcome depends on its dimension:

- Mode/gripper: simply threshold to -1; no oversize analog magnitude survives.
- EEF translation/rotation or torso: controller input clipping gives -1,
  then scales to -0.05 m / -0.5 rad / -0.05 m respectively.
- Base: the wrapper and legacy goal path retain it; frame transform and
  actuator scaling occur, then final actuator commands are clipped. With
  zero relative yaw, x/y saturates at -1 m/s, yaw at -1.5 rad/s. With rotated
  x/y, clipping happens after rotation and is not equivalent to clipping the
  original public vector first.

This out-of-range explanation comes from pinned source; no Phase 1A action or
out-of-range simulator action was executed.

### Neutral command, timing, and five-step result

There is **no exact 12D no-motion action for an arbitrary intermediate gripper
opening**: the wrapper cannot emit zero gripper direction. The tested neutral
representation means no intended base/torso/EEF motion, stable achieved-pose
mode, and continue opening the gripper:

```text
public:    [0,0,0,0,-1, 0,0,0,0,0,0,-1]
converted: [0,0,0,0,0,0,-1, 0,0,0,0,-1]
```

All-zero public input also selects opening and achieved mode because both
threshold comparisons use 0.5. It is not a gripper hold. The five-step test is
therefore an arm/base neutral smoke test with explicit, bounded gripper opening.

Control frequency is 20 Hz; model timestep and MuJoCo `opt.timestep` are both
0.002 s. Each public step measured 25 physics substeps and 0.05 s. Reset itself
contains 250 internal settling steps in Kitchen, so MuJoCo time starts at 0.5 s;
these are upstream reset physics, not extra public neutral commands.

| Step | MuJoCo time | Gripper joint delta (mm) | Finger actuator targets (m) |
|---|---|---|---|
| 1 | 0.50 -> 0.55 | [+0.859212,-0.832880] | [+0.024,-0.024] |
| 2 | 0.55 -> 0.60 | [+2.095389,-2.073464] | [+0.028,-0.028] |
| 3 | 0.60 -> 0.65 | [+2.928567,-2.916077] | [+0.032,-0.032] |
| 4 | 0.65 -> 0.70 | [+3.398869,-3.391865] | [+0.036,-0.036] |
| 5 | 0.70 -> 0.75 | [+3.662600,-3.658672] | [+0.040,-0.040] |

Total world base delta (metres): `[4.07668e-8,1.38386e-7,8.63710e-9]`, norm
**0.000144524 mm**. Total world EEF delta: `[1.94750e-6,3.67330e-6,-1.49471e-6]`,
norm **0.004418152 mm**. Total gripper joint delta is
`[+0.012944637,-0.012872958]` m; final positions `[0.033479556,-0.033404598]` m.
Base/EEF drift is small and finite. Gripper movement is commanded opening,
not numerical drift or a claim of a stationary gripper.

Every step's before/after base and EEF position/quaternion, gripper state,
raw/converted action, reward, success, termination/truncation, and finite check
is retained in `controls.json`, including all 25 `ctrl` vectors.

### Four controlled probes

Each used one public step, 0.50 -> 0.55 s. Before every probe, reset seed 42
and NumPy seed 42; complete qpos and qvel matched the initial state at absolute
tolerance 1e-9. Only the listed public dimension differs from neutral.
Base/EEF deltas below are in world coordinates and millimetres; gripper deltas
are the two signed joint changes in millimetres.

| Probe | Changed entry | Delta base xyz (mm) | Delta EEF xyz (mm) | Delta gripper (mm) |
|---|---|---|---|---|
| EEF +x | a5=0.05 | [-0.000932,0.000126,-0.000155] | [0.280779,-0.004031,-0.024527] | [0.858435,-0.833656] |
| Base +x | a0=0.05 | [0.040462,0.000128,0.000008] | [0.041729,0.003571,-0.000766] | [0.859152,-0.832940] |
| Close gripper | a11=1 | [0.000039,0.000136,0.000008] | [0.001935,0.003630,-0.001488] | [-1.207776,1.234101] |
| Desired mode | a4=1 | [0.000019,0.000053,0.000004] | [0.006397,0.015349,-0.004349] | [0.859440,-0.832652] |

The EEF target moved +2.5 mm in base x, and achieved EEF x increased ~0.281 mm
after 50 ms. Base target was `[0.050000001,-1.12688e-8,0]`, with a small positive
world-x response; this short low-amplitude response does not measure steady-state
speed. Closing reduced finger separation, opposite to the opening reference.
Desired mode retained goal `[0.257408623,-0.014622656,0.592061660]`, whereas
achieved mode refreshed it to `[0.257287691,-0.014911911,0.592192065]`. This
distinguishes the modes without large motion. All responses agree with the
source-derived targets; no amplitude escalation or parameter sweep was run.

### One measured action-to-physics trace

EEF +x probe public float32 action:
`[0,0,0,0,-1,0.05,0,0,0,0,0,-1]`.
LeRobot slices it into base [0,0,0,0], mode [-1], translation [0.05,0,0],
rotation [0,0,0], gripper [-1]. RoboCasa thresholds mode/gripper and reorders to
float64 `[0.05,0,0,0,0,0,-1,0,0,0,0,-1]`.

OSC clips/scales translation to [0.0025,0,0] m and updates the base-frame target
to `[0.259787691,-0.014911911,0.592192065]`. The first MuJoCo `ctrl` vector was:

```text
[0.087783687, 6.565354003, 0.970686635, 17.052255625,
 0.130236475, 4.096712515, 0.008610371,
 0, 0, 0, 342.607288583, 0.024, -0.024]
```

These 13 values are seven arm motor torques, three base velocity actuator
commands, one torso motor force, and two finger position actuator commands.
They are not another 12D policy action. Arm/torso controller output includes
holding/gravity compensation even when the public motion commands are zero.
The robot writes range-clipped controls at every substep; MuJoCo integrates
dynamics/contact using those actuator settings.

After 25 substeps, physical world EEF position changed from
`[3.285282440,-2.807327053,1.292192094]` to
`[3.285563219,-2.807331084,1.292167567]` m. The normal returned 16D observation was:

```text
[3.027993917,-2.792413950,0.699999869,0,0,-0.000001846,1,
 0.257534027,-0.014915142,0.592172086,
 -0.990760088,-0.031057045,-0.130045950,0.022759303,
 0.021318741,-0.021292893]
```

Returned observations use robosuite's sampled observation cache. The diagnostic
also forces a fresh observation for physical before/after snapshots; those
values need not equal the cached returned sample. For example, returned relative
EEF x is 0.257534027 while the fresh post-step value is ~0.257569395. The trace
keeps both rather than silently substituting one for the other. Instrumentation
wraps `set_goal` and `_pre_action` only to copy their actual values, then calls
the original methods. It adds no controller commands or physics steps.

### Success and timeout semantics

`CloseFridge` inherits `ManipulateDoor._check_success`, which calls
`fxtr.is_closed(env=self)` for close behavior. `Fridge.is_closed` defaults to
the fridge compartment and door region, passing all `_fridge_door_joint_names`
to `Fixture.is_closed`. Each hinge qpos is normalized using its joint range,
with reversed sign convention accounted for; **every fridge door must have
normalized opening <=0.005**. This is 0.5% of joint travel, not 0.005 radians.
It does not test freezer doors, interior drawers, visual appearance, or a hold
duration. Different fridge fixtures can have one or multiple checked doors.

This reset has two checked joints, `fridge_right_group_fridge_left_door_joint`
and `fridge_right_group_fridge_right_door_joint`; initial normalized openings
were 0.970936094 and 0.917289495. Success remained false in every test.
The simulator reward path and RoboCasa Gym wrapper evaluate success after
stepping; the latter returns sparse reward 1/0 and `info.success`. LeRobot
sets `terminated = done or is_success` and automatically resets on termination.

Runtime low-level horizon is 1000 but `ignore_done=True`, so that timeout is
disabled. LeRobot stores the task registry horizon 600 in `_max_episode_steps`,
but this direct wrapper's step does not enforce it and RoboCasa returns
`truncated=False`. A future rollout runner must impose its own 600-step cap
(or a smaller explicitly chosen diagnostic cap). No timeout was exercised here.

## Artifacts, reproduction, and remaining boundary

Run only when the bounded diagnostics are wanted again:

```powershell
& D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe -u -B scripts/phase1b_contract.py
& D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe -u -B scripts/phase1b_contract.py --controls
```

The first inspects reset only. The second performs the five neutral steps and
four one-step probes. The successful run was not repeated for reporting.
Windows extraction of long generative texture filenames needed the `\\?\`
extended-length absolute path prefix. Initial missing-asset and import errors
were resolved before controls. Optional missing mimicgen / robosuite_models /
mink warnings do not block PandaOmron with its actual OSC controller.

Artifacts under ignored `outputs/phase1b/`:

- `reset.json`: complete reset observations/configuration/scene metadata.
- `controls.json`: all nine actions, controller targets, physical state snapshots,
  sampled observations, rewards/flags, and 225 physics control vectors.
- `reset-cameras.png`: labeled left/right/wrist reset contact sheet.
- `controller-diagnostic.mp4`: 50 frames, 768x288, 10 FPS, five seconds; ten
  captured states held for 0.5 seconds each for legibility. This is a slowed
  diagnostic montage, with reset discontinuities before probes, not real-time
  playback or a learned rollout. Video decoded successfully and frames inspected.
- `archive-hashes.json`, dependency dry-run/install reports, original package
  snapshot, downloaded sources/assets and Numba cache.
- `failed-reset-missing-texture.txt`: retained earlier failed attempt; the later
  successful reset and controls supersede it. The diagnostic's periodic
  faulthandler stack dump during scene rebuilding was informational, not a
  simulation timeout or failure.
- `final.diff`: review diff, generated after documentation is complete.

Compact measured results are also kept in `phase1b-results.json`.

## Proposed Phase 1C — requires separate approval

Run one learned SmolVLA CloseFridge rollout, seed 42, using Phase 1A's exact
checkpoint/backbone revisions and saved processors, with the Phase 1B runtime.
Confirm model loading after the necessary NumPy change; do not rerun Phase 1A's
loss experiment. Map HWC uint8 cameras to the saved CHW float32 camera contract,
preserve state order and instruction, and log action normalization boundaries.
Use a hard 600-step maximum, success/termination stop, finite-value checks and
a recorded action-bound strategy. Preserve the native wrapper/controller
handling documented here unless a deliberate clipping change is approved;
record out-of-range public values and saturation rather than assuming Box clips.
Use native chunk_size/n_action_steps=50 and 10 flow integration steps, record
replanning times and executed actions, reset policy queues once at episode start,
and save video plus per-step metrics. One rollout is a diagnostic result, not
a success-rate benchmark. No training/backward/optimizer is part of Phase 1C.

## Pinned source references

- [LeRobot 0.6.1 wrapper](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/envs/robocasa.py)
- [RoboCasa conversion and Gym step](https://github.com/robocasa/robocasa/blob/56e355ccc64389dfc1b8a61a33b9127b975ba681/robocasa/wrappers/gym_wrapper.py)
- [Kitchen controller selection and reset settling](https://github.com/robocasa/robocasa/blob/56e355ccc64389dfc1b8a61a33b9127b975ba681/robocasa/environments/kitchen/kitchen.py)
- [HybridMobileBase mode handling](https://github.com/ARISE-Initiative/robosuite/blob/aaa8b9b214ce8e77e82926d677b4d61d55e577ab/robosuite/controllers/composite/composite_controller.py)
- [Legacy base controller](https://github.com/ARISE-Initiative/robosuite/blob/aaa8b9b214ce8e77e82926d677b4d61d55e577ab/robosuite/controllers/parts/mobile_base/joint_vel.py)
- [OSC controller](https://github.com/ARISE-Initiative/robosuite/blob/aaa8b9b214ce8e77e82926d677b4d61d55e577ab/robosuite/controllers/parts/arm/osc.py)
- [Panda gripper accumulator](https://github.com/ARISE-Initiative/robosuite/blob/aaa8b9b214ce8e77e82926d677b4d61d55e577ab/robosuite/models/grippers/panda_gripper.py)
- [CloseFridge and inherited success](https://github.com/robocasa/robocasa/blob/56e355ccc64389dfc1b8a61a33b9127b975ba681/robocasa/environments/kitchen/atomic/kitchen_doors.py)
- [Fridge door selection](https://github.com/robocasa/robocasa/blob/56e355ccc64389dfc1b8a61a33b9127b975ba681/robocasa/models/fixtures/fridge.py)
- [Normalized joint closure threshold](https://github.com/robocasa/robocasa/blob/56e355ccc64389dfc1b8a61a33b9127b975ba681/robocasa/models/fixtures/fixture.py)
