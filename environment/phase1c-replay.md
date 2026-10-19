# Phase 1C recorded-state viewer — 2026-10-18

Implemented in `scripts/phase1c_replay.py`. The completed trajectory is replayed
by loading recorded state into display-only MuJoCo data and calling `mj_forward`.
The policy runner is unchanged. No SmolVLA inference, learned action execution,
training, dependency change, or commit occurred.

## Run manually

In an Anaconda / Miniconda PowerShell prompt:

```powershell
conda activate embodiedops-robocasa
cd C:\Users\Alan\Projects\embodiedops
python scripts\phase1c_replay.py
```

The validated model already exists. This command loads it directly; it does not
recreate an environment or run an episode. Keep the PowerShell window available
for commands and state readouts. The viewer starts paused at recorded reset in a
free external camera view. Collision meshes and GUI target markers are hidden
to make the visual surfaces readable.

- Mouse drag rotates, right-drag pans, scroll wheel zooms, including while paused.
- Enter in PowerShell or Enter/space in the viewer: play to the next 50-action
  boundary. Press again during playback to pause early.
- Type any action endpoint `0` through `600` in PowerShell and press Enter to
  jump there and pause. Step 0 loads the recorded reset; it does not reset a simulator.
- Viewer keys `1`–`7` jump to `250,272,285,300,335,406,462`, respectively.
- Type `q` in PowerShell or close the viewer window to exit.

After a jump to 272, Enter advances through 300 and pauses. It preserves the
original action-chunk boundaries. State is loaded at 20 Hz, without interpolation
or dynamic integration. If rendering/printing is slow, playback can take longer
in wall time; it never skips recorded endpoints to catch up.

Every newly displayed endpoint prints action index, chunk, simulator time,
recorded base/EEF position and quaternion, gripper qpos and door angles. Paused
endpoints additionally print geometric distances and the closest named collision
geometry pairs. A background input thread keeps the viewer responsive while paused.

## Validation results

The accepted bundle is under:

`outputs/phase1c/retry-20261018/replay-ready/`

| Check | Result |
|---|---|
| RoboCasa | 1.0.0, source `56e355ccc64389dfc1b8a61a33b9127b975ba681`; 494 Python files matched archive |
| robosuite | 1.5.2, source `aaa8b9b214ce8e77e82926d677b4d61d55e577ab`; 204 Python files matched archive |
| MuJoCo / NumPy / LeRobot | 3.3.1 / 2.2.5 / 0.6.1 |
| Episode metadata | Exact Phase 1B match, including layout 6/style 22, fixtures, object configuration, cameras and initial robot pose |
| Robot / fixture / controller | Saved classes and controller configuration matched |
| Initial qpos / qvel | Exact numeric and SHA256 matches |
| Left / wrist camera | Pixel-identical to original reset |
| Right camera | One color-channel value differs by one uint8 level; mean absolute difference 0.000005086 |
| Named elements | Saved actuator order, fridge joints, base joint axes, camera parent bodies and local transforms matched |
| Model dimensions | nq=112, nv=108, nu=13, na=0; 281 bodies, 88 joints, 1598 geoms, 368 sites, 6 cameras |
| All 601 recorded states | Correct dimensions, finite values, contiguous indices and 0.05 s spacing |
| Recomputed door angles | Exactly match recorded joint angles at all 601 states |
| Recomputed base / EEF position | Maximum differences from recorded observations: 0.556 / 0.531 mm |
| Compiled-model reload | Full saved name inventory and structural arrays match after binary serialization |

The initial state hashes are:

- qpos: `c97454489d119ba49c96acda5f06b15f2c74f1e8e31107ae82b59d23edcb89be`
- qvel: `5721022093504d6ee5a8b4cb0562ea38ca3745ab2facbc7aae1fb62fe72844ee`

Camera acceptance permits at most a one-level per-channel difference and mean
absolute difference ≤0.001, to accommodate sparse rasterization rounding.
Metadata, geometry and state checks are independent of this image tolerance.
Fresh `mj_forward` positions are not bit-identical to the original sampled/cached
observations; the submillimeter discrepancies are compatible with the final
2-ms integration update. Recorded qpos/qvel are loaded unchanged.

The original run did not save a full joint/body/camera inventory or compiled
model. Therefore a complete original-versus-reconstructed name/structure byte
comparison is impossible. Validation instead combines the surviving named
metadata, exact reset vectors, near-identical images, source pins and all-state
kinematic checks. The full reconstructed inventory is now saved, together with
the compiled model, hashes and geometric target definitions. Default replay
rejects changes to that bundle or the original evidence files.

There are two unrecorded mocap bodies, `left_eef_target` and `right_eef_target`.
Pinned robosuite creates these solely as GUI markers. Validation checks that
they have no children, collisions or equality constraints. Replay leaves them
at their source-defined positions and hides them. There is no actuator activation
state (`na=0`). This does not make the recording a full physics restart.

Development validation first rejected these GUI markers with an overly strict
guard, then rejected a four-channel/one-level rendering difference under an
exact-pixel check. Those failed reports remain in `replay/` and
`replay-validated/`. Scene construction was retried to correct those validator
assumptions; no learned actions or recorded actions were executed. The only
integrated motion was RoboCasa's upstream scene-initialization/reset settling.

## Geometric diagnostics

`mj_geomDistance` measures signed clearance between the recorded-pose gripper
collision geoms and each door's collision geometry, including separately named
handle geoms. Results are in metres, capped at 2 m; negative values indicate
geometric overlap. These are collision-model distances, not distances between
rendered mesh pixels and not recovered historical contact forces. EEF-to-door
body-origin distance is separately labeled and is not surface clearance.

| Recorded step | Gripper → right door clearance | Gripper → right handle clearance |
|---|---:|---:|
| 250 | 0.049 mm | 186.5 mm |
| 272 | 0.058 mm | 194.8 mm |
| 285 | 7.065 mm | 202.9 mm |
| 300 | 13.126 mm | 213.9 mm |
| 335 | 11.098 mm | 224.8 mm |
| 406 | 37.287 mm | 350.8 mm |
| 462 | 106.266 mm | 447.6 mm |

These measurements support investigating near-contact with the door surface in
the main door-motion interval, rather than a handle grasp. They do not establish
force direction or sustained contact. At step 462, finger narrowing occurs far
from the door and handle. An additional geometric check finds about 0.175 mm of
overlap between `gripper0_right_finger1_collision` and the other fridge collision
geom `fridge_right_group_g6`. This supports inspecting that static fridge geometry
as a possible constraint; it does not reconstruct the original contact force.
The additional measurements are in `interior-proximity.json`.

The left-door and handle clearances are also printed at paused endpoints and
saved in `geometric-diagnostics.json`. No claim about the cause of the policy's
failure, or a need for fine-tuning, follows from geometric proximity alone.

## Testing, artifacts and limits

Headless checks loaded the saved compiled model, rendered external previews at
steps 0, 285 and 462, and exercised the viewer control loop with a test viewer:
start paused; play endpoints 1–50; remain paused at 50; jump to 272; advance to
300; jump to 462. Loaded qpos/qvel matched exactly. `mj_step`, `mj_step1` and
`mj_step2` were replaced by rejecting test stubs and were never called.
The native interactive window was not launched by the assistant; mouse/window
interaction awaits the user's manual run. No policy is imported by default replay.

Useful files in `replay-ready/`:

- `validation.json`, `model-inventory.json`, `geometry-spec.json`.
- `validated-scene.mjb`: cached compiled scene; keep it with the validation files.
- `geometric-diagnostics.json`, `interior-proximity.json`.
- `replay-control-checks.json`, `external-view-0.png`, `external-view-285.png`,
  `external-view-462.png`.
- `replay.diff`: diff for the new replay script and this report only.

`--validate` is a development-only scene reconstruction path, not needed for
manual playback of the delivered bundle. It refuses to overwrite an existing
validation attempt. Any future reconstruction failure stops before opening the
viewer. Default playback requires an accepted bundle and never falls back to
creating a new environment or running the policy.

This is sampled recorded-state visualization, not a full physics restart.
Intermediate physics states, controller memory and original contact forces
are unavailable. Phase 1A/1B evidence, `phase1c_rollout.py` and
`RESUME_EVIDENCE.md` remain unchanged; no commit was made.
