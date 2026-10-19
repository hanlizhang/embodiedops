# Seed 43: non-target interaction audit

Offline analysis only, 2026-10-19. No policy inference, physics, training, evaluation or intervention was run. Historical files were not changed.

**Conclusion:** A shows an interior-directed approach, actual finger closure near the shrimp image, and withdrawal without useful door closing. This is consistent with a non-target grasp-like sequence, but neither shrimp contact nor a successful grasp is established. B and C show different behavior: open-gripper door manipulation with limited progress. C directly establishes inefficient hand/arm-to-door contact. These are not one universal failure mechanism, and behavior does not reveal model intention.

## Sources and comparison

All paths below are repository-relative. A: `outputs/phase1d/evaluation/pretrained-seed43`; B: `outputs/probes/instruction-distribution-v1/approved-retry-1/task-id-seed43`; C: `outputs/control-horizon-v1/h20-seed43`. C is identified by `command-seed43.json` and its own manifest (seed 43, execution horizon 20, instruction CloseFridge).

Each manifest identifies `lerobot/smolvla_robocasa`, revision `f7af588d11e30b0069f200792fb285904b74c80b`. B/C record weight SHA256 `96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b`; A does not record a weight digest, so its checkpoint identity is revision-level evidence, not independently verified byte identity. The config's backbone/pretrained-path field is not the evaluated fine-tuned checkpoint identifier.

The synchronized comparison is `outputs/seed43-interaction-v1/three-run-comparison.mp4`: A/B/C rows, left/right/wrist columns, original 20 Hz, reset through action 600 (30 s). Each frame is aligned by executed action count, not wall-clock inference time or behavioral phase. No synthesized/replayed views. Key original frames include 350 (17.5 s), 440 (22 s), 450 (22.5 s), 508 (25.4 s), and 600 (30 s). JPEG stills at these times accompany the video.

| Episode | Instruction / execution | Gripper action 11, all 600 actions | Measured finger response | Door angle, degrees | Physical contact evidence |
|---|---|---|---|---|---|
| A | `Close the fridge door.` / H50 | Range −1.03343…1.03405; 72 close commands; 4 open→close and 4 close→open transitions. Close steps 429, 440–508, 520, 542 | During 440–508, joint separation proxy falls 78.883→1.000 mm (minimum 0.997 mm) | Initial 85.52388; best 85.52230 at 62; final 89.98809. Closure interval 440–508 closes only 0.00185° | No saved contact history; shrimp contact unresolved |
| B | `CloseFridge` / H50 | Range −1.02712…−0.95351; 0 close, 0 transitions | Reset separation 41.067 mm; opens to about 80 mm; final 79.888 mm | Same initial; best 80.85591 at 596; final 81.68723 (net closing 3.83665°) | No saved contact history; cannot reconstruct it |
| C | `CloseFridge` / H20 | Range −1.02621…−0.97078; 0 close, 0 transitions | Reset 41.067 mm; opens to about 80 mm; final 79.222 mm | Same initial; best 77.81422 at 486; final 83.35423 (net closing 2.16964°); reopens 5.54001° after best | Loaded robot contact in 345 actions, first 77, last 599; hand and links 5/6/7 contact moving fridge door; no recorded finger or shrimp contacts |

All three fail actual task success. Close is action[11] ≥0.5, as the archived runner's gate specifies. Transition counts exclude reset, where there is no policy command. Joint separation is qpos[0]−qpos[1], **not** a calibrated fingertip aperture. Single-step close pulses need not close the fingers: at 520 separation actually rises while the fingers are still responding to earlier opening.

## Motion, geometry and useful phases

EEF positions below are recorded simulator **world-frame** meters, orientations are saved xyzw quaternions. Full per-step positions/quaternions, both measured finger joints and every gripper command are in `A-all-actions.csv`, `B-all-actions.csv`, `C-all-actions.csv`. `measurements.json` also supplies segment path lengths and sign-invariant net orientation changes.

All start at EEF (0.8933, −1.1706, 1.3330). At step 76, A is (1.2184, −1.1681, 1.2045), B (1.3305, −1.4328, 1.3014), C (1.2826, −1.4626, 1.2827). Net orientation changes from reset are 21.51°, 52.14°, 35.98° respectively. This establishes early trajectory divergence, without assigning these world axes to door-normal directions.

A advances from (1.1418, −0.6523, 1.0753) at 350 to (1.1505, −0.4730, 0.9806) at 439; the shrimp enlarges near the center of the wrist image. It then closes and withdraws to (1.0575, −0.9044, 1.1770) at 508. At 508 the shrimp remains visually on the shelf, separated from the withdrawn hand. This supports a grasp-like approach/closure/withdrawal, not a demonstrated lift. During 451–500, EEF path length is 0.473 m and net orientation change 28.50°, while door closing is just 0.00134°. A opens the door by 4.4776° over 1–150; subsequent tiny changes do not undo it.

B's useful net closing occurs over 151–250 (5.5516°), 251–350 (1.1546°), and 351–450 (2.4437°), after opening it 4.4894° over 1–76. C opens 4.4863° over 77–150 despite recorded loaded contact, makes essentially no closing progress over 151–350, then closes 9.4063° over 351–450. C later loses progress. These phase deltas are relative to each phase's start, not all relative to reset.

C's geometry catalog identifies shrimp geoms 1680–1685 as `door_obj_*`, body `door_obj_main`; scene metadata identifies category shrimp / Shrimp007. None appears in any recorded robot-contact pair, including unloaded pairs. Loaded contacts require active constraint address and positive normal force; the 25 samples per action resolve physical contacts beyond RGB proximity. Earliest pair is `gripper0_right_hand_collision` against door geom g47 at 77. Peak recorded **single contact** normal force is 324.24 N. Link7 contacts begin 91; links5/6 appear 432/436. No recorded loaded finger contacts. The catalog associates the contacted fridge geometries with the moving door body.

There are no saved per-frame door/object transforms, calibrated contact-surface distances or contact-frame rotations sufficient here to compute EEF-to-shrimp clearance, door-local motion, effective world force or hinge torque. RGB apparent proximity does not fill that gap. No privileged state was provided to a policy in this audit.

For C's loaded actions, mean requested translation-increment norm is 20.798 mm versus achieved base-relative EEF displacement 2.192 mm per action; outside loaded actions these are 17.747 and 4.372 mm. Commands are clipped/scaled OSC increments, **not promises of one-step displacement**. This disparity plus physical contact supports constrained motion; it does not isolate a controller defect. Full per-step controller targets and saturation histories are absent. A's measured finger closure confirms that its sustained gripper command was substantially executed.

## Camera check and confidence

All 601 frames of each seed43 video were decoded: **zero all-black frames in each of all three camera tiles**, excluding the 32-pixel title band. The archived runner records the same observation RGB used by `prepare` on the next action; this supports nonblack source observations. Full postprocessed policy tensors at every replan were not archived, so their exact contents cannot independently be certified. Nonblack also does not prove every pixel or camera calibration is correct.

Brief seed42-H20 exception: video frames 45–285 inclusive have all three camera tiles exactly zero (241 frames); all three raw final camera PNGs also have maximum pixel value zero. This is not merely a video encoder blackout. The archived runner passes returned observations to both video recording and the next `prepare`; queued actions continue, and the first new chunk using a black source observation is action 61. This is a source-code-supported inference about policy inputs, not a saved network-tensor measurement. **The affected seed42 H50/H20 comparison is invalid as a clean horizon-only comparison.** Preserve its physical outcome but exclude it from causal horizon conclusions. No further infrastructure investigation was performed.

Confidence: high for commands, finger motion, angle changes and C's recorded contacts; moderate for A's visually shrimp-directed, grasp-like behavior; unresolved for A/B shrimp contact, intention, object-local clearance and controller tracking fault. C's contact history cannot be retroactively assigned to A or B. The three traces are descriptive comparisons, not a causal identification of instruction versus horizon effects.

## One proposed discriminating pilot — not implemented or executed

Test **visual distractor dependence** of A with a render-only shrimp visibility ablation. Keep the shrimp's collision/dynamics geometry and all scene state unchanged; hide only its visual appearance in all three policy cameras from reset. Label this explicitly as an **oracle diagnostic visual intervention**, not a deployable repair or policy input of privileged state.

Bound: one freshly matched A baseline and one ablation, seed43 only, natural-language instruction, H50/chunk50/10 flow steps, same pinned weights/processors/controller, maximum 600 actions each, identical initial state and inference RNG schedule. Before running, verify hiding the mesh does not change collision configuration, reset state, camera health or rendering of other objects. Save actual replan inputs, object poses, robot–object/door contacts and controller targets in both arms. Measure close-command intervals, finger positions, object-relative approach/withdrawal, door progress/success, contacts and compute cost.

Prediction: if the shrimp's visible appearance drives A's non-target sequence, hiding it should change the shelf approach and sustained closure, ideally increasing useful door motion. Unchanged behavior weakens that explanation; disappearance of closure without better door progress shows distractor dependence but not repair. One pair cannot establish robustness or distinguish semantic attraction from other visual sensitivity. Do not advance to training or claim a control improvement from this test alone.

## Reproducibility and scope

Run once into the exclusive output directory: `D:/Apps/miniconda/envs/embodiedops-robocasa/python.exe scripts/seed43_interaction_audit.py`. It refuses to overwrite an existing output directory. `source-sha256.json` records input hashes. The script imports only offline data/image libraries. Validation: three 600-action traces, 601 synchronized frames, actual gripper gate agreement, interval edge-case check, complete C contact sample counts, camera pixel scan. No historical H50 contacts were synthesized. New source: `scripts/seed43_interaction_audit.py`; new report: this file. Existing unrelated uncommitted localization work was preserved. No commits.
