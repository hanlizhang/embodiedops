# CloseFridge execution-horizon experiment — 2026-10-19

Pretrained task-ID success was **1/3 at H=50 and 1/3 at H=20**. H=20 did not
rescue either failing seed. Seed 42 stopped under the unchanged displacement
guard; it was not rerun. Exactly three new policy episodes, seeds 42–44, ran.
No training, H=10, additional seeds, dependency changes, or weight changes.

## Outcomes

| Seed | H | Success | Actions | Final door degrees | Stop |
|---|---:|---|---:|---|---|
| 42 | 50 | No | 600 | left 87.867 / right ~0 | Horizon |
| 42 | 20 | No | 285 | left 86.708 / right 82.495 | EEF displacement guard |
| 43 | 50 | No | 600 | 81.687 | Horizon |
| 43 | 20 | No | 600 | 83.354 | Horizon |
| 44 | 50 | Yes | 342 | -0.084 | Actual task success |
| 44 | 20 | Yes | 337 | 0.220 | Actual task success |

Success is the existing RoboCasa predicate, not an angle threshold introduced by
this analysis. Seed 42 is a censored trajectory: its H=20 final pose cannot be
compared as a 600-step endpoint. All new states were finite; MuJoCo warning
counters were zero. At step 285 its EEF moved **0.052210 m** in one 0.05 s action,
exceeding the inherited 0.05 m guard. This is a guard-stopped failure, not proof
of numerical divergence. The launcher initially halted on that exit code; only
the still-unexecuted seeds 43 and 44 were subsequently launched. No retry.

## Execution contract and interpretation

Installed LeRobot 0.6.1 `SmolVLAPolicy.reset()` makes an action deque with capacity
`config.n_action_steps`; `select_action()` extends it with only the first H rows
of `_get_action_chunk()` and pops one action per call. Empty queue triggers a
fresh prediction. Preflight exercised the installed methods using identifiable
synthetic action indices and distinct observation batches, at H=20 and H=50.

The runner verifies the original saved policy/processor configuration against
each reference, then sets **only the in-memory `cfg.n_action_steps=20`**, before
`policy.reset()`. All predictions remain `[1,50,12]` (internal `[1,50,32]`), with
10 flow steps. No checkpoint is saved or modified. Model SHA256 remains
`96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b`.
Revision is `f7af588d11e30b0069f200792fb285904b74c80b`.

At 20 Hz, H=50 uses one prediction for 2.5 simulated seconds; H=20 uses it for
1 second and discards 30 actions. It reduces how long the policy acts without
fresh visual feedback, but can interrupt a predicted manipulation sequence and
introduce a command discontinuity at each replan. It does not improve perception,
contact geometry, grasp planning, or controller stability by itself. Low-level
controller feedback still runs in both conditions. Simulation pauses during
inference: these runs do not test real-time latency compensation.

Policy inputs remain three RGB cameras, the original 16D robot state, and
`CloseFridge`. Saved normalization, camera mapping, controller, checkpoint,
precision, 600-action cap and sampling seeds are retained. Door states, contact
forces, full qpos/qvel and geometry catalog are diagnostic outputs only.

## Historical reuse and reproducibility limits

H=50 references: `outputs/phase1d/language-control-native` for seed 42, and
`outputs/probes/instruction-distribution-v1/approved-retry-1/task-id-seed43`
and `task-id-seed44`. Model/revision, processor hashes, versions, saved policy,
controller and scene contracts match. Reset qpos/qvel and 16D observations match
exactly. Camera matching retains the historical sparse one-level rasterization
tolerance (max 1/255, mean <=0.001 in uint8 units); this is not pixel-exactness.
Raw camera differences are saved in each paired-reset-validation file.

Despite matching contracts, generated prefixes are **not bitwise identical**:

| Seed | Max first-20 action difference | Max first-20 qpos difference |
|---|---:|---:|
| 42 | 0.0020964 | 0.0007204 |
| 43 | 0.0033296 | 0.0006772 |
| 44 | 0.0166971 | 0.0007403 |

These differences precede the first H=20 replan. Their source is not isolated;
historical sampler fingerprints are absent and strict GPU determinism was not
established. Do not attribute every trajectory difference solely to H. Seeds are
set once per episode; after step 20 the arms also use different numbers of noise
draws at the same simulation time. These are three known design scenes, not a
generalization benchmark or a controlled bitwise counterfactual.

## Physical contacts and trajectories

`control_diagnostics.py` reads `data.contact` and `mj_contactForce` immediately
after every actual physics step (25 samples/action, 0.002 s each). It records
geom IDs, contact position/distance, solver constraint address, local force in
N and torque in Nm. Geometry names and body ancestry map IDs to robot/fixture
parts. Logs retain robot-involving contacts and total ncon, including contacts
with zero force. A loaded contact requires a solver constraint and positive
normal force; geometric proximity alone is never counted as loaded contact.
Forces are simulator constraint estimates, not calibrated hardware measurements.
Contact frame orientation was not recorded; world-frame force/hinge torque cannot
be reconstructed from these logs. Pair impulses sum normal force * 0.002 s;
multiple collision geoms can contribute simultaneously.

- **42:** no loaded robot–fridge contacts before steps 281–282. Arm links 3/4
  contacted fridge body/left door, with peak single-contact normal force
  **2270.16 N** (link4/body), 768.01 N (link4/door), 1086.40 N (link3/body).
  There was no recorded gripper contact. Left door closed only 0.676 degrees;
  right door changed 0.061 degrees. Collision precedes the displacement guard,
  but does not alone establish its cause. The guard is at step 285, not itself
  a replan boundary; the relevant chunk starts at 281.
- **43:** **345/600 actions** contain loaded robot–fridge contacts, beginning at
  77. Hand collision against a door geom peaks at **324.24 N**; links 5/6/7 also
  contact door geometry. Body ancestry verifies the opaque g47/g33/g58/etc.
  geoms belong to the moving door. There are no recorded finger contacts.
  Door starts at 85.524 degrees, reaches 77.814, and ends at 83.354. Contact is
  present but fails to produce sustained closing progress. Force direction,
  hinge leverage, slip and obstruction are hypotheses, not established causes.
- **44:** **120/337 actions** contain loaded contacts, starting at 210. Hand
  contacts the handle (peak **78.57 N**), followed by finger/handle and door
  contacts; the door closes from 83.317 to 0.220 degrees. Gripper commands remain
  open: finger contact does not establish a commanded grasp or force closure.

H=50 has no contact log. Its forces, contact duration and collision rate are
**unavailable**, not zero, and cannot be inferred from EEF proximity. No extra
H=50 episode was launched to fill this gap.

| Seed/H | EEF world path m | Base path m | Base net m |
|---|---:|---:|---:|
| 42/50 | 3.1575 | 0.01073 | 0.00398 |
| 42/20 (285 steps) | 2.0450 | 0.01798 | 0.01481 |
| 43/50 | 2.0286 | 0.01004 | 0.00384 |
| 43/20 | 1.8313 | 0.03024 | 0.01991 |
| 44/50 | 1.4186 | 0.00527 | 0.00311 |
| 44/20 | 1.4603 | 0.00538 | 0.00306 |

Full position/quaternion trajectories are retained in steps.jsonl. Path length
does not measure useful manipulation progress. Endpoint counts differ for 42/44.

## Continuity and computation

Mean adjacent-command L2 over continuous dimensions 0–3 and 5–10, excluding
binary mode/gripper (mixed units, not physical jerk):

| Seed/H | Replan boundary | Within chunk | Inferences | Inference total s | Warm call mean s | Rollout wall s |
|---|---:|---:|---:|---:|---:|---:|
| 42/50 | .496 | .145 | 12 | 6.922 | .509 | 49.724 |
| 42/20 | .499 | .189 | 15 | 8.483 | .513 | 44.401 |
| 43/50 | .467 | .132 | 12 | 8.530 | .621 | 65.588 |
| 43/20 | .505 | .160 | 30 | 12.428 | .380 | 56.730 |
| 44/50 | .424 | .102 | 7 | 5.026 | .592 | 32.625 |
| 44/20 | .286 | .113 | 17 | 6.597 | .333 | 28.428 |

H=20 reduces boundary jumps on 44, not consistently across seeds. Boundary
changes remain larger than within-chunk changes in all six episodes. H20 mode
switches: zero; gripper switches: two on 42, zero on 43/44. At 600 actions,
H20 requires 30 inference calls versus 12 (2.5x), each still predicting 50.
H20 simulation-step wall time including physical diagnostics/rendering was
32.416 / 36.533 / 17.448 s. Simulated durations: 14.25 / 30 / 16.85 s.
Torch peaks for each new episode: 971,354,112 allocated / 1,008,730,112 reserved
bytes. These exclude other GPU allocations. Inference timing includes the
existing synchronized sample-copy instrumentation. Wall times exclude model
load/reset and final video flush; timing differences across historical runs and
unequal diagnostics prevent interpreting the lower wall time as an H20 speedup.

## Next improvement

Keep H50 as the current baseline; H20 is not established as better. Prioritize a
**contact-progress and controller-tracking diagnostic**: log contact frame
orientation, hinge axis/pivot and achieved-versus-commanded EEF displacement to
measure useful closing torque versus arm/body loading. Use simulator quantities
only as evaluation diagnostics. Then separately test an execution-layer
workspace/joint-limit and command-rate safeguard, or chunk-boundary blending,
with fixed weights and matched noise. The immediate evidence supports detecting
ineffective/high-load contact and preventing arm collision; simply replanning
more often did not solve these mechanisms. Those changes are proposed, not run.
The 2,500-update natural/mixed training proposal remains deferred.

## Tests, artifacts and commit boundaries

- Native queue and force preflight passed; exact plan/source hashes saved.
- Two unittest tests passed: instrumentation leaves qpos/qvel bitwise unchanged
  in a loaded test scene, and restores simulator methods on exceptions. Windows
  sandbox temporary writes initially failed; the authorized outside-sandbox
  rerun passed. No dependencies were added. Optional plotting was omitted because
  matplotlib is absent; the numerical analysis completed successfully.
- All new videos fully decode: 286, 601, 338 frames (reset + actions).
- SHA256 preservation check passed for **1,206 pre-existing files**, covering
  historical Phase1D/probe/Phase2 artifacts and existing environment reports.
- Raw artifacts: `outputs/control-horizon-v1/`; structured results:
  `environment/control-horizon-v1-results.json`. Includes failures, all pair
  contacts, trajectory statistics and prefix discrepancies.
- New code: `scripts/control_diagnostics.py`, `scripts/control_horizon_experiment.py`,
  `scripts/analyze_control_horizon.py`, `tests/test_control_diagnostics.py`;
  additive optional instrumentation in `scripts/phase1d_rollout.py`.

Recommended focused commits, subject to approval:
1. Phase1E instruction contrast: pre-existing rollout hunks and
   `instruction_distribution_probe.py`; promote the preserved successful
   `approved-retry-1/report.md` and results into tracked environment files.
2. Phase2 sanity training: existing trainer/audit, four configs, and phase2.md;
   keep the A/B proposal explicitly unexecuted.
3. This horizon experiment: new optional runner hunks, diagnostics, fixed matrix,
   analysis, tests, report/results and dated context/evidence addenda.

`outputs/control-horizon-v1/preexisting-tracked.diff` and
`phase1d-rollout-before.py.txt` preserve the initial rollout changes separately.
`new-work.diff` isolates this turn; `full-working-tree.diff` includes pre-existing
uncommitted work. No commit or staging was performed.
