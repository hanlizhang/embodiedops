# Sustained learning repair — training complete, robot verification blocked

Historical stopped-run report. The camera correction and completed diagnostic
evaluation are recorded in [the 2026-10-20 addendum](learning-repair-v1-completion.md).
Original measurements and the stopped-run audit below are preserved.

Initial deterministic milestone reviewed at `628f820` (implementation `bfc1978`).
Its 512-update candidate was not evaluated in robotics. This experiment executed
10,000 updates and 10 valid robot episodes. A raw-camera fault invalidated episode
11 and stopped the registered matrix. Both candidates are rejected for promotion;
the full learning-repair milestone remains incomplete. Results below are measured,
not an improvement claim. Execution crossed 2026-10-19/20 local time.

## Preflight and fixed design

Pinned original SmolVLA `f7af588d11e30b0069f200792fb285904b74c80b`, dataset
`b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4`, and the existing RoboCasa environment.
No dependency, architecture, precision, camera, controller or normalization change.
Native masked flow-matching objective; frozen VLM, trainable action expert and state
projection. All training and new evaluation use task ID `CloseFridge`, isolating
sampling rather than instruction format. NL validation is secondary, not selection.

Two independently pretrained-initialized arms: episode-balanced uniform anchors,
and a 50% uniform / 50% uniformly chosen nonempty motion-regime mixture within the
same episode. Each shuffled episode cycle visits all 461 training episodes before
reuse; episode order and per-update noise seeds match between arms. Frame RNGs and
the mixture selector are separate from episode ordering. The independent fixed
training probe and 104 validation anchors (two per each of 52 disjoint episodes)
are identical across arms. Local validation is not proven disjoint from pretraining.

The dataset's 513 episodes all contain a positive recorded reward. Its available
155,443 action/state frames support a five-frame forward-looking motion proxy:

- quiet: base, EEF translation and rotation command norms each below .05, with
  no gripper-command transition in the window;
- commanded low motion: mean clipped/scaled translation command norm >=5 mm per
  action and mean next-state base-relative EEF displacement <2 mm;
- other motion: remaining anchors, including the final frame without a next state.

Preflight counts across all demonstrations: 1,072 quiet anchors in 179 episodes,
8,249 commanded-low-motion anchors in 495 episodes, and 146,122 other-motion
anchors in all 513 episodes. Thresholds are fixed before training; per-arm sampling
uses only training episodes. These labels are **not contacts, grasp stages, door
progress or verified controller errors**. Action-to-next-state alignment and finite
controller response make them noisy proxies. The hypothesis is that balancing rare
motion regimes while retaining uniform coverage may improve manipulation execution.
Both methods retain broad episode coverage; the comparison tests that hypothesis.

Each arm: 5,000 updates, batch1, seed142, native AdamW (betas .9/.95, epsilon1e-8,
weight decay1e-10, gradient clip10), peak LR5e-5, 250-step warmup, cosine decay to
5e-6. This lower LR than the earlier short run is shared by both arms; the experiment
does not isolate its effect against that historical run. Probe every500 updates;
save at 1000/2000/3000/4000/5000. Select lowest fixed held-out task-ID loss among
those five, earliest tie. Reload every saved checkpoint in a separate process and
verify all model/optimizer/scheduler tensors, saved processors and fixed validation.
Do not select by simulation outcomes. Native BF16 optimizer rounding remains.

| Split | Seeds | Models | Episodes |
|---|---|---|---:|
| Diagnostic | 43,44,45,46 | pretrained, uniform selected, motion-mixture selected | 12 |
| Final | 120,121,122,123 | same three frozen checkpoints | 12 |

No prior saved manifest for final seeds was found. All 24 episodes are new, H50 /
prediction chunk50 / 10 flow steps / 600-action cap, with matched per-replan RNG,
physical reset and camera checks. Both candidates undergo this **research-only**
matrix even if validation worsens, after technical integrity checks. Final scenes
cannot tune sampling or checkpoint selection. Physical motion guards stop individual
episodes, which remain reported failures; invalid cameras/contracts/runtime faults
stop the workflow without replacement scenes. No privileged diagnostics enter VLA
inputs. Scene types, actual task success, door angles, contacts and timings are saved.

Measured results go to the existing `agent_workflow.acceptance` gate, separately
for diagnostic and final scenes. Validation regression independently prevents
promotion. Thus a technically complete evaluation can still reject both candidates.

## Resources and execution

Preflight: 12 lightweight tests passed; 33.2 GB disk free; 6 GB RTX3060. Prior
training used ~1.73 GiB CUDA allocation. Two arms run serially; ten full checkpoints
are expected to need ~13.2 GB, plus diagnostics/caches. Hard bounds: 10,000 updates,
24 episodes, six hours total; 90 minutes per training arm, 300 seconds per reload,
420 seconds per episode. Nonfinite losses/gradients or failures preserve artifacts
and stop; there is no automatic sweep, seed replacement or training restart.

```powershell
python -B -m unittest discover -s tests
python -B -u scripts/learning_repair_experiment.py --output outputs/agent-workflow-v1/learning-5000-v1
```

The output root is exclusive. Short generated dataset caches are under
`outputs/lrc-v1-1` and `outputs/lrc-v1-2` to avoid the previously verified Windows
lock-path limit. Existing weights and results are preserved. Source snapshots,
registered settings, actual commands and event trace are written before workers.
GPU memory and GPU-worker wall-hours are recorded; active GPU-hours and energy are
unavailable without valid utilization/power measurements. No new results are committed.

## Results

Source root: `outputs/agent-workflow-v1/learning-5000-v1`. The original registered
plan and execution trace are preserved there. `learning-repair-v1-results.json`
contains the audited concise numerical record; its `experiment_complete=false`
is distinct from `passed=true` for the artifact audit.

### Training and checkpoint selection

| Measurement | Episode-balanced uniform | Motion mixture |
|---|---:|---:|
| Actual optimizer updates | 5,000 | 5,000 |
| Training episodes / distinct anchors | 461 / 4,891 | 461 / 4,735 |
| Selected quiet / commanded-low-motion / other anchors | 31 / 252 / 4,717 | 328 / 1,220 / 3,452 |
| Fixed training task-ID loss, initial → final | 0.321402 → 0.316057 | 0.321402 → 0.315527 |
| Held-out task-ID loss, initial → selected | 0.369275 → 0.367256 | 0.369275 → 0.370945 |
| Relative held-out change | −0.55% | +0.45% |
| Selected checkpoint | 5,000 | 5,000 |
| Median / maximum pre-clipping gradient norm | 4.794 / 71.198 | 4.960 / 44.720 |
| Updates exceeding clip norm 10 | 461 | 568 |
| Training worker minutes, including initialization | 48.401 | 50.402 |
| Training loop minutes, including probes/saves | 47.464 | 49.403 |
| Synchronized optimizer-update wall seconds | 1,712.823 | 1,768.251 |
| Peak CUDA allocated / reserved GiB | 1.732 / 1.846 | 1.732 / 1.846 |

All 155 trainable tensors received finite gradients on every update. Both final
models changed 122 tensors; 33 trainable tensors remained byte-identical under
the unchanged native BF16/FP32 optimizer behavior. All frozen tensors stayed
unchanged. Nonzero gradients do not imply every weight changed. Resolved AdamW
settings match the registration. Actual maximum recorded LR was 4.98008e-5 under
the native scheduler, versus the configured peak 5e-5; every used LR is logged.

The five checkpoints per arm were selected only by fixed held-out task-ID loss.
All **10 independent reloads passed**: 500 model tensors, optimizer/scheduler
state, byte-identical saved preprocessing/normalization, and fixed validation
losses (rtol1e-5, atol1e-6). Rollout manifests match the selected weight hashes:

- Original: `96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b`
- Uniform 5,000: `36fa77a84e598df895729895f78ac66088ec458e62457738d9f08178a7fd7f6e`
- Motion mixture 5,000: `528c57b3eb8407606a29e1c6166156c7a8ab00bd5ef01fae0f1a3e9d97b55070`

![Measured fixed-probe losses](../outputs/agent-workflow-v1/learning-5000-v1/loss-comparison.png)

Per-update EMA and secondary NL validation curves are also in each arm's
`losses.svg`; exact probes, gradients, LR and timing are in its JSONL logs.
The small uniform validation improvement is not statistically established
generalization. The validation episodes may overlap original pretraining.

### Actual RoboCasa behavior

All rows below executed actual policy weights in MuJoCo. These are task-ID
`CloseFridge` results, **not natural-language baseline results or an official
success-rate reproduction**. Success is the environment's actual predicate.
Closing is initial minus final summed door angle; negative values mean opening.
Two-door seed 45 retains ordered left/right angles. Peak force is the maximum
active robot-involving contact-frame normal component over 25 physics substeps
per action, including self-contact; it is not net hand force or hinge torque.

| Seed | Policy | Success / actions | Final door angle(s), ° | Closing, ° | Peak normal, N |
|---:|---|---|---:|---:|---:|
| 43 | Pretrained | No / 600 | 88.64 | −3.12 | 428.04 |
| 43 | Uniform | No / 600 | 13.11 | 72.41 | 405.16 |
| 43 | Motion | No / 600 | 17.33 | 68.20 | 633.45 |
| 44 | Pretrained | Yes / 304 | 0.20 | 83.12 | 88.55 |
| 44 | Uniform | Yes / 220 | 0.31 | 83.01 | 171.22 |
| 44 | Motion | No / 600 | 65.16 | 18.15 | 219.74 |
| 45 | Pretrained | No / 600 | 85.46, 11.40 | 71.07 | 328.21 |
| 45 | Uniform | No / 600 | −0.01, 28.27 | 139.66 | 188.69 |
| 45 | Motion | No / 600 | 85.46, 89.93 | −7.47 | 350.09 |
| 46 | Pretrained | No / 600 | 89.99 | −3.58 | 400.57 |
| 46 | Uniform | **Invalid / 1** | Excluded | Excluded | Excluded |
| 46 | Motion | Not run | — | — | — |
| 120–123 | All three | **Not run** | — | — | — |

The valid matched intersection is **43–45 only**: pretrained **1/3**, uniform
**1/3**, motion **0/3**. Do not compare pretrained 1/4 against candidate 1/3 or
count the invalid camera episode as a manipulation failure. Mean closing on
the common scenes was 50.36°, 98.36°, and 26.29°, respectively. Mean of episode
mean inference latency was 0.641, 0.656, and 0.655 seconds per prediction chunk.
No physical displacement guard fired on these valid scenes. The exact CSV also
contains loaded-contact action counts, EEF path length, best door progress,
per-episode latency, scene type and source paths. Each episode directory includes
its real RGB `rollout.mp4`, step trace, contact log and geometry catalog.

Uniform changed behavior substantially without improving measured success:
seed 43 nearly closed its door, and seed 45 closed the left door while the right
remained open. Faster seed-44 success (220 versus 304 actions) came with a contact
regression. Its 171.22 N peak at action152 was hand–door-handle contact. Motion's
633.45 N seed-43 peak at action200 was link5–link7 self-contact. These identities
come from active constraints and the saved geometry catalog, not visual proximity.

### Observation blocker and repair decision

Uniform seed46 passed exact reset qpos/qvel/state and all three camera comparisons.
The first inference used nonblack reset cameras (max255). After action1, all three
raw camera arrays had min=max=0; saved individual PNGs independently confirm this.
`invalid-camera.json` records detection **before action2**. The guard stopped before
preprocessing or another model inference, so **no black image was fed through a
new policy prediction**. This is a raw observation fault, not merely black video.
There was only ~0.3 mm EEF movement on the first action. Cause is unresolved;
neither policy intent nor geometry explains the simultaneous blackout from this
evidence. No rerun, replacement seed, renderer repair or final rollout was launched.

`partial-acceptance.json` applies the **existing** `agent_workflow.acceptance`
gate to the three valid common paired scenes. Uniform fails `no_success_gain`
and `contact_force_regression`; motion additionally fails per-seed success and
door-progress checks. Its held-out regression independently blocks promotion.
Both final decisions also record the invalid-observation blocker and incomplete
matrix. **Neither candidate is accepted. Final evaluation is not performed.**
This is a registered researcher-controlled learning comparison using existing
training, rollout and acceptance adapters; it is not a new autonomous LLM decision.
The raw execution trace and separate offline review trace distinguish execution
from subsequent audit. Historical human discoveries are not relabeled agent output.

Total stopped-run wall time: **8,177.719 seconds (2.272 hours)**. Serial GPU-worker
process time: **2.269 hours**, including training, reloads, CPU setup/I/O and
simulation. This is occupied worker wall-time, not measured active GPU kernel time.
No integrated utilization or power trace exists, so active GPU-hours and energy
remain null. A spot power reading is insufficient for an energy estimate.

All four protected historical checkpoint hashes still match; baseline files and
prior results were preserved. No dependencies changed. Only the initial workflow
review was committed (`628f820`); this learning implementation and results remain
uncommitted. Training and reloads are technically complete; the full 24-episode
verification milestone is **not** complete.

### Reproduction and remaining work

```powershell
# Existing artifacts: audit only, no inference, training or physics
python -B scripts/learning_repair_summary.py --output outputs/agent-workflow-v1/learning-5000-v1 --allow-incomplete
# Fresh full experiment, only after resolving the camera blocker and approving a new run
python -B -u scripts/learning_repair_experiment.py --output outputs/agent-workflow-v1/NEW-EXCLUSIVE-NAME
```

The trace contains every exact worker command, including all reload commands;
source snapshots and config hashes fix the executed implementation. The artifact
audit independently checks all 10,000 updates against sample order, all checkpoint
hashes, saved selection, actual rollout hashes and historical preservation.

Next requirement is a narrowly bounded observation-integrity check on seed46,
capturing raw cameras at reset and after the same first action in fresh processes,
before any codec. Resolve and document the camera failure before resuming robot
verification. This is a proposed acquisition plan, **not executed work**. Completing
the original valid matrix would require one replacement of the invalid uniform46
episode plus 13 unstarted episodes; that needs an explicit registered amendment
because it would total 25 attempts. Keep the same frozen checkpoints and final
seeds120–123; do not tune on final scenes. No additional training is justified by
the current comparison. The rare-motion proxy did not improve this observed
manipulation set and must not be presented as a verified contact-aware repair.
