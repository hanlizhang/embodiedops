# Phase 2 v1 — instruction-conditioned SmolVLA learning

Date: 2026-10-19. Only the bounded sanity experiment and offline audits have run.
The sustained A/B comparison and all new simulator evaluations require approval.
No Agent/UI, dependency changes, commits, or evidence-ledger changes were made.

## Audit of Phase 1D

The committed `scripts/phase1d_train.py` matches its original experiment manifest
(SHA256 `b5f6c76293ada54a1af4041f78c9333cb7093e43e38992850b40c246dcc8d391`).
The step-300 checkpoint remains
`outputs/phase1d/training-v2/checkpoints/000300/pretrained_model`, SHA256
`0e011aa494e0d4e370b11feb5eec6e0c49af2fadd1f3f97ee03056909de0fe6d`.
Its existing separate-process reload verified all 500 state tensors and fixed loss.

Direct observations:

- 300 updates, batch size 1: 300 sampled transition anchors from 221 of 461
  training episodes. Each anchor supervises up to 50 future actions; this was
  not 300 complete episodes or an epoch through 139,724 training frames.
- Fixed eight-frame training loss rose 0.301324 → 0.352550; validation rose
  0.531797 → 0.594187. These small probes were noisy estimators of the dataset
  objective even though their noise was fixed. They are evidence against a
  loss-improvement claim, not proof that no useful control behavior changed.
- Frame task indices resolve exclusively to natural language. The 300 sampled
  anchors used 157 singular and 143 plural instructions. `CloseFridge` in the
  metadata task table was not used as the frame-level training instruction.
  Evidence: `outputs/probes/instruction-distribution-v1/finetuning-language-audit.json`.
- Prior provenance audit found bare task IDs in the historical unified dataset
  associated with the released model. Its training config has a null dataset
  revision: the exact executed upstream snapshot/overrides and pretraining
  overlap remain unknown. This does not prove the backbone never saw natural
  language. Community success claims are not local benchmark evidence.
- All 155 intended tensors received finite gradients; 122 changed and 33 BF16
  expert normalization weights did not. All frozen weights remained unchanged.
- AdamW: peak LR 1e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay 1e-10,
  gradient clipping 10. Native scheduler scaled its 1,000/20,000 warmup/decay
  preset to 15/300: first update LR 6.25e-6, last 2.502673e-6.

Installed LeRobot 0.6.1 source was inspected directly: `SmolVLAPolicy.forward`,
`VLAFlowMatching.forward`, `update_policy`, optimizer factory, scheduler and
checkpoint save/load utilities. The native update calls `policy.train()`,
forward, backward, gradient clipping, AdamW, zero-grad, then scheduler step.
No missing optimizer step or failure to restore training mode was found.

Saved model normalization is retained, rather than substituting local dataset
statistics. Three cameras map left/right/wrist to camera1/2/3. Collated RGB
tensors are `[1,3,256,256]`, converted from uint8 to [0,1]; native model image
preparation resizes/pads and rescales for the vision encoder. State is
`[1,1,16]`; target action chunk `[1,50,12]`; temporal padding mask `[1,50]`.
Internal state/actions are padded to width 32. With normalized action `a`,
Gaussian noise `e`, and sampled time `t`, the network receives
`x_t = t*e + (1-t)*a` and predicts velocity `e-a`. LeRobot reduces squared
velocity error over the 12 real action dimensions and valid future timesteps,
dividing by `12 * valid_timestep_count`. No padding-denominator bug was found.

Precision is a real limitation, but an optimizer bug is not established.
Native BF16 trainable parameters also have BF16 Adam moments; there is no FP32
master-parameter path here. Small updates to near-unit normalization weights
can round away. For the first expert input-normalization tensor, the historical
step-300 weights are all 1.0 and saved Adam moments are nonzero. A FP32 estimate
from its final moments gives a largest update of 1.5335e-6; rounding that update
back to BF16 changes zero elements. This supports rounding as an explanation for
that tensor, without reconstructing every historical optimizer operation.
Whether changing precision improves control is untested.
No architecture or precision modification was made to obtain the sanity result.

Historical closed-loop results remain 0/10 pretrained versus 3/10 step-300
fine-tuned on natural language. The task-ID probe was 1/3 versus 0/3 NL on
design seeds 42–44. Small samples, instruction mismatch, limited frame coverage,
and uncertain pretraining overlap make these inconclusive for generalization.
They do not isolate the causal role of precision, language, or sample count.

## New implementation and measured sanity test

`scripts/phase2_train_v1.py` is standalone and imports installed LeRobot training
primitives, not the historical training runner. It creates an exclusive output
directory and never imports or creates a simulator. Existing output paths are
rejected. `scripts/phase2_audit_v1.py` checks historical/source/checkpoint evidence
and preservation. All generated artifacts are ignored by Git under `outputs/`.

Pins retained:

- Model `lerobot/smolvla_robocasa`:
  `f7af588d11e30b0069f200792fb285904b74c80b`.
- Dataset `pepijn223/robocasa_CloseFridge`:
  `b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4`.
- SmolVLM backbone: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Torch 2.11.0+cu128, LeRobot 0.6.1, accelerate 1.15.0, NumPy 2.2.5, av 15.1.0.

Frozen VLM; trainable action expert and state projection: 99,880,992 parameters
in 155 tensors. Native BF16/FP32, no AMP, PEFT, or gradient checkpointing.
The supported native AdamW update is unchanged. Config is
`environment/phase2-sanity-v1.json`: 512 updates, batch 1, peak LR 1e-4,
16 warmup steps, cosine decay over 512 steps to 1e-5, seed 142.

Thirty-two distinct training frames from 32 episodes repeat in the same order
16 times, with fresh noise/time per update. Fixed probes use independent noise
seeds, so a lower probe loss is not merely memorization of one noise draw.

| Optimizer updates | Fixed training NL loss | Held-out NL loss | Held-out task-ID loss |
|---:|---:|---:|---:|
| 0 | 0.379791 | 0.405792 | 0.363459 |
| 128 | 0.418087 | 0.575067 | 0.580143 |
| 256 | 0.311280 | 0.540472 | 0.544884 |
| 384 | 0.249219 | 0.527524 | 0.533675 |
| 512 | 0.235881 | 0.522031 | 0.527872 |

Training probe decreased **37.89%**, with a transient increase at step 128.
Twenty-nine of the 32 fixed training samples improved; noise/time fingerprints
matched across every probe step.
Raw training loss means: first 64 updates 0.556108; last 64 0.229807 (different
noise draws, so these are descriptive averages). This demonstrates learning on
repeated observations, not complete memorization or improved task success.
Both held-out instruction losses worsened; no generalization claim is warranted.

All 512 updates had finite loss and gradients on all 155 intended tensors.
122 tensors changed; 33 normalization tensors stayed unchanged; all frozen
state tensors stayed byte-identical. The monitored expert normalization tensor
had gradient norms 0.000594–0.049565 and zero changed elements across all steps.

Training-loop time including probes and checkpoint work: **282.75 s** (4.71 min),
excluding imports, model/data setup, and the separate reload process. Mean
measured update time: 0.2744 s. Peak training CUDA allocated **1.732 GiB**,
reserved **1.846 GiB**. Initialization/system GPU usage is separate.

Saved checkpoint:
`outputs/phase2/sanity-v1/checkpoints/000512/pretrained_model`.
Weight SHA256:
`4b07c48525cb916be86b080501b5a23015dbda27f79869502bb378f786dd7407`.
Separate-process reload passed equality of all 500 model tensors, full optimizer
state, scheduler state, processor bytes and both fixed validation loss vectors.
RNG state is saved/restored by LeRobot; an interrupted-run continuation was not
executed. The runner exposes verification, not a resume-training command.

Artifacts in `outputs/phase2/sanity-v1/`: settings/resolved config, manifest,
split and complete sample order, raw/EMA update log, fixed-probe vectors and
noise/time hashes, parameter audit, checkpoint verification, summary, and
`losses.svg`. The checkpoint is a learning sanity artifact, not a candidate
recommended for deployment; its validation score is worse than pretrained.

## Reproducible A/B proposal — not executed

Keep the native precision so A/B changes only task conditioning. Use the same
pinned pretrained initialization for each arm, never the sanity or step-300
weights. Proposed budget: **2,500 updates per arm**, supported by the observed
loss reduction by 256–512 repeated-sample updates and feasible measured runtime.
This budget gives about 5.4 sampled anchors per training episode, not five epochs.
It is a bounded next experiment, not a guarantee of convergence.

| Setting | A: natural | B: mixed |
|---|---|---|
| Config | `phase2-natural-v1.json` | `phase2-mixed-v1.json` |
| Instruction | Original frame instruction | Even update indices NL; odd indices `CloseFridge` |
| Updates / batch | 2,500 / 1 | Identical |
| Samples and ordering | One random frame per shuffled episode, repeated episode cycles | Identical |
| Peak / final LR | 1e-4 / 1e-5 | Identical |
| Warmup / decay | 125 / 2,500 steps, native cosine | Identical |
| Validation / checkpoint interval | 250 / 500 updates | Identical |

B uses 1,250 NL and 1,250 task-ID updates. It does not duplicate the action
targets or give itself twice the optimizer budget. Demonstration actions,
processor statistics, camera mapping and architecture stay identical.

LeRobot's existing episode split is reused: 461 train / 52 validation episodes,
139,724 / 15,719 frames, no episode intersection. Training order visits all 461
episodes before revisiting an episode; random generators are local and seeded.
The 64 validation anchors cover every held-out episode at least once and contain
64 distinct frames. Validation is sample-weighted (12 episodes receive a second
anchor), not exactly episode-balanced. Each frame is tested in both instruction
formats with identical explicit noise/time tensors. Their hashes are recorded.
Training uses the same per-update torch seed across A/B; native noise is drawn
before language-dependent network computation. GPU determinism is requested
with warnings; cross-hardware bit identity is not promised.

Checkpoint selection is the lowest equally weighted NL/task-ID fixed validation
mean among saved steps 500/1,000/1,500/2,000/2,500; earliest tie. Report each
condition and the pretrained baseline score. If every candidate worsens, say so
and do not call the selected candidate an improvement. No simulation outcomes
may influence checkpoint selection. The 64-frame validation set is a tuning set,
not a final test set. A single fixed flow draw per frame still has estimator noise.

Expected local cost, extrapolated rather than measured: **15–20 minutes per
arm**, including probes, checkpoint saves and a reload check; allow 40–50 minutes
for both with startup and disk overhead. Five full checkpoints per arm use about
13.2 GB total across both arms (measured sanity checkpoint: 1.320 GB). Roughly
41.6 GB disk space was available after the sanity run. Budget and curves are
reported now for approval; no automatic extension or hyperparameter sweep.

## Closed-loop evaluation proposal — not executed

Exact matrix is in `phase2-evaluation-plan-v1.json`: three checkpoints
(pretrained, validation-selected A, validation-selected B) × two instruction
formats × consecutive seeds **100–109** = **60 new episodes**. Seeds 42–51 are
design evidence and excluded. Preregister seeds 110–119 for a subsequent full
60-episode confirmatory matrix if justified and separately approved.

Use the pinned RoboCasa/robosuite/MuJoCo stack and existing physical contracts:
600-action horizon, 50-action chunks/queue, 10 flow steps, identical cameras,
normalization and controller. For each seed, save one canonical reset and require
matching model/fixture metadata, qpos/qvel/time, 16D state, and all camera pixels
for every arm. Match policy sampling seeds per inference call and record noise
hashes without extra draws. Early success shortens only that arm; comparisons
after a divergence are trajectories, not identical physical-state interventions.

Classify single/two-door scenes from reset fixtures and relevant joints before
examining outcomes; keep all seeds even if the split is unbalanced. Report actual
RoboCasa success, door-angle progress, stop reasons, actions, latency, duration,
videos, individual paired outcomes and uncertainty. A nearly closed door is not
official success. Keep runtime failures visible rather than silently dropping
or replacing seeds. New exclusive output root: `outputs/phase2/evaluation-v1/`.

Historical rollout-only durations averaged 82 s (range 32–121 s); 60 episodes
therefore imply roughly 82 minutes of rollout time, plus reset/load/render/I/O.
Allow approximately two hours; this is an estimate, not a promised runtime.
All 60 are new scenes, so historical runs cannot substitute for matrix cells.

This measures CloseFridge instruction robustness and regression under task IDs.
It cannot establish retention of all 50 upstream tasks. Cross-task retention
requires a separately designed benchmark. Local episode-disjoint validation is
also not proof of independence from original VLA pretraining demonstrations.

## Commands

Run from an Anaconda/Miniconda PowerShell prompt:

```powershell
conda activate embodiedops-robocasa
cd C:\Users\Alan\Projects\embodiedops
```

Completed commands (training refuses an existing output directory; do not rerun
against this directory):

```powershell
python -B scripts/phase2_train_v1.py --config environment/phase2-sanity-v1.json --output outputs/phase2/sanity-v1
python -B scripts/phase2_train_v1.py --config environment/phase2-sanity-v1.json --output outputs/phase2/sanity-v1 --verify outputs/phase2/sanity-v1/checkpoints/000512
python -B scripts/phase2_audit_v1.py
```

Proposed commands, **only after approval**:

```powershell
python -B scripts/phase2_train_v1.py --config environment/phase2-natural-v1.json --output outputs/phase2/natural-v1
python -B scripts/phase2_train_v1.py --config environment/phase2-mixed-v1.json --output outputs/phase2/mixed-v1
$selectedA = (Get-Content outputs/phase2/natural-v1/selected-checkpoint.json | ConvertFrom-Json).checkpoint
python -B scripts/phase2_train_v1.py --config environment/phase2-natural-v1.json --output outputs/phase2/natural-v1 --verify (Join-Path outputs/phase2/natural-v1 $selectedA)
$selectedB = (Get-Content outputs/phase2/mixed-v1/selected-checkpoint.json | ConvertFrom-Json).checkpoint
python -B scripts/phase2_train_v1.py --config environment/phase2-mixed-v1.json --output outputs/phase2/mixed-v1 --verify (Join-Path outputs/phase2/mixed-v1 $selectedB)
```

The experiment demonstrates flow-matching supervision, padding-aware losses,
controlled instruction interventions, deterministic episode sampling, precision
and update auditing, checkpoint reproducibility, and separating offline fit
from closed-loop task performance. Sustained control improvement remains to be
measured; the current achievement is verified learning and a controlled design.
