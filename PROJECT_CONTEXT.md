# EmbodiedOps — Project Context

## Current verified milestone — 2026-10-19

The user's new authorization superseded the earlier agent/training deferrals below.
A bounded deterministic evidence/tool loop now executes configuration and learning
repair branches using the existing SmolVLA/RoboCasa runners. Four fresh paired
episodes gave 0/2 NL versus 1/2 task-ID success, but the candidate was rejected for
seed43 door/contact-force regressions. A new 512-update mixed-instruction run
covered all 461 training episodes, reloaded successfully, and was rejected because
mean held-out loss worsened 0.384626 -> 0.427864. No trained-policy rollout or final
seed120/121 evaluation was warranted. Historical checkpoints remain unchanged.
The backend is explicitly deterministic; no LLM API was used. A Windows cache-path
failure before any update required recorded operator recovery. See
`environment/agent-workflow-v1.md` and `outputs/agent-workflow-v1/run-01/` for the
registered budget, tool traces, losses, robot metrics, tests and limitations.
There is no accepted repair or generalization claim, and no automatic commit.

## One-line thesis

**EmbodiedOps is an agentic debugging and repair framework for learned robot policies. It uses controlled simulation evidence to diagnose why a policy fails, routes the failure to an appropriate repair, and verifies whether the repair actually restores performance.**

## Product framing

The core question is:

> A robot policy failed in rollout. Why did it fail, what should we change, and did that change actually make it better?

EmbodiedOps is not intended to be an LLM that directly replaces the robot controller.

The reasoning agent operates at the **robot-policy engineering layer**:

```text
ROLLOUT
  ↓
DIAGNOSE
  ↓
REPAIR ROUTING
  ├─ pipeline / config fix
  ├─ targeted data
  ├─ augmentation
  ├─ supervised VLA post-training
  └─ RL-based improvement later
  ↓
VERIFY
```

## Why this matters

Robot-policy failures can come from very different sources:

- instruction / task-conditioning mismatch,
- wrong camera mapping / ordering,
- preprocessing or normalization mismatch,
- action scaling,
- temporal alignment or latency,
- environment distribution shift,
- insufficient demonstration coverage,
- genuine policy limitations.

Blindly fine-tuning after every failure can waste compute and still fail to fix the real problem.

EmbodiedOps should use controlled diagnostic probes and real rollout evidence to decide whether the correct response is to:
- fix the pipeline,
- change the data,
- post-train the policy,
- collect targeted demonstrations,
- or take another bounded repair action.

## Agentic behavior

The system should be genuinely sequential:

```text
Observe failure
→ form hypotheses
→ choose diagnostic probe
→ execute probe
→ observe result
→ update hypotheses
→ choose repair
→ execute repair
→ verify
→ continue / stop
```

The next action must depend on the previous experiment result.

A fixed script that always runs the same tests is not the target.

## Research framing

A useful research question is:

> Can a tool-using reasoning agent diagnose robot-policy failures and choose the appropriate repair with fewer unnecessary experiments and less wasted training than fixed debugging / post-training workflows?

Possible future metrics:
- root-cause diagnosis accuracy,
- top-k diagnosis accuracy,
- probes-to-diagnosis,
- time-to-recovery,
- GPU-hours-to-recovery,
- avoided unnecessary retraining,
- repair success rate,
- recovered success rate,
- nominal-regression after repair.

A future controlled benchmark can inject known faults so root-cause ground truth is available.

## Short-term job-search milestone

Before building the full diagnosis agent, first obtain direct VLA evidence.

### Immediate target

**SmolVLA + RoboCasa + CloseFridge**

Deliver:
- run a real SmolVLA baseline in RoboCasa,
- inspect dataset structure,
- inspect observation/state/action spaces,
- inspect one real training batch,
- understand the VLA architecture and flow-matching objective,
- run a small real post-training / fine-tuning experiment,
- evaluate multiple rollouts,
- record success rate,
- record inference latency,
- save rollout video,
- write a concise technical report.

Only after these are actually completed should the resume claim direct SmolVLA/VLA fine-tuning.

## Hackathon direction

The hackathon version should then extend the VLA baseline into:

```text
SmolVLA / RoboCasa policy
        ↓
controlled failure / fault injection
        ↓
structured rollout evidence
        ↓
NVIDIA / Nebius reasoning agent
        ↓
diagnostic probe selection
        ↓
repair routing
        ↓
config fix OR targeted post-training
        ↓
verification
```

The reasoning model should likely use an NVIDIA model via Nebius Token Factory.

The low-level robot policy itself does not need to be an LLM.

## Role of PushT

PushT is **not** the headline task.

It is a lightweight regression / learning backend already used to validate:
- LeRobot installation,
- CUDA training,
- Diffusion Policy forward / backward,
- checkpoint saving,
- policy evaluation,
- rollout-video generation.

It may remain useful for fast smoke tests and learning, but the main visual demo should use 3D manipulation.

## Primary benchmark direction

Preferred main benchmark:
- MuJoCo / RoboCasa
- first task: `CloseFridge`
- second task candidate: `OpenDrawer`

Preferred main VLA:
- SmolVLA first
- pi0 later as a stretch adapter

Diffusion Policy remains useful as:
- a learned-policy baseline,
- a way to understand action diffusion / chunking,
- a lightweight development backend.

RL is a later extension, not an MVP blocker.

## Learning goals

Important robot-learning concepts should not be hidden behind one-line CLI commands.

For each training / evaluation stage, inspect and understand:

- dataset organization,
- observation space,
- proprioceptive state,
- camera inputs,
- action space,
- action representation,
- action chunking,
- normalization / preprocessing,
- batch tensor shapes,
- model architecture,
- loss function,
- optimizer,
- learning rate,
- batch size,
- training steps,
- checkpoint contents,
- rollout loop,
- success / reward metrics,
- inference latency.

### Diffusion Policy concepts

Understand:
- action trajectory noising,
- diffusion timestep sampling,
- epsilon prediction,
- observation conditioning,
- denoising at inference,
- action horizons and execution chunks.

### SmolVLA concepts

Understand:
- image inputs,
- language tokens,
- proprioceptive state,
- VLM backbone,
- action expert,
- flow matching,
- noisy / interpolated action chunks,
- post-training / fine-tuning,
- inference action chunks.

Useful inspection utilities:
- `scripts/inspect_dataset.py`
- `scripts/inspect_batch.py`
- `scripts/inspect_policy_config.py`

They should print real values and tensor shapes from actual data, never hard-coded educational output.

## Current verified local machine

Windows 11

GPU:
- NVIDIA GeForce RTX 3060 Laptop GPU
- 6 GB VRAM

## Verified policy environment

Conda environment: `embodiedops`

- Python 3.12
- PyTorch 2.11.0+cu128
- CUDA available
- Hugging Face LeRobot 0.6.1
- NumPy 2.2.6
- opencv-python-headless 4.13.0.92

Verified:
- PushT loads and renders,
- Diffusion Policy forward/backward runs on CUDA,
- checkpoint saving works,
- policy evaluation works,
- rollout MP4 generation works.

Important OpenCV note:
`gym-pusht` declares `opencv-python`, while LeRobot uses `opencv-python-headless`.
Do not install both merely to make `pip check` clean.

## Phase 1A static verification — 2026-10-16

A new isolated `embodiedops-robocasa` environment (Python 3.12.15, LeRobot 0.6.1)
loaded pinned SmolVLA / CloseFridge artifacts. A real CloseFridge dataset
episode/sample was loaded and inspected; saved preprocessing, forward-only
flow-matching loss, and static `[1,50,12]` action generation were verified
on the local RTX 3060. See `environment/phase1a.md` for revisions,
measured tensors, memory, timing, and limitations. No RoboCasa/robosuite installation,
simulation, backward pass, or training occurred. `RESUME_EVIDENCE.md` remains
unchanged at the user's explicit request. Phase 1B needs separate approval.

## Verified simulation environment

### Phase 1B simulator/controller verification — 2026-10-17

The isolated `embodiedops-robocasa` environment now runs pinned RoboCasa
`56e355cc` / robosuite `aaa8b9b2` with MuJoCo 3.3.1. RoboCasa required NumPy
2.2.5; all other pre-existing package versions were retained. A seed-42
CloseFridge reset rendered three 256x256 RGB cameras and exposed the matching
16D state / 12D action contract. Exactly five neutral-command steps and four
single-step controller probes completed with finite observations. Base/EEF
neutral drift was small; the binary gripper interface commands opening or
closing and cannot hold an arbitrary intermediate opening. See
`environment/phase1b.md` and `environment/phase1b-results.json` for measured
vectors, controller transformations, timing, success semantics, and artifacts.
No learned-policy rollout or training occurred. Phase 1C requires separate
approval. `RESUME_EVIDENCE.md` remains unchanged at the user's request.

### Earlier standalone MuJoCo verification

Conda environment: `embodiedops-mujoco`

- Python 3.10
- MuJoCo 3.3.7
- robosuite 1.5.2
- NumPy 1.26.4

Verified:
- Panda robot loads,
- robosuite `Lift` simulation runs,
- MuJoCo 3D viewer works.

The random-action demo is only a simulator sanity check and is not a trained Lift policy.

## Environment rule

Keep policy and simulation environments separate unless a deliberate new environment is designed.

Do not install robosuite into the existing `embodiedops` policy environment.

A previous robosuite install downgraded NumPy and broke the LeRobot dependency set.

Do not casually merge environments.

## Repository

Canonical local repository:

```text
C:\Users\Alan\Projects\embodiedops
```

GitHub repository:

```text
https://github.com/hanlizhang/embodiedops
```

The Git repository is the source of truth for code and project documentation.

Current scaffold includes:
- `README.md`
- `.gitignore`
- `RESUME_EVIDENCE.md`
- `environment/policy.md`
- `environment/simulation.md`
- `src/`
- `scripts/`
- `tests/`

## Evidence discipline

`RESUME_EVIDENCE.md` is the authoritative ledger of what has actually been implemented and verified.

Do not claim planned work as completed.

Current verified evidence does **not yet** include:
- SmolVLA fine-tuning,
- RoboCasa integration,
- agentic diagnosis,
- automatic repair routing,
- autonomous policy improvement,
- pi0 integration.

## Phases

### Phase 0 — complete
Environment and pipeline validation:
- CUDA / PyTorch,
- LeRobot,
- PushT,
- Diffusion Policy smoke training,
- checkpoint saving,
- policy evaluation,
- rollout video,
- MuJoCo,
- robosuite Panda Lift.

### Phase 1 — immediate
SmolVLA + RoboCasa + CloseFridge baseline:
- environment strategy,
- dataset,
- state/action spaces,
- batch inspection,
- pretrained checkpoint,
- baseline evaluation,
- small fine-tuning run,
- success rate,
- latency,
- rollout video.

### Phase 2
Controlled fault injection and diagnostic benchmark:
- instruction-conditioning mismatch,
- camera mapping / ordering,
- normalization mismatch,
- action scaling,
- pose distribution shift,
- selected visual / temporal perturbations where technically valid.

### Phase 3
Diagnostic probe library:
- controlled A/B tests,
- pose sweeps,
- camera checks,
- preprocessing checks,
- action / state checks.

### Phase 4
Nebius / NVIDIA reasoning agent:
- observe,
- hypothesize,
- choose probe,
- inspect result,
- update diagnosis.

### Phase 5
Repair routing:
- pipeline/config fix,
- targeted data,
- augmentation,
- SmolVLA post-training,
- other bounded repair mechanisms.

### Phase 6
Verification:
- before / after success,
- latency,
- regression checks,
- recovery metrics.

### Phase 7 — optional extensions
- RL / online policy improvement,
- pi0 adapter,
- more RoboCasa tasks,
- larger controlled benchmark.

## Coding / agent rules

Before substantial changes:
1. read this file,
2. read `README.md`,
3. read `RESUME_EVIDENCE.md`,
4. read `environment/*.md`,
5. inspect `git status`,
6. explain the planned change.

Do not:
- install / upgrade / downgrade dependencies without approval,
- launch long GPU jobs without approval,
- merge environments casually,
- restart WSL troubleshooting,
- fabricate benchmark results,
- mark planned work as completed.

Prefer:
- small reviewable commits,
- reproducible configs,
- fixed seeds where appropriate,
- structured metrics,
- explicit experiment manifests,
- real tensor / dataset inspection tools,
- clear before / after evidence.

## Immediate next task

Do **not** build the full agent yet.

First design and execute Phase 1:

**SmolVLA + RoboCasa + CloseFridge**

Before installing anything:
- inspect current LeRobot 0.6.1 support,
- determine whether a third environment is warranted,
- identify the dataset and pretrained checkpoint,
- identify observation and action spaces,
- estimate local 6 GB VRAM feasibility,
- determine what should run locally vs on Nebius,
- propose the first small reproducible baseline and fine-tuning experiment.

## Current control experiment — 2026-10-19

The earlier Phase 1 planning sections above are historical. The current verified
control work is recorded in `environment/control-horizon-v1.md`: three pretrained
SmolVLA task-ID H=20 episodes, paired with contract-matched historical H=50 seeds
42–44. Both have one success; H20 seed 42 stopped at the existing displacement
guard after arm–fridge collision. Contact-force and trajectory diagnostics are
implemented without privileged policy inputs. Historical prefixes are not
bitwise identical, and H50 contact measurements are unavailable. No H10, extra
seeds, training or dependency changes were made. Phase2 A/B training remains a
deferred proposal. Next proposed work is contact-progress/controller-tracking
analysis and execution safeguards, not an LLM diagnostic agent.

## Current learning milestone — 2026-10-20

The later deterministic workflow is implemented; its reviewed initial milestone
is commit628f820 (implementation bfc1978). The new sustained learning comparison
completed 5,000 updates per arm from original SmolVLA weights, comparing episode-
balanced and inferred-motion sampling. All ten checkpoints reloaded correctly.
Actual matched seeds43–45 yielded pretrained1/3, uniform1/3, motion0/3 success.
Both trained candidates were rejected for observed regressions. Uniform seed46
then produced all-zero raw cameras after action1; the observation guard stopped
the run before another prediction. Ten valid and one invalid episodes exist;
final seeds120–123 remain unexecuted. Do not label the full milestone complete.

See `environment/learning-repair-v1.md` and its results JSON for the registered
matrix, loss curves, physical evidence, checkpoint hashes and blocked decision.
All existing weights/results are preserved. No further training is planned;
the specific raw-camera integrity blocker must be resolved before an explicitly
registered continuation of the unchanged checkpoint/seed matrix. The earlier
natural-language/mixed-instruction A/B proposal remains separate and unexecuted.

### 2026-10-20 continuation completed

The camera fault was reproduced at MuJoCo readback after obsolete-renderer
garbage collection and corrected with owner-context resource cleanup. Two missing
seed-46 candidate episodes completed; all 12 diagnostic pairs are now valid.
Success is pretrained 1/4, episode-balanced 1/4, motion-mixture 0/4. Both candidates
are rejected by the existing gate; final seeds 120–123 remain unused because
neither qualifies. All 808 original experiment files are hash-preserved. No new
training or dependencies. See `environment/learning-repair-v1-completion.md`.

### 2026-10-20 controlled optimizer-precision result

One matched episode-balanced FP32-master run completed 5,000 updates. Actual BF16
forward changes appeared in all 33 normalization tensors unchanged under native
training, but fixed task-ID validation worsened: 0.387121 versus native 0.367256
and original 0.369275. The early loss spike also increased. Preflight reload and
exact resumed-update checks passed; the first sustained-checkpoint independent
reload timed out after 300 seconds. All five saved checkpoints passed CPU
serialized-state hash checks, which do not replace operational reload validation.
No new robot episodes ran, no candidate was promoted, and final seeds remain unused.
All 957 protected historical files are unchanged. See
`environment/optimizer-precision-v1.md`. No automatic retry or further variant.
