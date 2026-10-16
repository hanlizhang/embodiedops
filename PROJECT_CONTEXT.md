# EmbodiedOps — Project Context

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
