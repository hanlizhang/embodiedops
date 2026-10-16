# EmbodiedOps — Development Workflow

## Roles

### ChatGPT Project chat
Use for:
- project direction,
- architecture decisions,
- VLA / robot-learning explanations,
- paper / benchmark discussion,
- debugging strategy,
- hackathon narrative,
- resume / portfolio framing.

### Codex local project
Use for:
- reading and modifying the local Git repository,
- writing Python / configs / tests,
- running short diagnostics,
- inspecting datasets and tensor shapes,
- running approved training / evaluation commands,
- reviewing diffs,
- committing code.

### Windows terminal / GPU
This machine is the primary local compute worker:
- RTX 3060 Laptop GPU, 6 GB VRAM,
- local Conda environments,
- MuJoCo / robosuite,
- LeRobot / CUDA.

### GitHub
Canonical remote source:
- code,
- configuration,
- project documentation,
- small structured metrics / reports.

Do not commit:
- large checkpoints,
- large generated datasets,
- secrets,
- unnecessary binary artifacts.

Use Hugging Face Hub or cloud storage later for large checkpoints if needed.

## Local repository

```text
C:\Users\Alan\Projects\embodiedops
```

Codex should open this exact local repository.

Do not create a duplicate Codex copy of the project.

## Context persistence

The durable context lives in:
- `PROJECT_CONTEXT.md` — project state and direction,
- `AGENTS.md` — coding-agent rules,
- `RESUME_EVIDENCE.md` — verified implementation evidence,
- `environment/*.md` — verified environments,
- `README.md` — public-facing project description.

Chat history is useful but is not the source of truth.

At the end of each milestone:
1. update `PROJECT_CONTEXT.md`,
2. update `RESUME_EVIDENCE.md` only with verified facts,
3. commit the changes.

## Git workflow

Before work:

```powershell
cd C:\Users\Alan\Projects\embodiedops
git status
git pull
```

After a small verified change:

```powershell
git status
git diff
git add <specific files>
git commit -m "<meaningful message>"
git push
```

Avoid large unreviewed commits.

## Environment workflow

### Policy environment

```powershell
conda activate embodiedops
```

Use for:
- LeRobot,
- PyTorch / CUDA,
- PushT,
- Diffusion Policy,
- future policy-side tooling where dependencies remain compatible.

### Simulation environment

```powershell
conda activate embodiedops-mujoco
```

Use for:
- MuJoCo,
- robosuite,
- Panda simulation.

### RoboCasa / SmolVLA

Do not assume either existing environment is correct.

Codex must first propose:
- dependency requirements,
- environment strategy,
- whether a third Conda environment is needed.

No installation until approved.

## Training workflow

For every real training run:

1. Record the exact config.
2. Record dataset / revision.
3. Record seed.
4. Record model / checkpoint.
5. Record device and GPU.
6. Record start / end time.
7. Save machine-readable metrics.
8. Save only useful rollout videos.
9. Summarize the result in a report.
10. Update `RESUME_EVIDENCE.md` only if the result is verified.

Do not treat a smoke test as a meaningful model result.

## Current immediate workflow

```text
1. Inspect current repo and docs
2. Research current LeRobot 0.6.1 RoboCasa + SmolVLA support
3. Propose environment strategy
4. Inspect dataset and observation/action spaces
5. Run baseline evaluation
6. Inspect one real batch and model config
7. Run a small approved SmolVLA fine-tuning experiment
8. Evaluate success rate and inference latency
9. Save rollout video + metrics + technical note
10. Only then begin fault injection / agentic diagnosis work
```
