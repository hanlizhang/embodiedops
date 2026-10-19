# EmbodiedOps

**Agentic debugging and repair for learned robot policies.**

**2026-10-19: bounded deterministic workflow executed.** The CLI now connects
historical VLA evidence, diagnostic tool selection, versioned instruction repair,
SmolVLA training, checkpoint verification and matched RoboCasa evaluation.
Four new episodes and 512 optimizer updates completed. Instruction repair increased
diagnostic success from 0/2 to 1/2 but was rejected for regressions; the learning
candidate was rejected for worse validation loss. No accepted improvement or LLM
agent is claimed. See [the executable case study](environment/agent-workflow-v1.md)
and `scripts/agent_workflow.py`. Earlier status/roadmap text below is historical.

> Diagnose why a robot policy fails, choose the right repair, and verify the fix.

EmbodiedOps is an experimental open-source framework for closing the loop between robot-policy evaluation and repair.

The core question is:

> A robot policy failed in rollout. Why did it fail, what should we change, and did that change actually make it better?

Rather than assuming every failure requires another training run, EmbodiedOps aims to use controlled simulation evidence to distinguish between issues such as:

- instruction / task-conditioning mismatch,
- camera or observation-pipeline errors,
- preprocessing / normalization mismatch,
- action-scaling or temporal issues,
- environment distribution shift,
- insufficient demonstration coverage,
- genuine policy limitations.

The intended closed loop is:

```text
ROLLOUT
  ↓
DIAGNOSE
  ↓
CONTROLLED PROBE
  ↓
REPAIR ROUTING
  ├─ pipeline / config fix
  ├─ targeted data
  ├─ augmentation
  ├─ VLA post-training
  └─ RL-based improvement later
  ↓
VERIFY
```

## Current status

Verified locally:

- PyTorch + CUDA on an RTX 3060 Laptop GPU
- Hugging Face LeRobot 0.6.1
- PushT simulation
- Diffusion Policy forward / backward training on GPU
- checkpoint saving
- PushT evaluation and rollout-video generation
- MuJoCo 3.3.7
- robosuite 1.5.2
- Panda `Lift` simulation

PushT is a lightweight development / regression backend, not the intended headline demo.

The immediate project milestone is:

**SmolVLA + RoboCasa + CloseFridge**

Before implementing the full diagnosis agent, the project will first establish a real VLA baseline and small post-training experiment with measurable success rate, inference latency, and rollout videos.

## Planned roadmap

1. **VLA baseline**
   - SmolVLA
   - RoboCasa / CloseFridge
   - dataset / state / action inspection
   - baseline evaluation
   - small fine-tuning run

2. **Fault injection**
   - instruction-conditioning faults
   - camera / observation faults
   - normalization / action faults
   - controlled distribution shifts

3. **Diagnostic probes**
   - controlled A/B tests
   - pose / camera sweeps
   - preprocessing / action checks

4. **Agentic diagnosis**
   - NVIDIA / Nebius reasoning model
   - observe → hypothesize → probe → update

5. **Repair routing**
   - config / pipeline fix
   - targeted data
   - VLA post-training
   - other bounded repair operators

6. **Verification**
   - before / after success rate
   - latency
   - regression checks
   - recovery metrics

## Project documentation

- `PROJECT_CONTEXT.md` — current direction, verified setup, project phases
- `AGENTS.md` — rules for coding agents
- `WORKFLOW.md` — ChatGPT / Codex / Git / training workflow
- `RESUME_EVIDENCE.md` — evidence ledger for completed work
- `environment/` — verified environment notes

## Evidence discipline

Planned features are not treated as completed features.

`RESUME_EVIDENCE.md` is the authoritative record for what has actually been implemented and verified.
