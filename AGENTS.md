# AGENTS.md

## Project
EmbodiedOps is an agentic debugging and repair framework for learned robot policies.

## Read first
Before substantial work, read:
- `PROJECT_CONTEXT.md`
- `README.md`
- `RESUME_EVIDENCE.md`
- `environment/policy.md`
- `environment/simulation.md`
- `WORKFLOW.md`

## Source of truth
The Git repository is the source of truth for code and project documentation.

## Current priority
The immediate milestone is:

**SmolVLA + RoboCasa + CloseFridge**

The baseline, sustained fine-tuning, physical diagnostics and bounded deterministic
workflow are implemented. The current milestone is portfolio delivery from
completed evidence. Further training, policy inference, simulation evaluation,
sampling/precision/horizon sweeps and new Agent features are stopped unless
explicitly requested. See `PROJECT_CONTEXT.md` for current measured outcomes.

## Current project direction
Longer-term loop:

```text
rollout
→ diagnose
→ choose diagnostic probe
→ update hypothesis
→ route to repair
→ config fix OR targeted post-training
→ verify
```

PushT is only a lightweight regression / learning backend.

## Environment rules
- Do not install, upgrade, downgrade, or uninstall dependencies without explicit user approval.
- Do not launch long GPU training jobs without explicit user approval.
- Keep `embodiedops` and `embodiedops-mujoco` separate unless a deliberate new environment is proposed and approved.
- Never install robosuite into the existing `embodiedops` environment.
- Do not restart WSL troubleshooting unless explicitly requested.

## Engineering rules
- Inspect `git status` before substantial changes.
- Prefer small, reviewable changes and commits.
- Preserve reproducibility: configs, seeds, structured metrics, experiment manifests.
- Expose real dataset fields, tensor shapes, observation/action spaces, losses, and relevant configuration values.
- Do not fabricate benchmark numbers.
- Do not claim planned work as completed.
- Update `RESUME_EVIDENCE.md` only after real implementation and verification.

## Immediate task
Preserve original experiments and present only verified results. Use existing
artifacts for portfolio work. The executed backend is deterministic, not an LLM
agent. Do not claim proven robot-success improvement, world-model training or
hardware deployment. Wait for explicit approval before environment changes or
new training/evaluation.
