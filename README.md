# EmbodiedOps

**An autonomous AI engineer for robot policy evaluation and improvement.**

EmbodiedOps is an open-source robotics / Physical AI project exploring a closed-loop workflow in which an AI agent can evaluate learned robot policies, inspect failures, choose follow-up experiments, launch bounded training or configuration changes, and compare results.

## Current status

Environment and pipeline validation is complete for the first two backends:

- Hugging Face LeRobot + Diffusion Policy + PushT
- MuJoCo + robosuite + Panda Lift

The autonomous agent loop is **not implemented yet**.

## Planned MVP

1. Train and evaluate a Diffusion Policy baseline.
2. Add reproducible robustness / perturbation evaluations.
3. Expose narrow typed tools for evaluation, comparison, and training.
4. Connect an NVIDIA / Nebius reasoning agent to those tools.
5. Let the agent choose follow-up experiments from real rollout metrics.
6. Demonstrate before / after policy behavior in simulation.

## Environments

The policy-training and 3D-simulation stacks intentionally use separate Conda environments because their current dependencies differ.

See:

- `environment/policy.md`
- `environment/simulation.md`

## Evidence policy

`RESUME_EVIDENCE.md` tracks what has actually been implemented and verified. Planned work should not be presented as completed work.
