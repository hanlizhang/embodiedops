# MuJoCo / 3D Simulation Environment

## Current integrated RoboCasa environment

The learned-policy experiments use **`embodiedops-robocasa`**, with RoboCasa1.0.0,
robosuite1.5.2 and MuJoCo3.3.1. Three-camera observations and the PandaOmron
hybrid mobile/OSC controller are connected to SmolVLA; actual CloseFridge rollouts
and matched evaluations are complete. See [Phase1B](phase1b.md) and
[camera-corrected completion](learning-repair-v1-completion.md).

The remainder records the separate, earlier **`embodiedops-mujoco`** Lift smoke
environment. It is not the runtime used for the VLA results. No new simulator
execution or environment changes are needed for the portfolio demo.

This environment is intentionally separate from the LeRobot policy environment to avoid dependency conflicts.

## Conda environment

```powershell
conda create -n embodiedops-mujoco python=3.10 -y
conda activate embodiedops-mujoco
```

## Dependencies

```powershell
pip install "numpy==1.26.4" "mujoco==3.3.7" "robosuite==1.5.2"
```

Verified:

- MuJoCo: `3.3.7`
- robosuite: `1.5.2`
- NumPy: `1.26.4`
- Panda robot loads successfully
- `Lift` environment runs successfully on Windows

## Simulation smoke test

```powershell
python -m robosuite.demos.demo_random_action
```

Select:

- Environment: `1` — Lift
- Robot: `5` — Panda

The random-action demo validates the simulation stack only. It is not a trained policy and is not expected to solve the Lift task.

## Current scope

This environment is intended for 3D visualization and manipulation-task evaluation. It is not yet connected to the EmbodiedOps policy-training or agent loop.
