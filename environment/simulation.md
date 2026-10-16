# MuJoCo / 3D Simulation Environment

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
