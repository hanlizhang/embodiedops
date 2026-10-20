# EmbodiedOps — current project context

Updated 2026-10-20. This file describes current scope; dated experiment reports
retain original stopped-run findings and subsequent completion addenda.

## Current milestone

Portfolio delivery from completed SmolVLA / RoboCasa / CloseFridge evidence.
Further fine-tuning, sampling/precision/horizon sweeps and unrelated diagnostics
are stopped. Do not launch training, policy inference, RoboCasa evaluation or new
model integrations without a new explicit request.

The Terminal-window negative milestone is committed as
`53cc938e834681941e2e53a483c5264f81fb1f21` and pushed to `origin/main`.
Historical experiments and checkpoint weights remain unchanged.

## Implemented system

Pinned SmolVLA action-expert/state-projection Flow Matching fine-tuning is connected
to RoboCasa/MuJoCo closed-loop execution, observation/normalization/controller
checks, instruction probes, physical trajectory/contact measurements and bounded
deterministic repair routing. The system saves versioned candidates, structured
traces, optimizer/checkpoint evidence, matched contracts and explicit decisions.

The deterministic backend is rule-based. A generic endpoint adapter exists but
was never validated or used. Historical discoveries and later learning experiment
designs were human-directed. There is no executed autonomous LLM, world-model
training, hardware deployment, accepted repair or robust success gain.

## Completed learning evidence

All sustained arms start independently from the original pinned checkpoint:

| Arm | Updates | Global task-ID validation | Diagnostic seeds43–46 |
|---|---:|---:|---|
| Pretrained reference | 0 | 0.369275 | 1/4 success |
| Episode-balanced | 5,000 | 0.367256 | 1/4; rejected |
| Motion-mixture | 5,000 | 0.370945 | 0/4; rejected |
| FP32-master | 5,000 | 0.387121 | Not run: independent reload timeout |
| Terminal-window | 5,000 | 0.377230 | 1/4; rejected |

Dataset: 513 episodes / 155,443 frames, with 461 train / 52 validation episodes.
Checkpoint selection uses original fixed global task-ID validation, never robot
success. Terminal loss improved locally but physical closing regressed.
Native and motion arms passed ten independent reloads; terminal passed five
content reloads with a separately recorded runtime-cap overrun. FP32 serialized
hash checks do not substitute for the blocked full reload.

Final seeds 120–123 remain unused because diagnostic gates failed. The raw-camera
fault was localized to renderer readback and reproduced during obsolete-renderer
cleanup; corrected owner-context cleanup enabled the two missing seed 46 episodes.
The invalid attempt remains archived and excluded.

Earlier NL results use different conditioning/protocols: Phase1D 0/10 pretrained
versus 3/10 after 300 updates; Phase1E NL 0/3 versus task-ID 1/3. These are local
observations, not official reproduction or robust generalization. Do not pool them
with the four-scene task-ID matrix.

## Authoritative entry points

- `README.md`: public overview and result table.
- `RESUME_EVIDENCE.md`: dated implementation ledger.
- `docs/TECHNICAL_OVERVIEW.md`: interfaces, objective, sampling and validation.
- `docs/DEMO.md`: authentic media sources and three-minute talk track.
- `docs/INTERVIEW.md`: English/Chinese resume claims and technical answers.
- `environment/agent-workflow-v1.md`: executed deterministic vertical slice.
- `environment/learning-repair-v1.md` and `learning-repair-v1-completion.md`: original
  two-arm experiment and camera-corrected completion.
- `environment/optimizer-precision-v1.md`: closed negative optimizer ablation.
- `environment/terminal-window-experiment-v1.md`: rejected targeted repair.

## Environment and preservation

Canonical workspace: `C:\Users\Alan\Projects\embodiedops`.
Remote: `https://github.com/hanlizhang/embodiedops`.
Main environment: `embodiedops-robocasa` (Python3.12, LeRobot0.6.1,
RoboCasa1.0.0, robosuite1.5.2, MuJoCo3.3.1, PyTorch2.11.0+cu128).
Hardware: Windows11, RTX3060 Laptop6GB. Original `embodiedops` and
`embodiedops-mujoco` environments stay separate.

Git tracks code, configurations and concise evidence. `outputs/` holds original
large artifacts and is ignored. Do not overwrite checkpoints, results or videos.
No dependency changes or expanded experiments are authorized by portfolio work.
