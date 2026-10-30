# EmbodiedOps — current project context

Latest verified delivery (2026-10-31): the [real Nemotron Agent demo](environment/nemotron-agent-demo-v1.md)
completed 8 API requests, 7 native tool calls and one optional fixed two-episode
physical verification selected by the model. The independent gate rejected the
existing seed64 candidate. Both new episodes failed; archived frozen success is
preserved separately. Raw model prose contained a contact-gate error, annotated in
the report and Demo. No training or accepted learned repair occurred. The generic
legacy endpoint adapter remains unused; the new runner uses native tool calling.


Updated 2026-10-23. This file describes current scope; dated experiment reports
retain original stopped-run findings and subsequent completion addenda.

Latest delivery index: [2026-10-29 handoff](environment/deadline-handoff-2026-10-29.md)
(synchronized 2026-10-30). Recontact self-imitation v2 completed training and all
36 fixed physical conditions with no added qualified success; the real Nemotron
closed loop was not yet completed at that handoff. The dated milestones below are preserved.

## Current milestone

The separately authorized **failure-state recovery feasibility pilot is stopped
and incomplete**. Two historical action-prefix/control audits matched physical
and 16D state exactly over 1,150 actions. Four pretrained VLA candidates on the
seed48 failure state produced no fridge contact or success. Seed43's first
branch stopped before inference because its camera mismatch exceeded the frozen
tolerance. The one executed geometry expert had a reversed nearest-point
direction and is invalid as feasibility evidence; a static-tested repair has
not been rerun physically. No learning intervention is yet justified. The
[report](environment/recovery-feasibility-v1.md) preserves invalid/unrun attempts
and a conditional learning proposal. No automatic retries, further evaluation
or training are authorized by this stopped-run handoff.

The instruction-controlled 300-update SmolVLA study is **closed at 52/60 valid
cells**: one timeout-invalid cell and seven unrun cells will not be resumed.
Task-ID success is A 3/9, B 3/9, C 1/8; NL success is A 0/9, B 1/9, C 0/8.
These are incomplete diagnostic results, not a complete benchmark. Neither
fine-tune demonstrates task-ID improvement; the historical NL 0/10 → 3/10 gain
was not reproduced in the completed evidence. Runtime/numerical limitations and
all interrupted attempts are retained. C's step-300 inference validation passed,
but optimizer/scheduler state remains incomplete; no replay is authorized.

Next research milestone: a separately scoped **Corrective Imitation Learning**
design. Failure root cause remains unresolved; missing recovery demonstrations
are a hypothesis, not an established finding. The
[closed study and handoff](environment/instruction-control-v1.md) are authoritative
for this result. No data collection, training, inference, simulation or additional
diagnostics are authorized by this handoff.

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
