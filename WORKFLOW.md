# EmbodiedOps workflow

The current deliverable is the completed-evidence portfolio. Research entry points
below are capabilities to inspect, not automatic next steps. Opening this
repository does not authorize another training or evaluation run.

## Review and presentation

1. Read AGENTS, PROJECT_CONTEXT, README, RESUME_EVIDENCE and the policy/simulation
   environment notes; inspect Git status.
2. Verify claims against compact results and original artifacts. Keep invalid
   attempts and runtime deviations visible.
3. Build the local demo with `scripts/build_portfolio_demo.py`. It reads files
   only and requires no GPU or simulator.
4. Run relevant lightweight tests for code changes. Do not turn a documentation
   task into policy evaluation or training.
5. Review exact staged paths and `git diff --cached --check` before an authorized
   commit. Exclude outputs, weights, videos, caches, logs and unrelated work.

## Existing executable research workflow

| Responsibility | Entry point |
|---|---|
| Deterministic evidence/tool routing | `scripts/agent_workflow.py` |
| Matched learned-policy rollout and observation guards | `scripts/phase1d_rollout.py` |
| Flow Matching training / independent reload | `scripts/phase2_train_v1.py` |
| Episode-balanced / motion-proxy sampling | `scripts/learning_sampling.py` |
| Two-arm sustained experiment | `scripts/learning_repair_experiment.py` |
| Camera cleanup / missing-pair continuation | `scripts/renderer_lifecycle.py`, `scripts/learning_repair_resume.py` |
| Reward-window sampling / registered candidate | `scripts/terminal_sampling.py`, `scripts/terminal_window_experiment.py` |
| Physical contact instrumentation | `scripts/control_diagnostics.py` |

Later runners reuse training, rollout and acceptance components; they are not all
experiments autonomously selected by the deterministic loop. Exact commands and
source snapshots remain in each ignored run directory. The two-arm runner's
historical unconditional final loop was superseded by the documented continuation
gate. Do not blindly execute an archived command.

## Controls for any separately approved future run

Use exclusive output/cache paths, pinned revisions and saved processors. Declare
training/episode/runtime bounds and seeds before execution. Store sample order,
noise seeds, losses, gradients, parameter hashes, costs and checkpoint rules.
Separate integrity, offline validation and physical success: research-only
rollouts do not promote a candidate.

Require matched reset, instruction, cameras, controller and sampling settings.
Keep privileged contact/door diagnostics out of policy inputs. Stop on invalid
observations and preserve the attempt. Never select checkpoints or scenes by
final-test results. Report negative and partial outcomes.

## Environment and Git

Use `embodiedops-robocasa` for integrated VLA work. The older `embodiedops` PushT
and `embodiedops-mujoco` Lift environments remain separate. Installation, upgrades
and environment changes require explicit approval.

Git is the code/documentation source of truth. No automatic commits, pushes or
artifact publication without task authorization. Historical measurements are
immutable; add a labeled audit or addendum when needed. Public clones intentionally
omit model weights, simulator assets and local media.
