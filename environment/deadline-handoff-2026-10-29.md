# Deadline handoff — 2026-10-29

GitHub synchronization index, checked 2026-10-30. The filename retains the requested handoff date. Repository paths below are relative to the project root; Windows paths identify existing local-only artifacts.

Latest outcome: Recontact Self-Imitation v2 completed its fixed 300-update mixture training and all 36 physical conditions. Qualified full-task successes were frozen 1/6, ordinary_300 0/6, recontact_mix_300 0/6; qualified held-out recovery successes were 6/6, 3/6, 3/6. There is no qualified success gain, and the mixture loses frozen's seed 64 success. The current data plus mixture recipe is closed. Real Nemotron-driven closed-loop execution has **not** been completed; the executed Agent backend is deterministic. The original v1 stop and all invalid attempts remain preserved.

## Reports and committed evidence

- Main report: `environment/recontact-self-imitation-v2.md` (includes 36 final conditions, all 50 attempts, limitations and delivery-time overrun).
- Earlier reports: `environment/recontact-self-imitation-v1.md`, `environment/recovery-feasibility-v1.md`, `environment/recovery-feasibility-v1b.md`, `environment/recovery-completion-a48-v1.md`.
- Compact evidence entry: `docs/evidence/recontact-self-imitation-v2/README.md`.
- Fixed plan and split: `docs/evidence/recontact-self-imitation-v2/registered-plan.json`, `docs/evidence/recontact-self-imitation-v2/data-split.json`.
- Source index: `docs/evidence/recontact-self-imitation-v2/source-index-summary.json`.
- Final results, 36 conditions, 50 attempts, training exposure and reload evidence: `docs/evidence/recontact-self-imitation-v2/result.json`.
- Model identities: `docs/evidence/recontact-self-imitation-v2/frozen-models.json`.
- Provenance and stage times: `docs/evidence/recontact-self-imitation-v2/evidence-index.json`; source repairs: `docs/evidence/recontact-self-imitation-v2/runtime-source-amendments.jsonl`.
- Complete original local evidence: `D:\EmbodiedOpsOutputs\recontact-self-imitation-v2`.

## Existing Agent and evaluation interfaces

`scripts/agent_workflow.py` contains `DeterministicBackend`, the generic HTTPS chat-completions `EndpointBackend`, `Workflow.registry`, `Workflow.evaluate()` and the independent `acceptance()` gate. The registry names are `observation_contract`, `instruction_probe`, `trajectory_contact`, `configuration_repair`, `policy_training`, `evaluate_diagnostic`, `evaluate_final`, `stop`. The checked-in plan is `environment/agent-workflow-v1.json`; the executed workflow report is `environment/agent-workflow-v1.md`; relevant existing tests are `tests/test_agent_workflow.py`.

Existing invocation syntax, recorded for handoff only (not executed during this synchronization):

```powershell
conda activate embodiedops-robocasa
python -B -u scripts/agent_workflow.py --output outputs/agent-workflow-v1/<fresh-run-name> --plan environment/agent-workflow-v1.json --backend deterministic
# Generic external-model adapter: same command with --backend endpoint.
```

A fresh output directory is required. This research CLI can invoke training and simulation through its bounded tools; it is not a read-only Demo launcher. Rollout/physical diagnostics use `scripts/phase1d_rollout.py` and `scripts/control_diagnostics.py`; training/reload uses `scripts/phase2_train_v1.py`. The completed v2 experiment uses `scripts/recontact_v2.py`, `scripts/recontact_v2_worker.py`, `scripts/recontact_v2_train.py`, `scripts/recontact_v2_evidence.py`, and `scripts/recontact_v2_report.py`; its experiment-specific acceptance is reported separately from the older Agent plan.

For a future Nemotron endpoint connection, the current adapter reads exactly these environment-variable names: `EMBODIEDOPS_REASONING_URL`, `EMBODIEDOPS_REASONING_MODEL`, `EMBODIEDOPS_REASONING_KEY`. There is no implemented `NVIDIA_API_KEY` fallback. This handoff records no credential values and does not claim endpoint compatibility or a completed real-model tool loop.

## Local inference checkpoints

All three `model.safetensors` files exist. Hashes below are the recorded SHA-256 identities from `frozen-models.json`; weights remain local and are not committed.

| Model | Existing local directory | model.safetensors SHA-256 |
|---|---|---|
| frozen | `C:\Users\Alan\Projects\embodiedops\outputs\phase1a\hf-cache\hub\models--lerobot--smolvla_robocasa\snapshots\f7af588d11e30b0069f200792fb285904b74c80b` | `96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b` |
| ordinary_300 | `C:\Users\Alan\Projects\embodiedops\outputs\agent-workflow-v1\instruction-control-v1\attempt02\evaluation\inference-C` | `e3cc9a2084f081778b2353bfc8e8c9d027e933e224eca37205be2d2c9715a3c8` |
| recontact_mix_300 | `D:\EmbodiedOpsOutputs\recontact-self-imitation-v2\training\checkpoints\000300\pretrained_model` | `646ebc87d8a9ad8cdd00c4dc8e01e25e8a52e7a8d6f06c0b01ba36234b6f8cce` |

The base identity is `lerobot/smolvla_robocasa` revision `f7af588d11e30b0069f200792fb285904b74c80b`. Both trained controls use `CloseFridge`; ordinary_300 reuses the matched historical task-ID 300-update inference weights, and recontact_mix_300 is the new fixed step-300 checkpoint. Missing historical ordinary optimizer continuation state does not change its verified inference-control role.

## Demo and exact seed 64 videos

- Existing public Demo source: `docs/demo/index.html`; instructions: `docs/DEMO.md`; hosted URL: https://hanlizhang.github.io/embodiedops/demo/ . This existing presentation contains earlier experiments, not a newly updated v2 success Demo.
- Existing local portfolio page: `C:\Users\Alan\Projects\embodiedops\outputs\portfolio-demo-final-v1\index.html`.
- Complete local v2 video index: `D:\EmbodiedOpsOutputs\recontact-self-imitation-v2\video-index.html`.
- Seed 64 matched baseline, true success at action 439: `D:\EmbodiedOpsOutputs\recontact-self-imitation-v2\full\full-64-r1-frozen\rollout.mp4`.
- Seed 64 matched mixture, valid attempt 02, unsuccessful at action 600: `D:\EmbodiedOpsOutputs\recontact-self-imitation-v2\full\full-64-r1-recontact_mix_300-attempt02\rollout.mp4`.

These videos illustrate a regression, not an improvement. All listed local HTML/video paths were checked to exist; videos, raw images, model tensors and Junction targets are excluded from this Git synchronization.
