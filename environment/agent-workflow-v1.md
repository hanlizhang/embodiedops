# EmbodiedOps bounded diagnosis/repair vertical slice

## Registered experiment

The executable entry point is `scripts/agent_workflow.py`; the preregistered budget and acceptance policy are `environment/agent-workflow-v1.json`. New artifacts are exclusive under `outputs/agent-workflow-v1/run-01`. The user's current authorization supersedes earlier historical deferrals of agent implementation and bounded training. No package changes or commits are included.

Maximum: one 512-update training run, 16 simulator episodes, 600 actions/episode, 90 minutes overall. Worker limits are 420 s/episode, 1,200 s/training and 300 s/reload. Diagnostic scenes are seeds 43–44. Final scenes 120–121 are reserved and gated: a candidate failing diagnosis-stage acceptance never consumes them. A search of saved manifests before execution found no seed120/121 runs. This is local scene separation, not proof of independence from original VLA pretraining data.

Two real cases are supplied as historical inputs: natural-language pretrained failures and residual task-ID failures. Historical H20 contacts support diagnosis only; all new paired robot evaluation uses H50. No synthetic fault is needed for these cases. Historical results are human-discovered evidence, not discoveries retroactively credited to this implementation. The small local protocol is not a reproduction of an official success rate.

## Decision and execution boundaries

The CLI implements `evidence -> decision -> registered tool -> structured result -> updated decision`. The model can select only available registry names; it cannot issue shell commands or change the registered budget. An external HTTPS chat-completions-compatible JSON backend is pluggable through `EMBODIEDOPS_REASONING_URL`, `EMBODIEDOPS_REASONING_MODEL`, and `EMBODIEDOPS_REASONING_KEY`. No credentials are logged. No usable credential was present; the user explicitly chose the **deterministic evidence-rule backend**, which is not an LLM agent.

The rule baseline branches on observation validity, instruction mismatch, remaining failures, coverage evidence, validation improvement and measured robot gates. It is not seed-conditioned. Invalid observations stop repair; a mismatch selects configuration repair; residual failures can select the bounded learning hypothesis; unsuccessful verification stops. The registry includes observation/contract checks, historical instruction-distribution probing, trajectory/contact analysis, versioned configuration repair, native training, matched diagnostic evaluation, held-out evaluation and stopping. Instruction provenance is imported from the human-audited Phase1E report, not independently inferred from unavailable upstream training logs.

All decisions, hypotheses, available choices, tool results, exact subprocess commands, return codes and timings are written to `trace.jsonl`; each case also has a state JSON. Tool result timing includes actual probe work. The fixed-order comparator really executes observation, instruction and trajectory probes for each case. Only offline probe counts/time are compared: it does not duplicate expensive training/evaluation and does not support a claimed counterfactual GPU-cost advantage.

## Repair and verification contracts

Configuration repair creates a new candidate JSON with task-ID conditioning and unchanged pinned weights, processors, H50/chunk50/10-flow-step controller contract. Natural-language baselines remain explicitly separate.

Learning reuses the learning algorithm in `phase2_train_v1.py`: 512 mixed-instruction updates, batch1, native BF16/FP32, saved normalization, frozen VLM, trained action expert/state projection. The only trainer change is an optional shorter generated-cache path for Windows. Episode-stratified sampling visits all 461 training episodes before reuse, with seeded saved ordering. This covers more demonstrations than the 32-sample sanity experiment but is not a full frame epoch or proof of adequate training. Sixty-four validation anchors cover the 52 held-out episodes; paired fixed noise/time makes loss curves comparable. Save at 256 and 512; select minimum mean NL/task-ID validation loss, earliest tie. A separate process reloads model, optimizer, scheduler, processors and fixed validation. Selected loss must beat pretrained before spending simulation budget. Candidates never overwrite historical weights.

New rollout diagnostics are opt-in to the existing runner. They retain its controller/physical guards, record substep contacts and reject all-black source observations before selecting an action. Every replan records source-camera hashes and uses the same declared per-chunk RNG seed in both arms. Physical state, scene metadata and reset camera pixels are checked before candidate inference; existing sparse one-level rasterization tolerance remains explicit. Simulator diagnostics never enter policy inputs.

Acceptance requires strictly more successes, no per-seed success loss, valid matched episodes, no more than 1 degree per-scene regression in total closing progress, no new guard failure, and candidate peak single-contact normal force no greater than 1.25 times baseline plus 25 N. The force threshold is a conservative comparative rule, not a calibrated hardware safety limit. Contact count, EEF path, angles and timing are reported even if gates reject the candidate. These gates are enforced outside the reasoning backend. Final scenes cannot select checkpoints or prompt a tuning retry.

GPU allocated/reserved memory is measured by the existing runners. Summed worker elapsed time is reported as GPU-process hours, an upper bound on active GPU time that includes loading, CPU and simulation work. No energy estimate is made without validated power telemetry.

## Reproduction

From the project root in `embodiedops-robocasa`:

```powershell
python -B -m unittest discover -s tests
python -B -u scripts/agent_workflow.py --output outputs/agent-workflow-v1/run-01 --backend deterministic
```

The output directory is exclusive: choose a new name for a separately authorized repeat. Do not replay into run-01. Native Windows execution is needed for the known sandbox simulator/temp-file limitations. Nine lightweight tests passed outside the sandbox; five test the decision and acceptance boundaries. Source snapshots and registered settings are saved with the run.

The first training launch failed before optimizer creation: Windows rejected a 316-character dataset-cache lock path. LeRobot then attempted its metadata-download fallback, which correctly failed in offline mode. No missing dataset or dependency change was required. The failed `learning-candidate-v1`, `training.log` and `blocked.json` remain intact. A short cache under `run-01/c` reduces that lock path to 246 characters. The recorded operator-assisted recovery uses a fresh `learning-candidate-v2` and retains elapsed time, episode counts, seed matrix and 512-update limit:

```powershell
python -B -u scripts/agent_workflow.py --output outputs/agent-workflow-v1/run-01 --backend deterministic --resume-pretraining-failure
```

This narrow recovery refuses an existing update log/checkpoint or a second recovery marker. It does not resume a partly trained model or re-execute completed robot episodes. Source snapshots before/after the cache fix are retained; the trace does not attribute this filesystem fix to the deterministic decision backend.

## Measured outcome

Completed: **4 new RoboCasa episodes and 512 optimizer updates; both candidates rejected**. Final seeds 120–121 were not consumed. All three protected checkpoint hashes remained unchanged. No overall repair-improvement or generalization claim follows.

Case 1: historical NL failures -> observation check passed -> historical instruction mismatch probe -> versioned task-ID candidate -> fresh matched robot evaluation -> **rejected for door-progress and force regressions**. The scalar success count improved from 0/2 to 1/2, but this did not override the independent acceptance policy.

| Seed / condition | Actual success | Actions | Initial -> final door angle | Loaded-contact actions | Peak single-contact normal force | EEF path | Mean chunk inference |
|---|---:|---:|---|---:|---:|---:|---:|
| 43 / NL baseline | No | 600 | 85.524 -> 86.232° | 186 | 124.70 N | 2.103 m | 0.602 s |
| 43 / task-ID candidate | No | 600 | 85.524 -> 90.000° | 421 | 440.96 N | 1.968 m | 0.631 s |
| 44 / NL baseline | No | 600 | 83.317 -> 77.021° | 67 | 248.70 N | 1.211 m | 0.587 s |
| 44 / task-ID candidate | Yes | 304 | 83.317 -> 0.203° | 133 | 91.09 N | 1.580 m | 0.636 s |

The candidate on seed43 briefly achieved 9.385° closing relative to reset, then lost it. It finished 3.768° worse than the matched baseline and exceeded the 180.88 N comparison gate. No displacement guard or black-camera stop occurred in these four episodes. Physical reset states matched exactly; camera differences were within the declared sparse rasterization tolerance. Peak CUDA allocation was 971,354,112 bytes (926.36 MiB) per episode. Rollout durations excluding initialization were 60.80, 58.68, 52.12 and 24.91 seconds respectively. These are new trials with the matched per-chunk RNG schedule, not exact reproductions of historical trajectories.

Case 2: historical residual task-ID failures -> valid observations and matching instruction -> physical contact/limited progress plus sparse prior local-training coverage -> exploratory mixed-instruction learning candidate -> checkpoint reload -> **rejected for validation regression**. This does not prove broader training coverage cannot help; this specific update budget/settings did not improve the fixed held-out objective. No new robot success rate is claimed for the trained checkpoint, because it did not qualify for simulation.

| Updates | Fixed training NL loss | Held-out NL loss | Held-out task-ID loss |
|---:|---:|---:|---:|
| 0 | 0.379791 | 0.405792 | 0.363459 |
| 128 | 0.518204 | 0.566535 | 0.555070 |
| 256 | 0.433088 | 0.502397 | 0.489942 |
| 384 | 0.377147 | 0.436887 | 0.432628 |
| 512 | 0.373707 | 0.430901 | 0.424827 |

Actual coverage was **512 distinct frame anchors from all 461 training episodes**; the first 461 updates visited each once. Validation uses 64 anchors from 52 disjoint episodes. All 512 updates had finite gradients on all 155 intended tensors. Logged pre-clipping gradient norms ranged 1.708–33.588; native clipping remains 10. Of the trainable tensors, 122 changed and 33 remained byte-identical; all frozen tensors remained unchanged. Fixed training loss fell only 1.60%; selected validation mean worsened 0.384626 -> 0.427864 (11.24%). Native BF16 rounding remains a limitation, not a newly repaired mechanism.

The selected new checkpoint is `learning-candidate-v2/checkpoints/000512/pretrained_model`, SHA256 `46553a328519192dae29705d635f78efe5920b6d2d758f3faf81f353f7b0cdc2`. The step256 checkpoint is retained too. Separate-process reload verified all 500 model tensors, optimizer, scheduler, processors and both fixed validation vectors. Training-loop duration including probes/checkpoints was **398.25 s**; peak CUDA allocated/reserved were **1.732/1.846 GiB**.

End-to-end elapsed time including operator recovery was **1,103.11 s (18.39 min)**. Summed GPU-worker process time was **0.264935 hours**, including initialization, CPU/I/O and the failed pre-update launch; actual active GPU-hours were not measured. Energy is unavailable. The trace records 12 deterministic decisions, including the repeated training choice after the explicit operator recovery.

Adaptive diagnosis executed **5 offline probes / 2.093 s** versus the fixed-order comparator's **6 / 1.688 s**. The comparator ran second with warm caches. This demonstrates one avoided probe, **not a wall-time speedup**, nor a measured GPU-cost advantage against an independently trained fixed-order system.

## Audit, review and limitations

`audit_run.py` and `audit.json` in the run directory verify update counts, coverage/splits, finite gradients, reload, original weight hashes and the robot acceptance decision. `losses.svg` contains the actual learning curves. `trace.jsonl`, `*-state.json`, `*-candidate.json`, `instruction-diagnostic-evaluation.json` and `result.json` form the machine-readable case study. Full worker outputs remain available, including the failed launch.

An offline post-run review hardened two evidence fields without rerunning training or robotics: missing historical H50 contact measurements now serialize as null, and historical training coverage is derived directly from raw update/split files (300 updates, 221/461 episodes), rather than a curated boolean from the Phase2 report. The original trace is preserved: its fixed-order H50 contact zeros were unavailable placeholders accompanied by a warning, **not measured absence**. `postrun-schema-audit.json` records that clarification. All 12 recorded deterministic choices replayed unchanged. New robot evaluations always had measured contacts, so these schema changes do not alter either rejection.

The delivered system is a tested deterministic diagnostic baseline with an unexercised pluggable external-model backend. It is not evidence of LLM reasoning or autonomous infrastructure recovery. The coverage repair is hypothesis-driven but exploratory; no unique physical root cause was established. Two diagnostic seeds and fixed small validation probes cannot demonstrate robustness, and there is no accepted candidate for final evaluation. Force measurements are solver contact normals, not reconstructed hinge torque or certified collision safety. No simulator state was provided to SmolVLA.

Source changes are confined to the workflow CLI, plan, tests, report, opt-in rollout instrumentation, and optional trainer cache argument, plus concise documentation/evidence updates. `code.diff` in the run directory contains the exact task-specific diff; prior uncommitted localization work is excluded and preserved. Nothing was committed automatically.
