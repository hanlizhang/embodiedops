# Terminal-window learning repair — measured negative result

**2026-10-20. Rejected.** One native-precision, 5,000-update SmolVLA candidate
completed training, five independent checkpoint reload checks and four actual
RoboCasa diagnostic episodes. Success remained **1/4**, while the two intended
repair cases regressed substantially. Final seeds **120–123 were not run**.
There was also one runtime-limit enforcement deviation, documented below.

The [compact numerical record](terminal-window-experiment-v1-results.json)
includes fixed curves, paired metrics, selection/reload evidence, source hashes
and the runtime deviation for review without the large local artifacts.
Portfolio delivery review rehashed all 1,122 protected files, reconstructed the
gate from saved episode metrics, and passed all 21 lightweight tests. Two tests
initially hit sandbox temporary-file restrictions and passed outside the sandbox.
No policy inference, training or RoboCasa rollout was run for this review.

Run: `outputs/agent-workflow-v1/terminal-window-v1/`.
The original pretrained, Episode-balanced and FP32 artifacts were preserved:
all **1,122 protected files** matched their starting hashes.

## Controlled data intervention and preflight

The [audited proposal](terminal-window-repair-v1.md) and
[registered configuration](terminal-window-experiment-v1.json) define the experiment.
Initialization, native BF16/FP32 optimizer path, batch 1, masked Flow Matching,
`CloseFridge`, saved processors, AdamW, LR schedule, episode order, update noise
seeds and original validation/checkpoint rule matched archived Episode-balanced.
Only frame-anchor sampling changed. No historical checkpoint initialized the
candidate except the pinned original pretrained weights.

For first positive reward row `tau`, use full chunks starting at
`max(0,tau-49)..min(tau-1,T-50)` in eligible monotone-reward demonstrations.
Replace zero-based even updates using RNG242; retain archived anchors otherwise.
Reward indexes offline sampling; it is not a policy input or auxiliary loss.

| Actual sampling evidence | Episode-balanced | Terminal-window |
|---|---:|---:|
| Updates / training episodes | 5,000 / 461 | 5,000 / 461 |
| Anchors in terminal pool | 274 | 2,566 |
| Distinct selected anchors | 4,891 | 4,614 |
| Padded chunks | 939 | 469 |
| Valid action target rows | 226,916 | 238,655 |

The frozen order matched exactly: **2,438 targeted replacements, 2,500 uniform
anchors, 62 ineligible fallbacks**; 450 eligible training episodes, 7,200 possible
terminal anchors, 2,109 distinct targeted anchors. All episode visits stayed
identical; all 52 validation episodes remained separate. LeRobot's actual reader
passed 4,718 query/padding checks and 17 decoded action/RGB boundary examples.
No diagnostic or test rollout constructed training labels.

These are recorded pre-success windows, **not verified contacts or recovery
demonstrations**. Reader row/timestamp alignment is verified; the export's precise
physical reward offset remains unresolved. The additional 11,739 valid action
targets are a coupled change in supervision coverage, not an isolated phase effect.

## Training, selection and reload evidence

| Fixed task-ID Flow Matching loss | Original pretrained | Native step 5,000 | Targeted step 5,000 |
|---|---:|---:|---:|
| Original 32 training anchors | 0.321402 | 0.316057 | 0.326827 |
| Original 104 global held-out anchors | 0.369275 | 0.367256 | 0.377230 |
| 104 terminal held-out anchors | 0.301864 | 0.307188 | 0.296408 |

The terminal loss improved **3.51%** against native, but global loss worsened
**2.72%**. Early deterioration persisted: targeted global loss reached 0.437464
at update 500, and fixed training loss reached 0.381608 at 1,000.
The native terminal reference was measured anew on identical anchors/draws;
its original global validation was reproduced exactly.

Global task-ID validation among saved steps 1,000–5,000 selected **5,000** before
robot evaluation; terminal loss and robot success did not select it.
Selected model SHA256:
`3246ea6ec205e08530674bb7f91040b0ec6ee919043116971ee2d7724b2c32e3`.

All 5,000 updates had finite gradients; 409 exceeded clipping threshold10
(native: 461). LR values matched every archived update. **122/155 trainable
tensors changed; all frozen tensors stayed identical.** Five fresh processes
verified saved model/optimizer/scheduler contents, byte-identical processors and
original plus terminal fixed validation losses. These checks did not perform
resumed optimizer updates.

See [measured loss curves](../outputs/agent-workflow-v1/terminal-window-v1/analysis/loss-comparison.png),
`candidate/{probes,updates}.jsonl`, `parameter-audit.json`, `reload-*.json`,
and `analysis/loss-curves.json`. The legacy summary's `sanity_loss_reduced`
uses natural-language loss; the table above correctly uses task-ID loss.

## Actual paired manipulation

All comparisons use `CloseFridge`, H=50, chunk50, 10 flow steps, the 600-action
cap and unchanged controller. Resets, camera contracts and selected model hashes
matched; all four episodes had valid observations and no physical guard termination.
Native seed46 uses its valid replacement, excluding the historical blackout.

| Seed | Success native → targeted | Final door opening ° native → targeted | Peak recorded normal N native → targeted | Mean inference ms native → targeted |
|---|---|---|---|---|
| 43 | fail → fail | 13.11 → 90.00 | 405.16 → 575.12 | 637 → 594 |
| 44 | success → success | 0.31 → −0.09 | 171.22 → 152.14 | 711 → 693 |
| 45 | fail → fail | left −0.01 → 40.76; right 28.27 → 31.09 | 188.69 → 500.35 | 619 → 608 |
| 46 | fail → fail | 85.39 → 87.84 | 231.66 → 534.93 | 744 → 631 |

Pretrained, native and targeted each succeeded only on seed44: **1/4**.
Seed44's targeted success took 251 actions versus native220; lower native-relative
force there did not compensate for other regressions.

Recorded geometry/solver evidence explains the failure outcomes more directly
than terminal loss:

- **43:** best opening was still 72.75° at action465, followed by reopening to90°.
  Loaded door contacts occurred at66–580, with gaps; more contact did not produce
  effective closure. Peak door normal force was520.35N; the575.12N overall peak
  was robot link5/link7 self-contact.
- **45:** right-door loaded contact ended at233, leaving about31.09° open.
  Its final <0.1° variation plateau lasted18.75s while the EEF travelled another
  2.105m. Left-door contact began at538; it was still closing at the600-action cap,
  ending40.76° open. A hand/left-door contact reached500.35N.
- **46:** only1.83° best progress, then reopening; the534.93N peak was
  link5/hand self-contact.

These are measured contact-local normal forces, not proximity estimates, world
wrenches or inferred hinge torques. See [door traces](../outputs/agent-workflow-v1/terminal-window-v1/analysis/door-progress.png)
and `analysis/analysis.json`. New videos:
[43](../outputs/agent-workflow-v1/terminal-window-v1/diagnostic-43/rollout.mp4),
[44](../outputs/agent-workflow-v1/terminal-window-v1/diagnostic-44/rollout.mp4),
[45](../outputs/agent-workflow-v1/terminal-window-v1/diagnostic-45/rollout.mp4),
[46](../outputs/agent-workflow-v1/terminal-window-v1/diagnostic-46/rollout.mp4).

The preregistered gate rejected the candidate for no success gain, neither target
case repaired, individual/summed door regressions, force regressions against
both references, and worse global validation. Native-relative force limits failed
on43/45/46; pretrained-relative force limits failed on43/44/45/46.
No final evaluation was authorized by the gate.

## Cost and protocol deviation

Training worker: **109.02min / 1.817 GPU-worker hours**; loop including probes and
saves107.42min. Peak CUDA allocation **1.732GiB**, reserved1.846GiB.
Archived native worker:48.40min. These observed durations include CPU/I/O and
different runtime conditions; they do not isolate sampling-method efficiency.
The whole recorded workflow took165.15min before the final preservation audit;
GPU-worker wall time2.726h. Active GPU time and energy were not measured.
The four episodes executed2,051 actions and42 inference calls, averaging0.623s.

**Reload4000 exceeded its registered900s cap: 1,414.422s (23.57min).**
The saved start/result timestamps confirm the elapsed interval. Content checks
passed, but the subprocess timeout did not enforce the wall limit. Host suspension
or scheduling delay is a hypothesis, not a verified cause. Subsequent workers
continued, so this run did not satisfy every runtime bound.

`protocol-audit.json` records the deviation without changing the original trace
or results. A **post-run** source guard now rejects successful exits whose measured
duration exceeds the limit, preventing subsequent workers after such an overrun.
It was not applied retroactively and no experiment was rerun. The exact executed
runner remains saved as `terminal_window_experiment.py.txt`.
Twenty lightweight tests passed before execution; all four targeted tests passed
after adding the deadline regression case. No dependencies changed or commits made.

## Interpretation and reproduction

Emphasizing successful terminal actions improved the corresponding held-out
imitation loss, **but did not improve measured manipulation**. It degraded door
progress and contact forces in the intended repair cases. Success-adjacent
demonstrations did not establish recovery coverage for these policy-generated
states. This one matched training seed and four diagnostic scenes do not establish
generalization or an official benchmark result; pretraining overlap is unknown.

Actual execution command (existing destinations are deliberately refused):

```powershell
& 'D:/Apps/miniconda/envs/embodiedops-robocasa/python.exe' -B -u scripts/terminal_window_experiment.py --output outputs/agent-workflow-v1/terminal-window-v1 --dataset-cache outputs/twc-v1
& 'D:/Apps/miniconda/envs/embodiedops-robocasa/python.exe' -B scripts/terminal_window_summary.py --run outputs/agent-workflow-v1/terminal-window-v1 --output outputs/agent-workflow-v1/terminal-window-v1/analysis
```

Reproduction requires fresh output/cache paths and separate execution approval.
`trace.jsonl` contains every actual subprocess command; source snapshots, settings,
sample manifests, checkpoint hashes and gate results remain in the run directory.
The Python summary produces SVGs; the PNGs are rasterizations using the desktop's
already-bundled Sharp library, without changing the training environment.
