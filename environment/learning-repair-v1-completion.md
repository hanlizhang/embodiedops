# Learning repair continuation — 2026-10-20

This addendum preserves the [original training and partial evaluation report](learning-repair-v1.md).
No new training, checkpoint selection, task, seed, dependency change or commit.
New evidence is in `outputs/agent-workflow-v1/learning-5000-camera-v1` and
`outputs/agent-workflow-v1/learning-5000-resume-v1`.

## Camera fault: demonstrated mechanism and limits

The invalid episode-balanced seed46 attempt had valid reset images, identical to
pretrained seed46, then three zero-valued raw images after action1. The first
action moved the EEF approximately 0.3 mm; physical state remained finite and
MuJoCo warning counters were zero. The guard stopped before processing the black
observation. This was not a video-codec fault.

The pinned robosuite renderer creates a new GLFW context on hard reset. Its
`MjRenderContext.__del__` calls `self.con.free()` without making the obsolete
renderer's context current. The live renderer can therefore be current while
obsolete GPU resources are freed. Merely having the correct current context at
render time does not establish that its resources remain intact.

Bounded tests used the saved seed46 action, no model inference, three fresh
processes and two total commanded simulator steps:

| Probe | Result |
|---|---|
| Original cleanup + forced garbage collection after valid reset | 12/33 readbacks black; first zero at MuJoCo readback, before physics |
| Render-time rebinding probe | Stopped at exact-reset-image assertion before action; inconclusive. Cleanup logging also encountered a partially destroyed sim at shutdown. Retained as failed test evidence |
| Owner-context cleanup + forced garbage collection | 0/33 black readbacks; obsolete context destruction recorded; live context restored; exact reset images and exact saved first-step qpos; no physics warnings |

`scripts/renderer_lifecycle.py` binds the obsolete renderer's owning context
before its original destructor, then restores the previously current context.
The rollout adapter enables this explicitly with `--renderer-owner-cleanup` and
records its source hash. Installed dependencies are untouched. No image is
substituted, cached as a fallback, or retried after an invalid observation.

The controlled garbage-collection failure and correction establish a renderer
resource-lifetime defect. The original episode did not log destruction events:
its exact triggering allocation/collection remains inferred. Specific OpenGL
resource IDs were not instrumented. The historical seed42 H20 blackout is
consistent with this mechanism but not retrospectively proven to share it.
Its horizon comparison remains invalid; no historical contact or camera data
was reconstructed.

The probe records wall timestamps, simulation time, current/target context
identity, camera IDs (left1/right2/wrist5), shapes, dtype, min/max and nonzero
counts at MuJoCo readback, RoboCasa raw observation, RoboCasa Gym mapping,
LeRobot formatting, tensor bridging and saved policy preprocessing. See
`owner-cleanup/trace.json` and `unmodified/trace.json`.
Original readback becomes zero first; later formatting carries those zeros.
Preprocessing of black data in this probe was diagnostic only, without inference.

After the corrected first step, all cameras were uint8 `[256,256,3]`, with
65,536 nonzero pixels each: left range 7–255, right 3–255, wrist 26–255.
The tensor bridge uses float32 `[1,3,256,256]`; processor outputs and timestamps
are retained in the trace. This correction preserves rendered reset content and
the saved action's physics, rather than changing the benchmark observation.

## Continuation contract

The invalid attempt is archived **in place**, with a complete hash index in
`learning-5000-resume-v1/invalid-attempt-archive.json`; it is excluded from the
success denominator. All files under the original experiment root are hashed
before and after continuation. Completed episodes are reused.

Only diagnostic seed46 episode-balanced and motion-mixture were authorized for
continuation: maximum two policy episodes, 600 actions each, 420 seconds per
worker, 1,200 seconds total, zero optimizer updates. Both use the previously
selected 5,000-step checkpoint, unchanged processors, task ID `CloseFridge`,
H50/chunk50/10 flow steps, same controller and per-chunk random seed rule.
Reset qpos/qvel/state, scene metadata and all cameras are compared to the valid
pretrained46 reference. The all-black camera and physical-motion guards remain.

The latest user instruction requires the diagnostic acceptance gate before
final seeds120–123. This is recorded explicitly in `continuation-plan.json`;
it supersedes the sustained runner's unconditional final-scene loop. Neither
candidate may bypass diagnostic regressions merely because training completed.
No checkpoint was selected using diagnostic or final robot success.

## Training evidence retained

Both arms independently started from pinned pretrained SmolVLA. Each ran 5,000
native masked flow-matching updates over 461 training episodes with 52 disjoint
validation episodes/104 fixed anchors. Episode ordering, optimizer, noise,
architecture and saved normalization were matched. The motion-mixture arm
oversamples inferred quiet/commanded-low-motion regimes; these are **not**
verified contact or semantic-phase labels.

| Metric | Episode-balanced | Motion mixture |
|---|---:|---:|
| Distinct sampled anchors | 4,891 | 4,735 |
| Fixed training loss, initial0.321402 | 0.316057 | 0.315527 |
| Held-out task-ID loss, initial0.369275 | 0.367256 | 0.370945 |
| Selected step | 5,000 | 5,000 |
| Changed trainable tensors | 122/155 | 122/155 |
| Median / max gradient norm | 4.794 /71.198 | 4.960 /44.720 |
| Training-loop minutes | 47.464 | 49.403 |
| Training-worker minutes | 48.401 | 50.402 |
| Peak allocated / reserved GiB | 1.732 /1.846 | 1.732 /1.846 |

All five predeclared saved checkpoints per arm passed weight/processor reload
checks. Selection minimized held-out task-ID loss among steps1000–5000, with
earliest-step tie-breaking; both selected5000 before robot evaluation.
All 155 trainable gradients were finite at every update; 33 trainable tensors
remained byte-identical under native precision, and frozen tensors were unchanged.
The full hashes, learning-rate/loss curves, optimizer settings and reload records
remain in the original audit. Neither training-loss reduction nor a small
validation improvement alone establishes manipulation improvement.

![Measured fixed-probe loss curves](../outputs/agent-workflow-v1/learning-5000-v1/loss-comparison.png)

## Completed diagnostic results and decision

Twelve valid episodes cover every registered diagnostic pair. Ten existing
episodes were preserved, two missing pairs were executed, and one invalid
attempt was excluded. Both new episodes reached 600 actions, with finite policy
inference, valid cameras, no physical guard failures, exact paired reset
qpos/qvel/state and exact reset images. The replacement episode-balanced run's
first action and achieved first-step qpos exactly match the invalid attempt.
All **808 files** in the original experiment directory passed post-run hashes.

| Seed / scene | Pretrained success; closing° | Episode-balanced success; closing° | Motion-mixture success; closing° |
|---|---|---|---|
| 43 / single door | No; −3.12 | No; 72.41 | No; 68.20 |
| 44 / single door | Yes; 83.12 | Yes; 83.01 | No; 18.15 |
| 45 / two doors | No; 71.07 | No; 139.66 | No; −7.47 |
| 46 / single door | No; −3.58 | No; 1.03 | No; −2.93 |
| Success total | **1/4** | **1/4** | **0/4** |

Closing is initial minus final joint angle, summed across doors; negative means
opening. Two-door sums are not interchangeable with single-door progress.
Per-door final angles, best progress, contact-loaded action counts, EEF path
lengths and source paths are in `paired-results.csv`. All success values use
RoboCasa's actual predicate.

| Cost / contact metric | Pretrained | Episode-balanced | Motion mixture |
|---|---:|---:|---:|
| Mean summed closing° | 36.87 | 74.03 | 18.99 |
| Peak recorded normal force N | 428.04 | 405.16 | 633.45 |
| Mean inference seconds / chunk, weighted by calls | 0.637 | 0.672 | 0.664 |
| Inference calls | 43 | 41 | 48 |
| Total rollout seconds, excluding loading | 301.77 | 312.62 | 342.97 |
| Peak torch CUDA allocated bytes | 971,354,112 | 971,354,112 | 971,354,112 |
| Physical guard failures | 0 | 0 | 0 |

**Episode-balanced rejected:** no success gain and a seed 44 force regression
(171.22 N versus 88.55 N, exceeding the fixed 1.25× +25 N gate). Its peak at
action 152 is hand/door-handle contact. A lower aggregate maximum across seeds
does not cancel that per-seed regression. Improved partial closure in seeds 43/45
and faster seed 44 success (220 versus 304 actions) are measured changes,
but do not satisfy acceptance.

**Motion-mixture rejected:** no success gain, loss of seed 44 success, door-progress
and contact-force regressions, plus held-out task-ID loss regression. Its seed 43
633.45 N peak at action 200 is robot link5/link7 self-contact. Motion-proxy sampling
did not yield an effective repair on this diagnostic set.

**Final seeds 120–123 remain unexecuted and separate. Neither candidate qualified.**
The learning-and-diagnostic-evaluation workflow is technically complete under
the latest gate instruction; there is no demonstrated robot-success improvement
and no final-set generalization claim. This evaluates the training-distribution
`CloseFridge` baseline, not the historical natural-language baseline or an
official benchmark success-rate reproduction.

Continuation worker time was 268.625 s (0.07462 worker-hours), giving 2.34330
combined worker-hours with the original experiment, excluding camera probes.
The two training workers account for 1.64671 hours; synchronized optimizer
timing totals 0.96696 hours but includes host overhead and is not a GPU-kernel
utilization integral. Neither active GPU-hours nor energy was measured.
The machine-readable gate, all paired metrics, archive index and preservation
proof are in `result.json`, `completion-audit.json`, `trace.jsonl`,
`invalid-attempt-archive.json` and `preservation-after.json` under the continuation.

## Reproduction and evidence

Use the existing `embodiedops-robocasa` Python. Executed worker commands and
source snapshots are retained in each continuation directory. The following
commands create exclusive new directories; they do not resume by overwriting:

```powershell
python -B -u scripts/camera_lifecycle_probe.py --output outputs/agent-workflow-v1/NEW-camera --collect --safe-free
python -B -m unittest discover -s tests
python -B -u scripts/learning_repair_resume.py --source outputs/agent-workflow-v1/learning-5000-v1 --output outputs/agent-workflow-v1/NEW-continuation --camera-evidence outputs/agent-workflow-v1/learning-5000-camera-v1
# Offline consolidation of the completed continuation; no simulation or inference:
python -B scripts/learning_repair_completion.py --source outputs/agent-workflow-v1/learning-5000-v1 --continuation outputs/agent-workflow-v1/learning-5000-resume-v1
```

All15 lightweight tests passed outside the sandbox; its temporary-file
permissions initially blocked two contact-recorder tests. Three new tests cover
cleanup ordering, restoring the other context, and exceptional cleanup. Native
MuJoCo reproduction supplies the necessary real-renderer evidence beyond mocks.

Remaining limitations: four diagnostic scenes are a small development set,
pretraining overlap is unknown, and final scenes are not evidence unless the
gate permits their execution. Contact force is the maximum recorded positive
contact-frame normal component, including self-contact; it is not net EEF force,
hinge torque, or a certified safety threshold. Worker wall-hours include CPU/I/O;
active GPU-hours and energy remain unmeasured.
