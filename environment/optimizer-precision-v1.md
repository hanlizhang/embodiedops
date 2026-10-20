# Controlled FP32-master experiment — 2026-10-20

**The single approved run completed 5,000 updates. FP32 accumulation changed
actual BF16 forward weights, but worsened the early loss spike and final fixed
losses. Post-training verification timed out before robot evaluation. Robot
improvement is unmeasured; the candidate is not promoted.**

This investigation is closed by the user's 2026-10-20 decision. No follow-up
precision variant or reload retry is planned. The concise repository evidence is
`environment/optimizer-precision-v1-results.json`; raw artifacts remain unchanged.

Artifacts: `outputs/agent-workflow-v1/optimizer-precision-v1/` (abbreviated `OUT`
below). The original run, checkpoints, logs and timeout evidence remain in place.
No further training, retry, variant or simulation was executed after the blocker.

## Controls and numerical correctness

The archived baseline is `outputs/agent-workflow-v1/learning-5000-v1/episode_balanced`.
Both runs independently initialized from SmolVLA revision
`f7af588d11e30b0069f200792fb285904b74c80b`, model SHA256
`96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b`.
Dataset revision: `b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4`.

Both use batch 1, task `CloseFridge`, the same 461 training episodes, all 5,000
sample/order and LR records, 4,891 distinct anchors, and episode-disjoint
validation (104 anchors from 52 episodes). Training noise seeds are
`100000 + zero-based update`; fixed train/validation seeds are `72000+i`/`82000+i`.
Initial model hashes/dtypes, fixed probe noise/time draws and initial losses match.
The architecture, normalization and four saved processor files are unchanged.

AdamW settings remain beta=(0.9,0.95), epsilon=1e-8, weight decay=1e-10, global
gradient clip=10; LR warms up over 250 steps to 5e-5 and decays to 5e-6 over 5,000.
`FP32Masters` parametrizes 129 originally BF16 trainable tensors with FP32 leaves
and differentiable casts to BF16 for forward computation. The other 26 trainable
FP32 tensors and 345 frozen tensors retain their precision. Optimizer parameter
identity is preserved. All 155 trainable gradients and Adam moments are FP32.
This intervention also changes gradient accumulation and clipping-norm arithmetic;
it does not isolate master storage from moment precision.

The disposable preflight saved step 8, then independently reloaded all 500 model
tensors, masters, optimizer, scheduler and processors, reproduced fixed validation,
and reproduced the exact next optimizer update. All 5,000 sustained updates had
finite gradients. All 17 lightweight tests passed outside the Windows sandbox;
the sandbox attempt had two temporary-directory permission errors.

At step 5,000:

- All 33 normalization tensors unchanged in the archived run changed in both
  master and BF16 forward representations. All 129 BF16 trainable tensors changed.
- The monitored first input norm accumulated master changes on every update;
  its BF16 forward value first changed at update 1,084, and changed on 124 updates.
  Six of its 720 forward elements differ from initialization at step 5,000.
- Total trainable forward-weight L2 displacement is **14.7792**, versus **6.22065**
  for native precision. All frozen tensors remain unchanged.
- After eight preflight updates, those 33 norms had changed only in master space;
  this was not claimed as a forward-weight improvement. Other forward tensors
  already changed, and the sustained run subsequently crossed norm quantization
  boundaries.

Sources: `initial-contract.json`, `updates.jsonl`, `parameter-audit.json` under
`OUT/candidate`; `OUT/preflight-passed.json`, `parameter-comparison.json` and
`preflight-parameter-comparison.json`.

## Measured optimization result

These are fixed task-ID Flow Matching losses, not task-success estimates.

| Update | Native train | Master train | Native validation | Master validation |
|---:|---:|---:|---:|---:|
| 0 | 0.321402 | 0.321402 | 0.369275 | 0.369275 |
| 250 | unavailable | 0.551135 | unavailable | 0.610198 |
| 500 | 0.368924 | 0.518811 | 0.433147 | 0.589431 |
| 1,000 | 0.372998 | 0.509189 | 0.425926 | 0.545173 |
| 2,000 | 0.347859 | 0.413688 | 0.393211 | 0.449940 |
| 3,000 | 0.333426 | 0.388100 | 0.382392 | 0.437211 |
| 4,000 | 0.320272 | 0.350799 | 0.370705 | 0.398120 |
| 5,000 | 0.316057 | 0.335195 | 0.367256 | 0.387121 |

The preregistered test required at least a 50% reduction in maximum validation
excess above initialization at the common early probes (500 and 1,000).
Instead, excess increased **3.447-fold**. Master loss was worse on 99/104 held-out
anchors at step 500 and 78/104 at step 5,000. These anchors are not 104 independent
episodes. No matched native probe exists at 250, so the orange peak there cannot
be compared with an observed native value.

Unclipped gradient norm median/max: native **4.794/71.198**, master **5.038/43.458**.
Clip threshold exceeded on **461/5,000** versus **489/5,000** updates.
Loss recovered substantially after the spike, but remained worse than initialization
at step 5,000. Removing update loss was numerically effective and did not resolve
the optimization deterioration under this fixed schedule. Larger effective parameter
movement may explain part of the deterioration; causation by a specific LR, layer
or optimizer component is not established by this single paired run.

All measured probes are in `OUT/audited-summary.json`; plot:
`OUT/precision-loss-comparison.png`. No interpolation is treated as a measurement.

## Selection, saved-state integrity and the blocker

The rule selected **step 5,000**: lowest task-ID validation loss among saved
steps 1,000/2,000/3,000/4,000/5,000, earliest tie. No robot outcomes were used.
Selected forward SHA256:
`4c26a75e89a69fd0d0a2c03ddcec948ac7e7b6ef90a429f8673346a094c9ef51`.
Selected master SHA256:
`9d0c76ee0ec8c86bed84b2ccca1432220e57063d70839b2fdc7e629b0660800d`.

All five checkpoints passed a read-only CPU serialized-file audit against their
save-time digests: 500 model tensors, 129 masters, 465 optimizer tensors,
optimizer groups, scheduler and original processor bytes. All selected masters
cast exactly to their exported native forward tensors.
**This is not an independent full policy reload or validation reproduction.**

The first sustained-checkpoint reload (`reload-1000`) exceeded the registered
**300-second** limit. Its last log output is “Loading weights from local directory”;
there is no worker assertion or completed verification result. The exact internal
stall location and cause are unresolved. Steps 2,000–5,000 were not attempted.
The runner stopped according to its no-retry rule. `blocked.json`, `trace.jsonl`
and `reload-1000.log` preserve the evidence; no successful `result.json` exists.
Preflight reload passed; sustained-checkpoint operational reload remains incomplete.

## Robot reference outcomes and decision

The fixed diagnostic matrix remains seeds 43–46, `CloseFridge`, H=50, chunk=50,
10 flow steps, maximum 600 actions. Archived pretrained/native manifests match
on instruction, versions, simulator-source hashes, horizon and RNG rule; all
native reset-equality checks passed. Native seed 46 uses the valid replacement in
`learning-5000-resume-v1`; its earlier black-camera attempt is excluded.

| Seed | Pretrained success / final door angles, degrees | Native success / final door angles, degrees | FP32 master |
|---:|---|---|---|
| 43 | No / 88.644 | No / 13.114 | Not run |
| 44 | Yes / 0.196 | Yes / 0.308 | Not run |
| 45 | No / left 85.456, right 11.398 | No / left -0.011, right 28.270 | Not run |
| 46 | No / 89.995 | No / 85.393 | Not run |

Archived success is **1/4 versus 1/4**; new success is **unmeasured**, not 0/4.
Both archived groups have zero guard failures. Native peak contact normal force
in seed 44 rose from 88.552 N to 171.222 N; this historical regression is not an
FP32 result. Mean per-episode mean inference latency is 0.641 s pretrained and
0.678 s native. Full door progress, loaded-contact steps, self/external contact
distinctions and trajectories are in `OUT/audited-episodes.json`.

No new success, door motion, contact, safety or inference measurements exist.
The robot acceptance gate was **not run**. Operational verification is blocked,
and validation regression independently prevents promotion. No robot improvement
or robot regression can be attributed to this candidate. Final seeds 120–123
remain untouched.

## Resource use and preservation

| Measured quantity | Archived native | FP32 master |
|---|---:|---:|
| Training loop, including probes/checkpoint writes | 47.46 min | 131.37 min |
| Synchronized update-call total | 28.55 min | 81.98 min |
| Peak CUDA allocated / reserved | 1.732 / 1.846 GiB | 2.648 / 2.773 GiB |

The master training worker took **132.70 minutes (2.212 worker-hours)** including
startup; preflight, training and the timed-out reload together took approximately
**143.91 minutes (2.399 wall-hours)**. Worker-hours include CPU/I/O/waiting and are
not active GPU kernel hours. Sparse GPU observations show changing P-states and
memory clocks; runtime differences cannot be assigned solely to optimizer precision.
Power was not sampled adequately for integration, so no energy estimate is reported.

Only the authorized runtime caps changed: training 90→150 min and main workflow
150→210 min. `registered-plan.json` and `runtime-amendment.json` retain both records.
The running training worker was not restarted or reconfigured. The 300 s reload
cap was unchanged. All **957 protected historical files** remain hash-identical.
No dependency, checkpoint or historical report was modified; no commit was made.

## Code, tests and reproducibility

Implementation: `scripts/fp32_master.py`, optional master branch in
`scripts/phase2_train_v1.py`, bounded runner `scripts/optimizer_precision_experiment.py`,
offline auditor `scripts/optimizer_precision_summary.py`,
`tests/test_fp32_master.py`, and `environment/optimizer-precision-v1.json`.
Executed source snapshots, original configuration and the runtime handoff source
are preserved in `OUT`. `OUT/source-diff.patch` contains this experiment's code and
documentation changes, excluding the pre-existing learning-followup audit/report.

Executed launch (do not rerun into the existing directory):

```powershell
& 'D:/Apps/miniconda/envs/embodiedops-robocasa/python.exe' -B -u scripts/optimizer_precision_experiment.py --output outputs/agent-workflow-v1/optimizer-precision-v1
```

The exact worker commands, including the timed-out independent reload, are in
`OUT/trace.jsonl`. Offline reproduction and lightweight tests require no training
or physics:

```powershell
& 'D:/Apps/miniconda/envs/embodiedops-robocasa/python.exe' -B scripts/optimizer_precision_summary.py --output outputs/agent-workflow-v1/optimizer-precision-v1 --allow-incomplete
& 'D:/Apps/miniconda/envs/embodiedops-robocasa/python.exe' -B -m unittest discover -s tests -v
```

The three separate conclusions are: **numerical accumulation verified;
optimization improvement failed; robot-task improvement unresolved**. Work stops
at the registered verification blocker, with all negative and positive evidence
preserved.
