# Phase 1D portfolio text — proposed for review

These are proposed additions, not edits to README.md or RESUME_EVIDENCE.md.
The authoritative measured report is [phase1d.md](phase1d.md), with compact
[results](phase1d-results.json) and [audit](phase1d-audit.json) snapshots.

## README-ready result section

### Measured SmolVLA / RoboCasa result

Implemented guarded CloseFridge rollouts, recorded-state MuJoCo replay, and real
SmolVLA action-expert fine-tuning with LeRobot 0.6.1. A 300-update run on an RTX
3060 Laptop (6 GB) used a frozen VLM, batch size 1, and 1.685 GiB peak allocated
CUDA memory. Finite gradients, changed trainable weights, unchanged frozen
weights, and checkpoint save/reload were verified.

On ten paired simulation seeds (42–51), actual RoboCasa task success was **0/10
pretrained versus 3/10 fine-tuned**. Successes occurred on seeds 44, 47 and 49;
all were scenes with one task-relevant fridge door. Both policies failed all
three two-door scenes. Seed 48 looked nearly closed (5.298°) but officially failed.

This is observed improvement on a small evaluation sample, not established
generalization. The fixed eight-frame validation probe worsened from 0.5318 to
0.5942. Original pretraining overlap with our data and evaluation scenes is unknown.
The diagnosis/repair agent remains planned, not implemented.

Local demo: `outputs/phase1d/comparison.html` (seed 44 selected by default).
The page needs the ignored local videos; cloning Git alone does not provide them.
See `environment/phase1d.md` for pinned revisions, configuration and all seed results.

## Proposed exact RESUME_EVIDENCE.md additions

Pending user review; the existing ledger is unchanged.

### Verified SmolVLA / RoboCasa experiment

- [x] Ran guarded SmolVLA CloseFridge inference in pinned RoboCasa/robosuite/MuJoCo,
  preserving three-camera mapping, 16D state, 12D actions and the 50-action queue.
- [x] Implemented recorded-state MuJoCo replay with scene validation and geometric
  diagnostics; replay loads saved poses without policy inference or physics integration.
- [x] Completed 300 real fine-tuning optimizer updates on an RTX 3060 Laptop 6 GB,
  freezing the VLM and training the action expert and projections (99,880,992 parameters).
  Peak Torch CUDA allocated/reserved memory: 1.685/1.801 GiB.
- [x] Verified finite training losses/gradients, 122 changed intended parameter tensors,
  unchanged frozen tensors, and separate-process checkpoint reload matching all 500
  state tensors and saved fixed-noise validation outputs.
- [x] Completed ten paired CloseFridge seeds: pretrained 0/10, fine-tuned 3/10 using
  the actual RoboCasa success predicate. One-door scenes: 0/7 versus 3/7; two-door
  scenes: 0/3 versus 0/3. Saved continuous three-camera rollout videos and manifests.
- [x] Recorded limitations: validation probe worsened (0.5318 to 0.5942), only one
  short training run, limited seeded evaluation, unknown original pretraining overlap;
  no claim of robust generalization, real-robot transfer or autonomous repair.

Evidence: `environment/phase1d.md`, `environment/phase1d-results.json`,
`environment/phase1d-audit.json`, and local `outputs/phase1d/` artifacts.
Checkpoint: `outputs/phase1d/training-v2/checkpoints/000300/pretrained_model`.

## Resume-ready description

Built a reproducible SmolVLA/RoboCasa evaluation and fine-tuning pipeline on a 6 GB
RTX 3060; completed 300 optimizer updates with verified frozen/trainable weights
and checkpoint reload, observing CloseFridge simulation success of 0/10 pretrained
versus 3/10 fine-tuned across ten paired seeds, with continuous rollout demos and
explicit validation/generalization limitations.

## Review and suggested commits

1. `Record guarded SmolVLA CloseFridge rollout and state replay` — Phase 1C rollout,
   replay scripts and their historical reports. Keep the initial failed-attempt report
   alongside the successful retry report; they describe different attempts.
2. `Document offline CloseFridge demonstration analysis` — the existing Phase 1C
   demonstration exporter/report, separate from the training result.
3. `Verify SmolVLA fine-tuning and ten-seed paired evaluation` — Phase 1D training,
   rollout/evaluation/report scripts, resolved config, final report, small JSON evidence
   snapshots, and this proposed portfolio text.

No commit was made. Videos/checkpoints/caches remain under ignored outputs/.
The exact resolved training config retains three original absolute Windows paths
as provenance; historical reports also identify the original machine. These are
not secrets, but need local path adaptation for reproduction elsewhere. Tested
training/evaluation/rollout scripts were not changed during this final review.

Known follow-ups, not implemented here: README.md and PROJECT_CONTEXT.md still
describe earlier milestones; reconcile them after approving the proposed result
text. The evaluation resume helper trusts existing completed summaries and writes
aggregate files/logs; preserve partial runs before any separately authorized resume,
as done for seed 47. It is not a general transactional experiment manager.

## Final delivery checks

All 20 videos decoded through their last recorded action. All comparison-page
links and video paths resolve locally; JavaScript syntax and an offline DOM-stub
check passed for default seed 44, choices 47/49/48, and play/pause handlers.
Browser rendering and playback were not visually verified: the browser security
policy rejected the local file URL. No alternative browser access was attempted.
Open the HTML manually for final visual acceptance.

The Git candidate set contains only 16 small text files; no videos, checkpoints
or caches are tracked. A targeted credential/private-key scan found no matches.
Full additions relative to HEAD are in local `outputs/phase1d/final-review.patch`;
ordinary `git diff` is empty because these candidates are still untracked.
