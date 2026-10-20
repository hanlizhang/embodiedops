# Portfolio delivery review — 2026-10-20

The Terminal-window milestone was reviewed, committed and pushed to `origin/main`
as `53cc938e834681941e2e53a483c5264f81fb1f21`.
Exactly 11 files were included: trainer changes, terminal sampler/audit/runner/
summary source, tests, registration, proposal, report, compact results and ledger.
No checkpoints, videos, caches, temporary logs or unrelated follow-up work entered
that commit. The proposal/audit were included because they define the frozen rule.

## Evidence review

- Recomputed checkpoint selection from saved global-validation probes.
- Compared sample episode order, every logged LR, valid-target counts and finite
  gradients against the registered/native evidence.
- Recomputed episode metrics from saved traces and reconstructed the acceptance
  decision; it matches the stored rejection.
- Checked the selected model SHA256, all five reload records and raw result/source
  snapshot hashes. No model was reloaded or executed during this delivery review.
- Rehashed all 1,122 protected historical files successfully.
- Confirmed no final episode directories and no final seeds evaluated.
- Preserved the recorded reload 4000 runtime overrun; it was not relabeled a pass
  of the runtime bound. The existing post-run guard is covered by a regression test.
- All 21 lightweight unit tests passed. Two initially failed on sandbox temporary
  file permissions; the identical suite passed outside that sandbox. The suite
  includes tiny CPU optimizer and MuJoCo contact fixtures, not RoboCasa rollouts.

## Portfolio artifact review

The public documentation replaces obsolete roadmap/current-status text, while
dated experimental reports retain their original measurements. It distinguishes
NL/task-ID baselines, deterministic routing/human-designed experiments, completed
content checks/runtime failures, and loss/task-success conclusions.

The local HTML demo contains three authentic original recordings: native44
success, native43 incomplete closure and terminal43 rejection. All three decoded
at 20fps, with 221/601/601 frames respectively, matching reset plus 220/600/600
actions. Final frames were inspected. No videos were edited or synthesized.
All local HTML/Markdown references were checked; the builder refuses overwrites.
The two public SVGs are exact copies of measured plots, with source hashes in
`assets/provenance.json`. They are small static evidence, not new experiments.

Browser visual QA was unavailable because the browser tool blocks local file
URLs. No alternate browser or network workaround was used. HTML structure,
links, media decoding and recorded outcomes were verified offline; final browser
layout/playback is not claimed as inspected.

Detailed checks are retained locally under `outputs/portfolio-delivery-v1/`.
The generated demo has its own media/source SHA256 provenance. Videos remain
local and ignored; a fresh public clone cannot play them without those artifacts.
No training, policy inference, RoboCasa episodes, dependencies or model integrations
were launched for this delivery.
