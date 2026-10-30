# Recontact self-imitation v2 — review evidence

Exported from existing local records on 2026-10-30. No training, inference,
simulation or new experiment audit was run for this Git synchronization.

- [Registered plan](registered-plan.json): fixed recipe, split hashes, models,
  36 conditions, noise streams, budgets and acceptance rules.
- [Whole-trajectory split](data-split.json): 10 training / 3 held-out sources,
  duplicate grouping and selected sample IDs.
- [Source index summary](source-index-summary.json): all 13 sources and 39 sample
  metadata entries, observation/action hashes, validity masks, recorded recontact
  events, 24-collection distribution and original test-seed audit. Action arrays
  and duplicated nested sample payloads were omitted; originals remain local.
- [Final result](result.json): all 36 final condition records (`conditions`), all
  50 attempts (`attempts`), invalid reasons and retry mapping by condition ID and
  attempt number, training exposure, step-300 hash and independent reload. The
  existing CSV, attempt ledger and separate training summary are not duplicated.
- [Frozen evaluation models](frozen-models.json): exact inference paths and hashes.
- [Runtime source amendments](runtime-source-amendments.jsonl): preserved source
  hashes for the two infrastructure repairs.
- [Evidence index](evidence-index.json): source paths, byte sizes, SHA-256 hashes,
  export rules and existing stage timing records.

The first five copied artifacts listed in the evidence index retain original
bytes; the scoped `.gitattributes` disables line-ending conversion for those copies.
Source/observation/video/checkpoint paths inside the JSON are local
provenance references, not files included in this repository. No model tensors,
raw RGB, videos, substep logs, datasets or Junction targets are published here.

The completed pilot found no added qualified success. Full-task qualified counts
were frozen 1/6, ordinary_300 0/6, recontact_mix_300 0/6; recovery counts were
6/6, 3/6, 3/6. Seed 64 regressed relative to frozen. All 36 final conditions were
valid, with 14 retained invalid/superseded infrastructure attempts and no hard
stops. Experiment execution finished in about 2.08 hours, but final delivery
exceeded the original 8-hour total budget.

Read the [complete report](../../../environment/recontact-self-imitation-v2.md)
and [delivery handoff](../../../environment/deadline-handoff-2026-10-29.md).
