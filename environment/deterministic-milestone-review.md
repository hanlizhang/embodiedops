# Initial deterministic workflow milestone — reviewed 2026-10-19

The workflow source, tests, configuration and report are already recorded in
`bfc197843935c6231d946f3b9ac91fb0ec47c3ea`. The working tree was clean on review.
That commit contains no checkpoint weights, videos or generated output directories;
the existing ignore rules exclude them. Local experimental artifacts are retained.

The saved run audit passes: four new RoboCasa episodes, 512 optimizer updates,
461 covered training episodes, valid selected-checkpoint reload and preserved
historical weight hashes. Nine lightweight tests passed again before this record.
The instruction candidate's 0/2 -> 1/2 diagnostic success change was rejected for
door/contact-force regressions. The learning candidate's held-out loss worsened
0.384626 -> 0.427864 and it was not evaluated in the simulator. Neither candidate
is an accepted repair. This is a deterministic rule baseline, not an LLM agent.
The pre-update Windows cache-path recovery was operator assisted. These limitations
remain explicit in `agent-workflow-v1.md`; no material evidence inconsistency was
found. The next learning milestone must measure new trained weights in closed loop.
