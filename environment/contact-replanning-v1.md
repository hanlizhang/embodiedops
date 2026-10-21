# Contact-aware event-triggered replanning v1

**Status: four-pair milestone complete; candidate rejected.** H50 succeeded on
2/4 diagnostic scenes versus 1/4 for the candidate. The user explicitly authorized
one seed46 baseline retry and the remaining candidate after the original control
stopped on CUDA OOM before inference/action. Both continuation episodes completed.
The failed attempt remains preserved and excluded. Frozen thresholds, weights,
controller, inference settings, seeds and per-episode limits stayed unchanged;
only the authorized retry exception and fresh artifact paths differ.

Measured artifacts: [compact results](contact-replanning-v1-results.json),
[paired CSV](../outputs/agent-workflow-v1/contact-replanning-v1-seed46-continuation/paired-results.csv),
[full analysis and events](../outputs/agent-workflow-v1/contact-replanning-v1-seed46-continuation/analysis.json).

![Measured per-door trajectories and event timing](../outputs/agent-workflow-v1/contact-replanning-v1-seed46-continuation/door-progress.png)

## Measured task outcomes

| Seed | H50 success / actions | Candidate success / actions | Final opening°, H50 → candidate | Actual replans / discarded |
|---|---|---|---|---|
| 43 | Yes / 472 | No / 600 | 0.424 → 6.602 | 1 / 28 |
| 44 | Yes / 219 | Yes / 220 | −0.079 → 0.274 | 0 / 0 |
| 45 | No / 600 | No / 600 | [−0.004, 24.914] → [−0.006, 26.523] | 0 / 0 |
| 46 | No / 600 | No / 600 | 84.423 → 89.994 | 1 / 1 |

Success uses the actual CloseFridge predicate. Seed45 requires both doors to close.
Best / final summed closing progress: seed43 H50 **85.100/85.100°**, candidate
**78.922/78.922°**; seed44 **83.396/83.396°** versus **83.043/83.043°**; seed45
**143.095/143.012°** versus **141.438/141.405°**; seed46 **1.996/1.996°** versus
**0.034/−3.575°**. Per-door best progress is in JSON.

| Seed | Peak normal force N, H50 → candidate | EEF world path m | Inference calls | Inference total s | Rollout wall s |
|---|---|---|---|---|---|
| 43 | 451.12 → 1339.61 | 2.122 → 2.791 | 10 → 13 | 7.66 → 10.48 | 102.32 → 128.72 |
| 44 | 181.21 → 172.98 | 1.486 → 1.493 | 5 → 5 | 4.52 → 4.50 | 42.68 → 42.11 |
| 45 | 137.18 → 135.28 | 3.485 → 3.448 | 12 → 12 | 9.23 → 7.84 | 204.00 → 165.55 |
| 46 | 232.33 → 208.85 | 1.886 → 2.275 | 12 → 13 | 8.60 → 9.23 | 124.67 → 124.87 |

No inherited guard activated in these eight valid episodes. Peak torch CUDA
allocation was **971,354,112 bytes** in each; this excludes renderer/other-process
allocations. H50/candidate totals: **39/43 inference calls**, **30.01/32.05 s**
inference, **473.67/461.24 s** rollout, **709.88/675.25 s** full worker time.
Worker time includes loading/setup/finalization; rollout time excludes them.
Host/runtime variation and unequal termination lengths prevent a speedup claim.
H50 executed 1,891 actions; candidate 2,020, each within its own 600-action cap.
Simulated durations were 23.60/30.00 s, 10.95/11.00 s, 30.00/30.00 s and
30.00/30.00 s by pair. These totals exclude the invalid reset-only OOM attempt.
Across all four candidate episodes, two events discarded 29 actions.

The existing gate rejects the candidate for **lost seed43 success, no success
gain, door-progress regressions on 43, 45 and 46, and seed43 contact-force regression**.
Its force ceiling on43 was 588.90 N (1.25×451.12+25), below 1339.61 N.
The full four-pair evaluation is complete. No final seeds were used.

## What the two events did

### Seed 43: transient closure without completion

At action **122** (simulator time 6.600 s, 6.100 s after reset), five consecutive
loaded tracking-stall records caused interruption. At the trigger, command-time
target residual was **28.08 mm**, achieved EEF movement **0.609 mm**, and peak
hand/door and finger/door normal forces **66.97 N / 31.78 N**. Geometry67/69 are
the right hand/finger; geometry261 belongs to the fridge door. Mode was achieved,
with an open-gripper command; contact does not establish a grasp.

H50 would have continued the saved **28 rows for actions 123–150** from the chunk
predicted at 101. Its next scaled translation increment was
**[7.39, −6.90, −12.07] mm**. The fresh observation at123 instead generated a
chunk beginning **[13.15, 9.12, −3.53] mm**. Translation-command RMS difference
over those 28 replacement rows was 0.480 public units. Full 12D tails are retained;
unexecuted commands do not prove a counterfactual physical trajectory.

Over 20 actions before/after the trigger, closure increased from **0.026° to 7.628°**,
door-stall counts fell18→1, mean EEF movement rose2.46→4.56mm/action, and neither
window had a >0.1° opening reversal. The new chunk changed local behavior, but
at action 140 the episode also recorded its **link5/link7 self-contact peak of
1339.61 N**. The door later stalled at 6.60°; the last 100 actions had no ≥1 N
robot–fridge contact, so this registered contact rule could not fire there.
The intervention did not deliver useful task completion or learned recovery.

No candidate event occurred on 44 or 45: neither had any action satisfying all
three conditions simultaneously. Seed45's baseline had five qualifying actions,
but its longest run was only two. Thresholds were not weakened to force events.
The rule identifies a loaded execution discrepancy, not a semantic manipulation
stage, useful hinge torque, or whether SmolVLA knows a recovery action.

### Seed 46: one action discarded, failure repeated

At action **99** (simulator time5.450 s;4.950 s after reset), hand contact against
fridge-door geometries244/264 had peak normal force **16.03 N**. The command-time
target residual was **15.32 mm**, while achieved EEF movement was **0.371 mm**.
Actions95–99 met the frozen five-step criterion. The goal-update mode was achieved.

Only **one queued action** remained. H50 would execute it at100 and naturally
replan at101. Its scaled translation increment was **[6.32, −3.43, −4.61] mm**;
the fresh chunk at100 instead began **[−7.53, −2.42, −10.53] mm**. Translation-command
RMS difference for the replaced row was0.174 public units. This was a one-step
advance in observation/replanning time, also shifting subsequent noise seeds and
chunk boundaries; it was not a new recovery policy.

In the20 actions before/after, summed closure changed **−3.586° → +0.191°**,
opening reversals10→0, door-stall counts10→20, and mean EEF movement2.925→0.799mm.
The replacement changed commands but largely repeated the physical stall.
Persistent contact did not cause repeated firing: there were91 qualifying actions
and a longest qualifying run of23, but only one event under the release/cooldown
safeguard. The last100 actions had no ≥1N fridge contact. Final opening was
**89.994° versus84.423°** for H50, a **5.571° door-progress regression**.
The overall peak force decreased, but link5/hand self-contact peaked at208.85N
versus112.76N in the baseline; peak-force category details remain in JSON.

## Reproducibility limits and authorized completion

Reset state/qpos/qvel, scene/controller, checkpoint/processor hashes, cameras,
instruction and sampling schedule passed the existing paired contracts. All eight
valid logs passed exact action/prediction correspondence, queue conservation,
deterministic decision replay and final nonblack-camera checks. The initial953
historical files and all prior experiment artifacts, **1,355 files total**, were
hash-preserved during continuation. Frozen source/config hashes were unchanged.
The original28 lightweight tests passed; the unchanged execution code was reused.

Matching seeds did **not** yield bitwise trajectories. First-action maximum
differences on43/44 were0.003020/0.000678 public units. Before the event on43,
maximum action difference reached0.02450. Even no-event seed45, whose first action
matched exactly, diverged later (max action difference0.8370). The numerical or
runtime cause was not isolated. Thus seed45's regression cannot be blamed on
replanning, and the full seed43 outcome is not an isolated causal estimate.
The fresh seed43 H50 success also differs from its archived13.11° failure.
For seed46, actions and qpos matched **exactly through action99**, before queue
interruption. It offers a cleaner matched-prefix comparison than43, while the
shifted replanning time still changes observations and stochastic sampling.
These are four diagnostic scenes, not evidence of generalization.

Seed46's fourth control attempt failed during the first preprocessed-batch finite
check with `torch.AcceleratorError: CUDA error: out of memory`: **zero inference
calls, zero physical actions**. CUDA reporting may be asynchronous. The runner
stopped as registered, preserved the reset/exception, and did not launch candidate46.
A later read-only GPU snapshot showed1592MiB used /4403MiB free; it does not establish
memory availability at failure or its cause. The OOM is not a robot-task failure.
The user then explicitly authorized exactly one baseline retry and the remaining
candidate. `scripts/contact_replanning_resume.py` ran only those two episodes in
`outputs/agent-workflow-v1/contact-replanning-v1-seed46-continuation/`, retaining
the same600-action/600-second limits and all frozen settings. Both reached600
actions with valid cameras. The OOM cause remains unresolved; successful retry is
not proof of a resource-lifetime fix. Prior reports/results were snapshotted before
updating this report, and the original failed attempt was never overwritten.
A concurrent read-only analysis briefly hit MemoryError; after the rollout worker
exited, the complete audit succeeded without changing the environment or episodes.
No environment change, threshold sweep, training, further retry or commit occurred.

The unchanged displacement/numerical guards did not stop the high-force contact.
They are not comprehensive force-safety protection; the independent force gate
rejected it. The supervisor neither learned recovery nor made the VLA contact-aware
through training. The authorized four-pair comparison is now complete and stopped.

## Interview summary

“I added a simulation-only execution supervisor around a fixed SmolVLA policy,
using verified fridge contact and sustained command-tracking stalls to interrupt
its native 50-action queue. Unit tests and real rollout audits verified refill,
action accounting and deterministic event decisions. Across four completed pairs,
two events changed commands but did not recover task success; the candidate was
rejected at1/4 versus2/4 control successes, with door and force regressions.
I preserved a failed CUDA attempt and completed only the explicitly authorized
retry, retaining frozen thresholds and the numerical-reproducibility limitations.
Correct event detection was not evidence of a successful repair.”

## Registered protocol (before evaluation, 2026-10-21)

This original registration is retained below. The later user authorization allowed
one additional seed46 baseline attempt after OOM; it did not change the event rule,
candidate count, per-episode limits or other execution settings.

Hypothesis: a fresh RGB/proprioception-conditioned action chunk after a loaded
tracking stall can improve execution without changing the learned policy.
The simulation-only supervisor is deterministic and has no door-angle or success
input. Privileged contact is never supplied to SmolVLA.

Both arms use the selected episode-balanced step-5000 checkpoint, CloseFridge,
50 predicted actions, 10 flow steps, unchanged hybrid controller and guards,
600 physical actions maximum, and diagnostic seeds 43–46. Four fresh H50 controls
are needed because archived logs lack per-step actual controller goals and full
predicted chunks. At most four controls and four candidates; no retries, final
seeds, threshold sweeps, environment changes or training. Worker deadline: 600 s.

The frozen rule requires **five consecutive 20 Hz steps** satisfying all three:

- Robot–fridge solver contact with normal force at least **1 N**: excludes tiny
  numerical forces and proximity; self and other environmental contact are separate.
- Actual OSC translation target error at least **0.01 m**: a substantial residual
  goal, not an assertion that an action increment must execute within 50 ms.
- Measured EEF displacement at most **0.001 m per step**: sustained low motion
  over 0.25 s despite that residual goal and loaded contact.

Target and achieved displacement use the same command-time controller base frame,
with its actual origin/rotation and goal, including achieved/desired update mode.
The public translation increment clips at ±1 and scales by 0.05 m; it is logged
separately from actual target error. Forces stay contact-local, never hinge torque.
Fridge geometry comes from the actual fixture's contact geometry membership.

An event clears only the remaining native action deque. The next loop checks
camera validity before processing a fresh observation and native chunk refill.
Persistent contact cannot repeatedly fire: rearming requires **five unloaded
steps** and **20 steps (1 s) since the last event**; at most **12 events** occur.
Unsafe or terminal steps cannot interrupt; original motion, numerical, controller
and camera guards remain authoritative. There is no new force-based safety claim.
The baseline computes shadow decisions but never clears its queue.

Before every new chunk, RNG seed = episode_seed × 100000 + action_index.
Matched resets do not imply identical subsequent actions: interruption changes
observation timing and the noise seed. All predicted chunks and discarded tails
are retained; they describe unexecuted commands, not a simulated counterfactual.

Acceptance uses the existing independent gate: strict paired success gain, retain
every reference success, valid matched episodes, no new guard failures, no summed
door-progress regression over 1°, and peak normal force ≤1.25×baseline+25 N for
each seed. Door joints and actual success remain evaluation-only measurements.
Partial progress, more calls and correctly fired events alone do not pass.

Code/config hashes are frozen in the exclusive output root before any episode.
Tests and measured results are appended after verification; this registration
does not claim the experiment has completed.

## Implementation and preflight

`scripts/contact_replanning.py` implements the independently testable rule;
`phase1d_rollout.py --execution-feedback baseline|candidate` integrates it without
changing SmolVLA, its native queue refill, action conversion or physics control.
`contact_replanning_experiment.py` freezes source/config hashes and runs the
bounded matrix. All **28 lightweight tests passed**, including seven new feedback
tests and the installed native H50 queue test. Tests ran before evaluation.
Initial sandboxed test processes appeared stalled during imports, but both later
returned successfully (seven tests each); no sandbox/native-runtime defect was
established. The full suite and episodes ran in the existing environment outside
the sandbox, with no dependency modifications.

The logged residual is specifically the **command-time OSC target point** minus
the achieved endpoint in that same command-time frame. The hybrid controller
updates its base origin during physics substeps, so this is not its instantaneous
final-substep torque-control error. World EEF displacement is rotated into the
command-time frame; it includes mobile-base motion. No action component is treated
as a guaranteed displacement. This limits interpretation during base movement.

Loaded means at least one of the 25 sampled solver contacts in an action meets
the 1 N threshold; it does not mean loading was continuous for every substep.
Each event requires five consecutive such action records with the motion/error
conditions. The baseline logs **shadow** decisions with `enabled=false`:
its `decision.events` and `decision.actions_discarded` are hypothetical, while
its actual queue remains intact and analysis reports zero actual replans/discards.

Full chunk predictions, discarded tails, frame transforms, contact identities,
controller mode, per-step physical states and independent door angles are saved
under `outputs/agent-workflow-v1/contact-replanning-v1/`. The offline audit checks
exact predicted/executed action correspondence and deterministic decision replay.
Twenty-action before/after event windows use descriptive door-stall (<0.05°/step)
and reversal (>0.1° opening/step) counts, which never enter the online rule.
