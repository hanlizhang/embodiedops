# Terminal-closing data audit and one proposed experiment — 2026-10-20

**Supported scope: reward-aligned terminal-closing imitation examples.
Unsupported scope: verified contact labels, per-door completion labels, or
recovery demonstrations.** No training, policy inference or simulator execution
was performed. The FP32-master investigation is closed with its negative result;
this proposal uses the archived native-precision Episode-balanced method.

## Failure and data evidence

Existing Episode-balanced seed 43 loses recorded robot–door contact after
action 251; door motion settles near 13.11 degrees by action 285 and no useful
closing resumes. Seed 45 loses right-door contact after 209 with that door near
28.27 degrees, then closes the left door late in the episode. These are incomplete
manipulations; their exact learning cause is not established.
Sources are the existing diagnostic `steps.jsonl`/`contacts.jsonl` and geometry
catalogs in `outputs/agent-workflow-v1/learning-5000-v1/`, with calculations in
`outputs/learning-followup-v1/final/measurements.json`.

The new offline audit reads all **513 demonstrations / 155,443 frames**, retaining
the existing **461 train / 52 validation** episode split and pinned dataset
revision `b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4`.
It checks actual `next.reward`, `next.done`, action[12], state[16] and 20 Hz
timestamps, then decodes existing RGB at metadata-aligned video timestamps.

- All 513 end with positive reward and `next.done=true` only at the last frame.
- **502** have one zero-to-one reward transition followed by exactly **16**
  positive-reward frames. Of these, **450 are training episodes** (300 singular
  instruction, 150 plural) and all **52 validation episodes** qualify.
- **11 training episodes** lose and regain reward. They are excluded from the
  targeted pool, but remain in the ordinary uniform branch. Their IDs are
  50, 83, 103, 111, 253, 339, 344, 426, 427, 431 and 446.
- The eligible training episodes' last 100 pre-reward frames have median
  base-relative EEF travel **0.764 m**, commanded translation offset **35.51 mm**
  per action, achieved base-relative movement **7.64 mm** per frame, and median
  quiet-action fraction **0**. The last 50 frames have median EEF travel **0.422 m**.
  These are active motion examples, not simply recordings of a stationary closed
  fridge. OSC target offsets are not guaranteed displacements in 50 ms.

State indices 7:10 are base-relative EEF position, 10:14 its quaternion and
14:16 gripper joints. Commands 5:8 are OSC translation, 8:11 rotation and 11
gripper. No unaligned world-trajectory comparison or simulator force inference
is used. The export has no door joints, fixture transforms or contact records.
`next.reward` supplies the dataset's success annotation; exact converter timing
relative to the displayed observation is not independently recovered here.

Six training demos were chosen **before viewing**, using length quartiles
(.25/.5/.75) within singular/plural instructions, with episode ID breaking ties.
All three camera views were inspected. Three previously reviewed examples were
also retained, explicitly separate from this stratified sample.

| Demo | Instruction | First reward frame / time | Proposed anchor frames | Visual evidence |
|---:|---|---|---|---|
| 430 | singular | 183 / 9.15 s | 134–149 | Door moves inward through final sweep, then appears shut. |
| 124 | singular | 216 / 10.80 s | 167–182 | Open door rotates inward; hand stays near its exterior through completion. |
| 26 | singular | 258 / 12.90 s | 209–224 | Continued inward door motion during final arm sweep. |
| 169 | plural | 367 / 18.35 s | 318–333 | One door is already visually shut by 244; terminal sweep addresses the other. |
| 267 | plural | 405 / 20.25 s | 356–371 | First door appears shut by 270; terminal window addresses the remaining door. |
| 397 | plural | 444 / 22.20 s | 395–410 | First door appears shut by 296; final sweep closes the other. |

“Appears shut” is visual evidence, not a measured hinge threshold. Neither hand
proximity nor concurrent door motion establishes contact force or contact identity.
The prior demos 0/437/5 show the same general closing pattern and open-finger
commands. Their reward frames are 413/468/284.

The two longest reward-drop intervals were also inspected: demo 50 has reward
1 at 161–173, 0 at 174–203, then 1 at 204–219; demo 427 has 1 at 235–238,
0 at 239–267, then 1 at 268–283. Their RGB shows small pose/door changes near
apparent closure. These could involve threshold crossing or renewed pushing;
they do **not** establish recovery from the large residual openings in policy
seeds 43/45. No “recovery” label is assigned.

Evidence: `outputs/terminal-window-audit-v1/final/{summary,episodes,video-frames}.json`
and `demo-<id>-reward-windows.jpg` (rows: left, right, wrist).
All **258 extracted frames** have valid nonzero RGB and timestamp agreement
within 0.002 s. Only this subset of video frames was checked. No validation video
or final evaluation scene was viewed.

## Exact proposed data intervention

**One change: replace approximately half of uniform frame anchors with full
action chunks spanning the recorded success transition.**
Keep the original masked Flow Matching objective; reward is only an offline
sampling index, never a model input, auxiliary target or online control signal.

For an episode of length \(T\), let \(\tau\) be its first frame with
`next.reward=1`. It qualifies only if reward is zero before \(\tau>0\), remains
one through the final frame, actions/states are finite and frames consecutive.
Define inclusive anchor set:

```text
P(e) = { t : max(0, tau - 49) <= t <= min(tau - 1, T - 50) }
```

Every selected observation precedes reward, and its original actions `t:t+50`
form a full, unpadded chunk containing the first positive-reward row.
For the 450 eligible training episodes, \(T=\tau+16\), so the set is exactly
**tau−49 through tau−34: 16 anchors**, or **7,200 eligible anchors** in total.
Each targeted chunk contains 34–49 reward-zero rows, the first reward-positive
row and 0–15 later rows. It neither starts after success nor adds synthetic
actions. Original RGB/state/action tensors and processors are used unchanged.

The exact 5,000-update proposal is frozen in
`outputs/terminal-window-audit-v1/final/proposed-order-NOT-TRAINED.json`:

1. Retain every episode ID in the archived Episode-balanced 5,000-update order.
2. On zero-based even updates, choose uniformly from that episode's eligible
   pool using NumPy `default_rng(242)`. Draw only for eligible episodes.
3. On other updates, keep the exact archived frame index. For an ineligible
   episode, also retain the archived index instead of drawing.

This produces **2,438 targeted replacements**, **2,500 untouched uniform
updates**, and **62 ineligible fallbacks**. All 461 episodes keep their original
10–11 visits. All 450 eligible episodes receive targeted samples (1–10 each);
the proposal contains 4,614 distinct anchors overall, including 2,109 distinct
targeted anchors. Including uniform samples that happen to fall in the pool, exposure
changes from **274/5,000 (5.48%) to 2,566/5,000 (51.32%)**.
Padded chunks decline from 939 to 469 because the targeted branch requires full
chunks; that accompanying change in valid target coverage must be reported.
The unchanged uniform half retains approach, first-door and ambiguous-tail data.

The archived baseline already has terminal examples (1,874 anchors in the last
100 pre-reward frames and 942 chunks containing the first reward row).
This is an **emphasis hypothesis**, not evidence that terminal examples were absent.
It tests whether more intact demonstrations of finishing a closing sweep prevent
premature departure. It does not teach a measured return-to-door recovery strategy.
For seed 45, whole-task reward does not localize finishing the first door;
transfer from single-door/final-door examples is uncertain.

## Matched training, selection and verification — proposed only

- **Baseline:** archived native Episode-balanced 5,000-update run, selected step
  5,000, validation loss 0.367255928. Do not resume or replace its checkpoint.
- **Candidate:** one independently initialized run from the same pinned original
  pretrained SmolVLA, **5,000 updates**, native BF16/FP32 path. No FP32 masters.
  Same batch 1, task-ID `CloseFridge`, split, normalization, architecture, trainable
  tensors, noise seeds `100000+k`, AdamW settings, gradient clipping and LR schedule.
  No extra epochs, optimizer variants, horizon changes or hyperparameter search.
- **Fixed probes:** retain original 32 training and 104 validation anchors/draws,
  measured at 0 and every 500 updates. Add an independently reported terminal
  probe with two anchors per held-out episode (the two endpoints of `P(e)`), fixed
  noise/time seeds `92000+i`. Its 104 anchors are frozen in
  `terminal-validation-anchors.json`; no validation episodes enter training.
  These are correlated within each of 52 episodes.
- **Checkpoint rule:** save 1,000/2,000/3,000/4,000/5,000; select lowest **original
  104-anchor global task-ID validation loss**, earliest tie. The new terminal
  probe is diagnostic and does not change selection. Evaluate that probe on the
  archived native selected checkpoint as the direct reference. Never select by
  diagnostic or final robot success.
- **Integrity:** independently reload saved model, optimizer, scheduler and
  processors, reproduce fixed losses, and check hashes/frozen weights before
  robot use. A timeout remains an unresolved verification blocker, not a pass.
  The previously timed-out FP32 checkpoints remain closed and are not retried.

Log actual stochastic/fixed losses, LR, gradients/clipping, parameter changes,
sample coverage, valid action counts, memory and duration. There is no new
training implementation or invocation yet; only the audit and proposed index
manifest exist.

## Preregistered robot test and acceptance criterion — proposed only

**Pilot:** at most **four new candidate episodes**, diagnostic seeds **43–46**,
compared with the valid cached native and pretrained episodes. Require matching
reset, instruction, cameras, processors, controller, 50-action prediction/execution
horizon, 10 flow steps and 600-action cap. Use the established renderer correction.
Native seed 46 refers to its valid replacement, not the excluded blackout attempt.
Run the selected technically valid checkpoint even if its offline loss is worse,
explicitly as research evaluation; offline regression still blocks promotion.

The candidate passes the diagnostic repair criterion only if all of these hold:

- Actual success rises from 1/4 to **at least 2/4**, **at least one of seeds 43/45
  succeeds**, and seed 44 retains success.
- No new camera/reset/inference invalidity or guard failure. Every door's final
  angle is no more than **1 degree worse** than the matching native episode.
- The existing acceptance gate passes separately against both archived references:
  no lost per-seed success, no summed door-progress regression over 1 degree, and
  peak recorded normal force no greater than `1.25 * reference_peak + 25 N`.
  These are comparative simulator gates, not certified hardware safety bounds.
- Selected global validation loss is **no worse than the native 0.367255928**
  (and therefore better than pretrained 0.369275332), and terminal-probe mean
  loss is lower than the archived native checkpoint under identical draws.

Measure actual predicate, each door's final/minimum angle, stall duration, last
loaded door contact and subsequent re-contact, wall/self contacts, force peaks,
EEF path and inference latency. Contact geometry supplies identity; positive
solver force supplies loaded contact. Do not infer forces from EEF proximity.
Report robot gains even when another gate rejects promotion; a lower loss alone
does not count as repair.

**Gated final comparison:** only if the frozen pilot candidate passes, evaluate
the unchanged native reference and candidate on **120–123**, exactly eight further
episodes (maximum **12 new episodes** across pilot and final). No scene or
checkpoint selection uses final results. Final acceptance requires strict paired
success gain and the same validity/door/contact regression gates versus native;
otherwise reject. Four final seeds provide limited evidence, not robust
generalization or an official benchmark reproduction.

**Expected cost, not execution authorization:** archived native training took
48.4 worker-minutes and 1.732 GiB peak CUDA allocation. Allow roughly 60–150 minutes
for this native candidate plus added probes under variable laptop clocks; about
2 GiB training allocation is an estimate to verify. Proposed hard caps: 180 min
training, 15 min per independent reload (five), 7 min per robot episode, 6 hours
total including gated final evaluation. One 5,000-update candidate only; timeout
or integrity failure stops without automatic retries. No energy estimate without
adequate power sampling. Training and robot execution await approval.

Reproduce the offline audit into a fresh output directory:

```powershell
& 'D:/Apps/miniconda/envs/embodiedops-robocasa/python.exe' -B scripts/terminal_window_audit.py --output outputs/terminal-window-audit-NEW
```

The final audit's `audit-validation.json` verifies the proposed order, full
targeted chunks, unchanged episode order, disjoint validation and video timestamps.
Source/data/manifest hashes are recorded. This audit and proposal are deliberately
outside the prepared negative-precision-ablation commit.
