# Bounded instruction-distribution probe
Two valid pretrained episodes completed, task-ID seeds 43 and 44. The prior seed-43 instrumentation failure (one inference, zero actions) remains preserved and is excluded from task outcomes.
Both seed-42 arms and natural-language seeds 43/44 were reused without modification.
Checkpoint: lerobot/smolvla_robocasa @ f7af588d11e30b0069f200792fb285904b74c80b.
Original Phase 1D 0/10 pretrained versus 3/10 fine-tuned results remain unchanged.

## Paired direct measurements
Natural-language success: 0/3. Task-ID success: 1/3.
Door angles below follow recorded joint-name order; seed 42 is left/right, others
have one task-relevant door. Full joint names, initial/minimum/final angles,
motion metrics, timing and artifact paths are in results.json.

| Seed | Arm | Official success | Final door degrees | Actions | Inferences | Base net m | EEF world path m | Close-command steps |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 42 | natural | False | 87.265, 80.183 | 600 | 12 | 0.1405 | 1.658 | 0 |
| 42 | task-id | False | 87.867, -0.000 | 600 | 12 | 0.0040 | 3.158 | 0 |
| 43 | natural | False | 89.988 | 600 | 12 | 0.1403 | 2.099 | 72 |
| 43 | task-id | False | 81.687 | 600 | 12 | 0.0038 | 2.029 | 0 |
| 44 | natural | False | 89.963 | 600 | 12 | 0.0094 | 0.986 | 0 |
| 44 | task-id | True | -0.084 | 342 | 7 | 0.0031 | 1.419 | 0 |

All six videos decode reset plus every action endpoint. New runs passed paired
qpos/qvel/state, camera, scene metadata, model/processor and controller checks.
All logged states remained finite and no guard failures or exceptions occurred.
Success uses the actual RoboCasa predicate, not visual appearance or loss.

## Fine-tuning supervision
The saved split and all 300 sample indices resolve to 157 singular natural-language
instructions and 143 plural natural-language instructions. No update uses
CloseFridge. The task table does contain CloseFridge at index 2, but no frame's
task_index selects it. The separate task-name annotation is not the policy's task.
The trainer passes dataset task text to the original newline/tokenizer processor.
This reconstructs training inputs from pinned data and the unchanged training
script, verifying each sample's episode against the update log; it is not a new
training execution or an independent historical token capture.

## Interpretation and limits
This is a within-checkpoint instruction intervention on three fixed consecutive
seeds. Reused outcomes were already known; it is not a wholly unseen trial.
Any outcome differences demonstrate instruction sensitivity in these episodes,
not generalization, an exclusive failure cause, or proof that training is unnecessary.
The fine-tuning run used natural-language supervision, so interpretation of its
previous natural-language improvement must account for possible language adaptation
as well as task/control learning. This probe does not separate those mechanisms.
World-space motion comparisons are within matched scenes; path length alone does
not measure useful door-closing motion. Close-command counts are commands, not
proof of gripping or contact. No contact forces are inferred.

Existing CPU/CUDA per-scene seeds and sampling calls are retained. Optional
fingerprint instrumentation was removed before these valid runs; the original
noise sampler is untouched. Exact historical noise equality is unverified.
GPU bitwise determinism is not guaranteed. No policy sampling settings changed.

## Preservation and delivery
1157 original files match their pre-probe sizes, mtimes and SHA256 values.
No retraining, dependency changes, weight changes, original-result edits or ledger
updates. No additional seed or automatic retry; the seed-43 replacement was explicitly approved. New Numba cache writes use a copy
inside this probe root. Git diff is saved in git.diff; no commit was made.

## Specific observed differences

- Seed 42: task-ID conditioning closed the right door (~0 degrees), but left the
  left door open (87.867 degrees). Both arms failed. Base net displacement fell
  from 0.1405 m to 0.0040 m; EEF path length increased from 1.658 to 3.158 m.
- Seed 43: task-ID final angle was 81.687 degrees versus natural-language 89.988
  degrees. Both failed after 600 actions. Base net displacement fell from 0.1403
  to 0.0038 m; close-gripper commands fell from 72 to zero. Neither result is close
  to satisfying the closure predicate.
- Seed 44: task-ID closed the relevant door to -0.084 degrees and succeeded at
  action 342; natural language ended at 89.963 degrees and failed at action 600.
  Task-ID EEF path was 1.419 m versus 0.986 m; both issued zero close-gripper commands.
  Small negative hinge angles are reported as measured, without clipping.

This is evidence of instruction sensitivity, including one changed task outcome.
It does not establish that instruction format explains every failure. The
three-seed task-ID result cannot be substituted into the original ten-seed
natural-language benchmark, nor compared directly with its 3/10 fine-tuned rate.
All six valid episode records are complete. The earlier instrumentation failure
remains excluded and explicitly recorded in results.json.
