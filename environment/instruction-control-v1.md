# Instruction-controlled 300-update study

## Final status: closed with 52/60 valid cells — 2026-10-22

**The user closed this study. Do not resume its remaining eight cells.** This
closure supersedes every continuation, recovery and confirmation proposal in
the historical sections below. The final evidence is an interim analysis of
52 valid completed cells, not a complete 60-cell benchmark. The frozen ledger
and output reports retain their original stopped/incomplete state; closure does
not relabel unrun cells or manufacture a completion marker.

The authorized continuation completed 31 additional valid cells, including the
fresh B-ID-45 recovery. The study is **incomplete: 52 completed, 1 invalid,
7 unrun**. C-NL-50 exceeded the original 600-second whole-worker cap after
479 recorded actions. The driver terminated that worker and stopped; there was
no automatic retry. C-ID-50 and all six seed-51 conditions remain unrun.
The cause of the slower startup/throughput is not established. This was a
wall-clock guard stop, not a disk breach or a scored manipulation failure.
No Python workers remained in the post-stop native process check.

The original 21 valid cells and both historical interrupted attempts are intact.
SHA256 preservation checks passed for 1,572 prior evaluation files and 994
historical files. All A/B/C checkpoint, processor and frozen execution-source
checks passed. The D: output junction remains correct. Across 1,217 active-worker
monitor samples, minimum free capacity was C: 17.05 GiB and D: 167.25 GiB, above
the unchanged 2 GiB floor. Maximum observed pagefile allocation was 12.11 GiB.
The final audit read C: 30.98 GiB and D: 167.25 GiB free. Pagefile settings were
not changed. No training, confirmation seeds, final seeds 120–123 or commits ran.

Results use valid completed cells only; invalid/unrun cells are not failures:

| Checkpoint | CloseFridge | Single / double | Natural language | Single / double |
|---|---:|---:|---:|---:|
| A: pretrained | 3/9 (33.3%) | 3/6; 0/3 | 0/9 (0%) | 0/6; 0/3 |
| B: original NL 300 updates | 3/9 (33.3%) | 3/6; 0/3 | 1/9 (11.1%) | 1/6; 0/3 |
| C: task-ID 300 updates | 1/8 (12.5%) | 1/6; 0/2 | 0/8 (0%) | 0/6; 0/2 |

A-ID and B-ID succeed on the same seeds 44, 47 and 49. C-ID succeeds only on 49;
B-NL also succeeds only on 49. All valid double-door episodes fail. The report
contains outcomes for every registered scene, distinguishing invalid from unrun.
On the common eight complete scenes (42–49), ID success is A 3/8, B 3/8, C 1/8;
mean final closure progress per door is 44.11°, 43.13°, 39.56°, respectively.
Neither fine-tuned checkpoint has demonstrated improved CloseFridge success on
those matched scenes. This is a descriptive partial result, not a final ten-scene
or generalization conclusion. B-NL shows one observed gain over A-NL, but the
historical 0/10 → 3/10 result has not been reproduced: historical B-NL successes
44 and 47 now fail, while 49 succeeds. Seed 51 remains unrun, so the registered
aggregate comparison is incomplete and will remain so in this closed study.

Physical evidence from completed cells (C has one fewer scene):

| Arm | Mean final progress / door (°) | Mean loaded fridge actions | Median episode peak fridge force (N) | Mean chunk latency (s) |
|---|---:|---:|---:|---:|
| A-ID | 42.81 | 182.0 | 168.7 | 0.691 |
| B-ID | 39.13 | 160.6 | 493.4 | 0.742 |
| C-ID | 39.56 | 227.9 | 372.2 | 0.672 |
| A-NL | 6.43 | 93.7 | 64.5 | 0.663 |
| B-NL | 26.24 | 185.9 | 353.1 | 0.674 |
| C-NL | 11.73 | 120.4 | 337.4 | 0.640 |

The per-door mean pools doors; it does not establish task completion. For example,
A-ID-50 ends at [83.14°, 15.49°] and B-ID-50 at [83.14°, 49.71°]: both fail despite
partial closure. Loaded contact means at least 1 N sampled solver normal force
within an action, not a verified grasp. B-ID's maximum fridge peak was 1,160.5 N;
B-NL's maximum self-contact peak was 904.8 N. More contact or force is not evidence
of better manipulation. Raw per-door initial/final/best progress and phase-window
contact evidence are retained in the CSV and JSON reports.

Successful completion times at 20 Hz, in seed order: A-ID [15.20, 8.15, 16.45] s;
B-ID [15.45, 8.85, 17.05] s; C-ID-49 19.70 s; B-NL-49 16.35 s. These are simulated
action times, not worker wall times. Completed workers total 8,018.754 s
(133.65 min), of which instrumented inference is 392.188 s (6.54 min); rollout
time is 5,461.479 s. Peak reported PyTorch allocation is 971,354,112 bytes
(0.905 GiB), not total GPU/process memory. The new invalid worker additionally
consumed its 600-second allowance. Completed-worker totals exclude all interrupted
workers and orchestration/audit overhead; no energy or active GPU time was measured.

Existing first chunks provide exact fixed-observation evidence for NL seed 44:
all A/B/C raw RGB and state arrays match. A–B action RMSE is 1.105 for the goal-mode
coordinate and [0.278, 0.092, 0.104] for translation-command coordinates, before
controller clipping/scaling. All three NL rollouts fail. Prediction changes alone
therefore do not establish physical improvement. Other comparisons with sparse
pixel differences are explicitly not labeled exact fixed-observation evidence.
No new fixed-state inference was run. Noise bytes were not independently hashed;
historical RNG/renderer differences, permitted camera rounding, known seed-43
runtime sensitivity and this new timeout limit numerical reproducibility claims.
These are a small diagnostic set, with no confirmation stage or population claim.

C remains the verified step-300 inference checkpoint with original saved
processors. Its optimizer/scheduler state is incomplete; independent inference
reload and fixed-probe validation passed before evaluation. No training replay.

Artifacts under `attempt02/evaluation/resume-after-archive-20261022`:
`ledger.json`, `stopped.json`, `final-integrity.json`, `report.md`,
`all-60-cells.csv`, `comparison-plot.png`, `comparison-plot.svg`,
`physical-comparison.json`, `physical-phase-windows.json`,
`existing-first-chunk-comparison.json`, and `runtime-storage.jsonl`.
The eight incomplete cells are closed without execution: C-NL-50 (invalid),
C-ID-50 (unrun), and all six seed-51 conditions (unrun). Earlier A-NL-45
(zero actions) and B-ID-45 (12 actions) remain separate invalid attempts;
their authorized replacement episodes are among the 52 valid cells.

## Closure verification and evidence links

Read-only verification on 2026-10-22 recomputed checkpoint/processor SHA256s,
matched the 52 unique result paths to the 60-cell ledger, checked recorded action
counts and saved paired-reset validation for every valid cell, and confirmed all
52 videos and summaries remain accessible. The two original interrupted attempts
and 479-action timeout are preserved. All 1,572 files in the continuation's
preservation inventory and all 994 historical files still match their hashes.
Frozen execution-source hashes still match the registered plan. No model reload,
loss recomputation, training, inference, simulation or new diagnostic experiment
was performed for closure; saved reload evidence was verified by reading it.

Checkpoint identities (`model.safetensors`, SHA256):

| Arm | Evaluated artifact | SHA256 |
|---|---|---|
| A | Pinned `lerobot/smolvla_robocasa`, revision `f7af588d11e30b0069f200792fb285904b74c80b` | `96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b` |
| B | `outputs/phase1d/training-v2/checkpoints/000300/pretrained_model` | `0e011aa494e0d4e370b11feb5eec6e0c49af2fadd1f3f97ee03056909de0fe6d` |
| C | `attempt02/evaluation/inference-C`; weights match retained `attempt02/C/checkpoints/000300/pretrained_model` | `e3cc9a2084f081778b2353bfc8e8c9d027e933e224eca37205be2d2c9715a3c8` |

All three evaluated artifacts use byte-identical original processor JSON and
normalizer/unnormalizer files, checked against the four hashes in
[the original audit](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/audit.json).
The C inference bundle's original processors are distinct from the raw
reserialized files left by the failed training-state save. The original C model
file also matches the evaluated bundle. C is inference-validated, not a complete
optimizer/scheduler checkpoint; no recovery replay was executed.

Recorded fixed-probe means, checked against saved sample arrays and audits:

| Trajectory / conditioning | Step | Fixed training loss | Fixed held-out loss |
|---|---:|---:|---:|
| A at start of B / NL | 0 | 0.30132371559739113 | 0.531797094270587 |
| B / NL | 150 | 0.4431479685008526 | 0.5798786552622914 |
| B / NL | 300 | 0.35254982113838196 | 0.5941866030916572 |
| A at start of C / ID | 0 | 0.27756017353385687 | 0.4483210798352957 |
| C / ID | 150 | 0.4243288952857256 | 0.5719739766791463 |
| C / ID | 300 | 0.3419970776885748 | 0.5918905511498451 |

These are eight-anchor probes per split with saved noise seeds. The NL and ID
columns arise from different conditioning; they are not cross-model losses under
one fixed instruction. Both trajectories worsen their own held-out mean. Neither
these losses nor changed predictions identify the cause of physical failure.
[B's saved reload](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/B-reload-verification.json)
matches its recorded validation mean; [C's saved reload](../outputs/instruction-storage-audit/reload/result.json)
reports all 500 tensors and exact agreement for all 16 fixed probes with both
processor variants (maximum difference zero). These are historical reload results,
not new execution at closure.

Verified local output links (large artifacts remain ignored by Git and are not
bundled with a public clone):

- [Frozen plan](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/registered-plan.json)
  and [final 60-cell ledger](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/ledger.json).
- [52 valid results](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/results.json),
  [per-cell CSV](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/all-60-cells.csv)
  and [scene/result table](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/report.md).
- [Comparison plot](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/comparison-plot.png),
  [physical metrics](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/physical-comparison.json)
  and [fixed-observation evidence](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/existing-first-chunk-comparison.json).
- [Timeout record](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/stopped.json)
  and [preservation/integrity audit](../outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation/resume-after-archive-20261022/final-integrity.json).

Source review confirmed the study's opt-in raw-observation recording, task-ID
substitution before preprocessing, original-processor use, junction validation,
whole-worker timeout and valid-cell reporting paths. The existing three mock
renderer-lifecycle tests passed at closure, and study Python sources parsed.
Those tests do not cover full training/recovery/orchestration behavior; no such
execution is claimed. No source or test implementation was changed for closure.

## Handoff: Corrective Imitation Learning milestone

**Manipulation failure's root cause remains unresolved.** Missed or ineffective
contact, incomplete terminal closure, reopening and high contact forces are
observations. Missing recovery demonstrations are a hypothesis, not an established
data deficiency or a demonstrated causal explanation. This study did not measure
recovery coverage or test corrective supervision. Controller geometry, observation
and numerical effects, and action-distribution limitations remain possible factors;
the current evidence does not isolate them.

The next milestone is to design a separately authorized Corrective Imitation
Learning study using this closed baseline and retained failure trajectories.
Specify what corrective state-action supervision would address, how demonstrator
quality and episode-disjoint coverage would be checked, and what matched physical
success, per-door progress and contact-force criteria would support improvement.
Keep the pretrained A control, explicit instruction contracts, fixed validation
selection and untouched final seeds as design constraints. Do not choose a repair
or checkpoint merely from changed predictions or training loss. This handoff is a
planning boundary, not authorization to collect data, train or run evaluations.
The next milestone must not silently fill or pool this study's eight closed cells.

## Historical record — superseded execution plans and intermediate findings

The remainder preserves dated evidence and earlier authorizations. References to
"current", "latest", recovery or resumption below describe their historical point
in time; the closure decision above governs current work.

## Authorized continuation after archival — 2026-10-22

Following approved deletion of the two verified C: archive backups, the user
explicitly authorized the remaining39 cells. This supersedes the historical
pause below for this bounded continuation only. The21 completed cells remain
unchanged; B-ID-45 uses a fresh `B-ID-45-resume1` directory and its original
12-action attempt remains invalid evidence. No other new retry is authorized.

Current ledger/results/telemetry live in
`attempt02/evaluation/resume-after-archive-20261022`. The same frozen sources,
checkpoint/processor hashes, instructions, controller, renderer, refill RNG,
seeds42–51,600-action/600-second limits and2GiB reserve remain enforced.
Native free-space and EnumPageFilesW allocation/usage are recorded before/after
episodes and every5 seconds during workers. The parent terminates its own worker
on a breached resource guard and records the invalid cell; it does not retry.
Prior complete worker costs plus600 seconds for each of the two historical
interruptions are reserved against the original10-hour budget. Historical
artifacts and reports remain separate from this continuation. No additional
confirmation/final seeds, training, pagefile changes or automatic commit.

## User pause and manual pagefile recovery plan — 2026-10-22

Evaluation is explicitly PAUSED. No more RoboCasa episodes or policy inference
until the user changes Windows settings manually, reboots, the actual configuration
and evidence pass independent verification, and the user explicitly approves
continuation. No OS settings were changed by this audit.

Native audit at 15:58–16:02 UTC found global AutomaticManagedPagefile=true,
PagingFiles=`?:\pagefile.sys`, and only active `C:\pagefile.sys`. An empty
Win32_PageFileSetting result under global automatic management does not mean
paging is disabled. Win32_PageFileUsage reports allocated4,629 MiB, current722 MiB,
boot peak usage1,237 MiB, TempPageFile=false. Physical memory visible to Windows
is15.81 GiB. Latest GetPerformanceInfo commit is13.31–13.34 GiB /20.33 GiB limit,
with7.43–7.44 GiB available physical memory. CIM counters independently show
similar current values. Boot-wide peak commit is31.51 GiB; this peak cannot be
attributed to this experiment alone and is not the same as pagefile PeakUsage.
No Python or study workers remain. Current idle samples show no sustained paging
storm, but cannot exclude earlier transient pressure or a longer-term leak.

C: and D: are basic NTFS partitions3/4 on disk0, a healthy online writable
Micron MTFDKBA512TFH NVMe SSD. D: is a fixed local volume, not the separate Samsung
disk1. It is suitable for a system-managed pagefile; this relocates capacity demand,
not I/O to an independent disk. Current free space is C:11.07 GiB / D:185.51 GiB.
CrashDumpEnabled=3 means Small memory dump; directory C:\WINDOWS\Minidumps.
No dedicated dump file is configured.

Recommended MANUAL settings:

1. Run `sysdm.cpl`; Advanced → Performance Settings → Advanced → Virtual memory
   Change. Clear “Automatically manage paging file size for all drives”.
2. Select C:, choose Custom size, Initial1024 MB and Maximum1024 MB, then Set.
3. Select D:, choose System managed size, then Set. Leave other volumes unchanged.
4. In Advanced → Startup and Recovery Settings, retain Small memory dump and the
   existing minidump directory. Apply all dialogs and reboot after saving work.

The bounded1GiB C: file comfortably supports the currently selected small dumps.
It does not guarantee kernel/automatic/complete dumps. If those are required later,
review adequate boot-volume backing or a dedicated dump file on D: separately;
changing only the final MEMORY.DMP output path does not establish dump backing.
Microsoft documents the [dump setting mapping and boot-volume requirement](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/configure-system-failure-and-recovery-options)
and [commit-dependent pagefile sizing / dedicated dump files](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows).

Post-reboot checks (not yet performed):

1. Run `scripts/instruction_pagefile_audit.ps1 -Label after-reboot` in native
   PowerShell and compare LastBootUpTime with the before-reboot audit. Verify both
   intended files in Win32_PageFileUsage, not just requested settings/registry:
   C approximately1024MiB, D active system-managed, no temporary fallback file.
   Verify Win32_PageFileSetting/registry agree with C1024/1024 and D system-managed.
   Check crash setting remains3 and disks remain healthy, online and writable.
2. Run `D:/Apps/miniconda/envs/embodiedops-robocasa/python.exe -B
   scripts/instruction_reboot_preservation.py --verify`. This independently checks
   the junction/token, native GetDiskFreeSpaceExW/GetDiskFreeSpaceW/shutil capacity,
   and all1,586 inventoried evaluation files, including21 valid cells and both
   interrupted attempts. It grants no permission to run evaluation.
3. Observe idle RAM/commit/free-space trends for at least60 seconds with no model
   or simulator launched. Check there are no orphaned workers or temporary pagefile
   warnings. Confirm free capacity remains comfortably above the unchanged2GiB
   stop floor. Budget growth on D against boot-wide31.51GiB peak commit plus an8GiB
   planning margin: about24GiB total pagefile backing beyond available physical RAM,
   with additional room for existing artifacts and remaining evaluation outputs.
   This is a conservative capacity estimate, not proof of future peak demand and
   not a change to experiment thresholds. Do not treat the initial allocation of
   a system-managed D: file as its maximum capacity.
4. Report actual settings/headroom and any uncertainty to the user. Only after
   explicit approval, use a new ledger-aware continuation to skip all21 valid cells,
   replace B-ID-45 in a fresh directory and run the other38 unrun cells. Preserve
   A-NL-45 and B-ID-45 interrupted directories. The old resume driver is hardcoded
   for18 cells and must not be rerun. Use serial episodes and the unchanged native
   active/between-episode2GiB checks, with a monitor bound to the new verified PID.
   Stop on genuine storage, CUDA, camera or checkpoint errors; no automatic retries.

All frozen weights, processors, instructions, controller, renderer, RNG, seeds,
success criteria and thresholds remain unchanged. No training, extra seeds or
automatic commits. Reboot checks and renewed approval are pending.

## Historical measured status — native storage stop, 21 valid cells

The continuation stopped at 2026-10-22 15:50:14 UTC because native DriveInfo
reported C: 1,741,422,592 bytes (1.62 GiB), below the unchanged 2 GiB reserve.
D: had 199,189,495,808 bytes (185.51 GiB). The exact driver and its rollout child
were stopped. No retry, further inference, training, new checkpoint or commit.

Three additional valid cells completed: A-NL-45, A-ID-45 and B-NL-45, all task
failures at 600 actions. B-ID-45 is invalid/incomplete after 12 recorded actions,
with no summary; it is not counted as a completed task failure. Current ledger:
21 completed, 1 invalid, 38 unrun. The original zero-action A-NL-45 attempt remains
separate. All 1,305 pre-continuation evaluation files and 994 historical files
passed SHA256 preservation checks, as did all frozen sources and model weights.
Both execution processes were absent after shutdown.

After shutdown at 15:52:30 UTC, C: recovered to 11,883,188,224 bytes (11.07 GiB).
Pagefile allocation subsequently measured 4,629 MiB. The 2,550 → 9,123 → 4,629 MiB
allocation sequence establishes transient pagefile growth, but does not account
for every byte of C: pressure. A safe restart cannot be inferred from idle free
space. No Windows pagefile settings, dependencies or additional files were changed.

### Partial physical results; not a full-matrix conclusion

| Model | Task-ID success / valid | ID single / double | NL success / valid | NL single / double |
|---|---:|---|---:|---|
| A pretrained | 1/4 | 1/2; 0/2 | 0/4 | 0/2; 0/2 |
| B original NL step300 | 1/3 | 1/2; 0/1 | 0/4 | 0/2; 0/2 |
| C task-ID step300 | 0/3 | 0/2; 0/1 | 0/3 | 0/2; 0/1 |

Coverage is unequal: compare all six arms only on scenes42–44. On those common
scenes, ID success is A1/3, B1/3, C0/3; NL success is zero for all three. The only
success scene is44: A-ID closes in304 actions (15.20 simulated seconds), B-ID in309
(15.45 seconds); C-ID reaches12.24 degrees at600 actions and fails. No tuned arm
has a newly successful scene over A in completed matched evidence.

Historical NL evaluation recorded A0/10 versus B3/10 (B successes44,47,49).
Current B-NL-44 fails at7.69 degrees, so that particular historical success does
not reproduce under this shared current protocol. Scenes47/49 are unrun: the
historical aggregate improvement cannot yet be confirmed or rejected.

For double-door scene45, final degrees are A-NL[85.46,81.79],
A-ID[85.46,7.03], B-NL[85.46,64.52]. The first door barely moved (0.127 degrees)
in all three. Loaded fridge contact occurred in4,148,47 actions, with peak normal
forces53.51,282.53,223.70 N respectively. Progress/contact on one door did not
satisfy the two-door task predicate. Across completed cells, tuned policies can
show larger door progress and substantial contact without task success; this
does not establish reliable manipulation improvement.

Completed workers total2,755.644 seconds (45.93 minutes), including579.204 seconds
for the three newly completed cells. Instrumented inference totals151.583 seconds;
mean chunk latency by arm is approximately0.609–0.639 seconds. These exclude the
interrupted worker and orchestration/audit overhead, and are not GPU-active time
or energy measurements. Successful A-ID/B-ID44 workers took67.922/71.469 seconds.

Offline first-chunk analysis found exactly identical raw RGB/state for A/B/C
NL seed44. Their predictions differ: A–B per-dimension RMSE for goal mode is1.105,
and for translation command coordinates[0.278,0.092,0.104]. These are public action
values before controller clipping/scaling, not achieved displacement. All three
NL rollouts failed. Other compared observations have explicitly recorded sparse
pixel differences; they are not claimed as exact fixed-observation comparisons.
No additional fixed-state inference or demonstration-target experiment was run
after the storage stop. Refill seeds/sampler are shared, but noise bytes were not
independently fingerprinted. Historical RNG/renderer differences, sparse allowed
camera rounding and known scene43 runtime sensitivity preclude bitwise historical
reproduction claims. No learning conclusion is drawn from changed predictions.

Detailed artifacts under `attempt02/evaluation/resume-20261022`:
`report.md`, `all-60-cells.csv`, `ledger.json`, `physical-phase-windows.json`,
`physical-comparison.json`, `existing-first-chunk-comparison.json`, `stopped.json`
and `preservation.json`. C's optimizer/scheduler serialization remains incomplete;
independent inference reload and fixed-loss reproduction passed before evaluation.

## Storage investigation and authorized continuation — 2026-10-22

The user explicitly authorized retaining the 18 valid episodes and completing only
the remaining 42 cells. The zero-action A-NL-45 attempt remains preserved; its
replacement uses `A-NL-45-resume1`. No new seeds, training, checkpoint or tuning.
Continuation metadata and the explicit completed/unrun/invalid ledger are under
`attempt02/evaluation/resume-20261022`; previous reports and results remain intact.

The original driver checks `shutil.disk_usage(ROOT).free` on repository C: and
`shutil.disk_usage(out).free` through the evaluation junction to D:, requiring
strictly more than 2,147,483,648 bytes. Its saved checks passed. The interruption
was a manual stop following a separate external C: reading around 0.4 GiB, not
a recorded exception from the driver's own checks. No synchronized independent
API samples exist for that transient reading, so its cause remains unresolved.

At 15:35:42–46 UTC, native .NET DriveInfo, CIM Win32_LogicalDisk and Get-PSDrive
agreed on C: 14,101,172,224–14,101,630,976 bytes and D: 199,228,194,816 bytes.
Native and sandbox Python samples also agreed across shutil, GetDiskFreeSpaceExW
and GetDiskFreeSpaceW. These are multiple API checks, not independent physical
sensors. The junction and its token resolve to
`D:/EmbodiedOpsOutputs/instruction-control-v1-evaluation`.

Native TEMP/TMP is `C:/Users/Alan/AppData/Local/Temp`; the sandbox uses its own
package AC/Temp. Native TEMP inventory found 239,487,422 bytes, NVIDIA GLCache
38,883 bytes, DXCache 3,938,548,736 bytes and D3DSCache 10,650,124 bytes. The latter
two caches' newest inspected timestamps precede this continuation. The C: pagefile
was allocated 2,550 MiB, with 772 MiB current / 1,237 MiB peak usage. These current
snapshots do not establish the earlier transient's cause or exclude deleted
temporary files. No additional cache or historical files were deleted/migrated.

Rollout manifests confirm HF cache remains the pinned C: cache with offline
loading, while Numba cache and new experiment outputs resolve through the D:
junction. GPU driver and process temporary files can still use C:. The new
orchestrator preserves the 2 GiB floor on both volumes and records timestamped
Win32/shutil readings before and after every episode and after every scene.
Any API reading below that floor stops continuation; no automatic retries.
Audit JSON is in `outputs/instruction-storage-guard-audit`.

Active-worker follow-up: at 15:42:29 UTC all native Python/Win32 APIs measured
C: 4,614,569,984 bytes, followed by 13,349,498,880 bytes after the first worker
exited at 15:42:36. Native CIM at 15:43:38 reported pagefile allocation 9,123 MiB
(previously 2,550 MiB), establishing pagefile growth as a substantial contributor
to current storage pressure. Sampling times differ; this does not fully attribute
the entire drop or retrospectively explain the earlier 0.4 GiB reading.
A separate native .NET watcher samples both drives every five seconds during
active workers and stops this exact driver/child on the unchanged 2 GiB floor.
Its append-only telemetry is `active-native-watch.jsonl` in the audit directory.

The explicit continuation authorization permits one fresh replacement of the
preserved zero-action attempt: at most 60 valid matrix cells, with the old
interrupted attempt separately retained. It does not authorize automatic retries
of any new failure. This supersedes the original no-retry attempt-count wording
below only for A-NL-45.

## Historical status before authorized continuation — 18 episodes

18 episodes completed: all six A/B/C x ID/NL conditions on seeds42,43,44.
The seed42 execution preflight passed all registered contracts; the next two
six-condition batches also passed. Attempt19, A-NL seed45, was interrupted
during initialization, before a recorded reset or action. There are41 unattempted
cells. No retry, training replay, confirmation stage or final120–123 execution.
The intended60-episode comparison is **incomplete**.

| Checkpoint | ID success | NL success | ID single-door | ID double-door |
|---|---:|---:|---:|---:|
| A pretrained | 1/3 | 0/3 | 1/2 | 0/1 |
| B original NL300 | 1/3 | 0/3 | 1/2 | 0/1 |
| C task-ID300 | 0/3 | 0/3 | 0/2 | 0/1 |

All successes were seed44 ID: A304 actions, B309. C ended at12.245degrees and
failed. B's338.80N fridge contact peak exceeded A's88.55N; C's peak was420.03N.
In C's last100 actions,97 had loaded fridge contact but the door reopened1.20
degrees net. This supports a remaining ineffective-contact/terminal-motion
hypothesis, not an established torque or geometric cause. On seed43 B-ID,
the last100 actions had no loaded fridge contact and essentially no closure;
last loaded productive door-motion action488. No fixed-state cross-model action
analysis was run after the resource stop, so no causal learning mechanism is claimed.

Seed42 ID final left/right degrees: A75.801/36.540, B87.265/84.565,
C18.981/42.584; all failed. C made better summed partial closure than A but
increased fridge peak force168.74->379.76N. Seed43 ID final single-door degrees:
A76.255, B59.991, C51.339; all failed. Partial progress does not establish repair.
The original B-NL seed44 success did not reproduce here (7.690degrees, failure),
but the original ten-scene0/10->3/10 claim cannot be assessed from three scenes.

Completed-worker wall time2176.44s (36.27min); total chunk inference128.868s.
Per-episode mean chunk latency0.585–0.671s. These costs exclude the interrupted
attempt and reporting, and are not active GPU-kernel time or energy. Full per-door
progress, EEF path and trajectories, contacts, self-contact peaks, mode/gripper
counts, latency and worker time remain in the complete18-row metrics and60-cell CSV.

Storage-stop limitation: external monitoring during C-NL42 reported C free space
about0.4GiB, below the2GiB reserve. The first narrowly targeted stop request found
that worker already finished; while tool approval/execution was pending, further
registered episodes completed. A later explicit stop terminated the driver and
then its active A-NL45 worker. No instantaneous-stop guarantee was met. Native
batch-boundary records all showed over11.9GiB free on C; after stopping, C reported
about13.14GiB free. The transient/context-dependent discrepancy is unresolved;
do not claim a proven physical disk exhaustion, pagefile cause or sandbox quota.
No actual model/CUDA/camera exception was recorded in the18 completed episodes.

All994 inventoried historical files and all frozen A/B/C weight hashes, including
the original C source model, were preserved. New outputs occupy about1.06GiB on D.
No newly successful fine-tuned-versus-A scene exists among completed pairs; no
before/after improvement video is claimed. All actual rollout videos are retained.

The primary task-ID subset shows no success gain for B and a lost success for C.
Do not promote this to a final negative result for the unfinished ten-scene matrix.
Recovery data is a plausible direction for missed contact and failed terminal
closure, but a final recovery-data/RL/world-model decision remains deferred until
the registered evaluation is completed with reliable resource monitoring.

Authoritative new artifacts under `D:/EmbodiedOpsOutputs/instruction-control-v1-evaluation/`:
`report.md`, `all-60-cells.csv`, `comparison-summary.json`,
`physical-phase-windows.json`, `stopped.json`, `preservation.json` and per-episode
directories. The repository junction retains path compatibility.

## Authorized evaluation continuation — 2026-10-22

The user approved inference-only continuation, explicitly prohibited training
replay, approved exact audited pip-cache deletion and new-output storage on D:.
Exactly1008 validated pip-cache files (3,758,328,388bytes) were deleted; no Conda
archives or experiment files were deleted. Free C space after cleanup11.98GiB.
A first precheck stopped before deletion because UTC epoch conversion was wrong;
corrected DateTimeOffset timestamp checking passed before the actual deletion.

The new-only repository path `attempt02/evaluation` is a verified Windows
junction to `D:/EmbodiedOpsOutputs/instruction-control-v1-evaluation`. Target and
logical-path readback agree. No historical path or artifact was migrated.
Storage and cleanup records are `attempt02/evaluation-storage.json` and
`outputs/instruction-storage-audit/pip-cleanup-result.json`.

The runner uses an independent inference-C bundle: byte-identical verified
step300 weights plus original processor bytes. Inference verification passes;
optimizer/scheduler completeness remains false. The complete seed42 six-condition
scene is the execution preflight; remaining42–51 matrix entries proceed in
six-episode scene batches only after integrity checks. Check C and D free space
before each episode and at batch boundaries; retain2GiB headroom on each.
Freeze source/checkpoint hashes and instruction mapping before first simulation.
Each fresh A-NL reset is also checked against its historical scene reset; all
other arms are paired to current A. No runtime, CUDA, camera or storage retry.
The original60-attempt,600-action/600-second per-episode and10-hour total bounds
remain. Final120–123 and additional20-seed confirmation are excluded.

Earlier status paragraphs below describe the preceding stopped state and audit.

2026-10-22. Status: C completed300 updates; training-state serialization failed,
but the saved step300 policy now passes independent inference-only verification.
At the user's request, only storage audit and fixed-probe reload were performed;
training recovery and RoboCasa evaluation remain stopped. Zero new RoboCasa
episodes have run. Public README, demo and CV remain unchanged. The overall
definition of done has not been met.

## Storage audit and inference-only reload addendum

The user's requested follow-up independently loaded all500 saved step300 tensors
and matched their audited hashes. All eight fixed training and eight held-out
losses reproduced exactly with both the saved processors and original processors:
maximum absolute difference0, train0.3419970777, held-out0.5918905511. The worker
took36.875s, created no optimizer and ran no backward pass or simulator. Original
checkpoint files remained unchanged. Evidence:
`outputs/instruction-storage-audit/reload/result.json`.

**Recommendation superseding the earlier recovery proposal: do not replay
training merely to enable inference evaluation.** Missing optimizer/scheduler
state prevents faithful training resume, but is not required to use these verified
weights for inference. A future evaluation should explicitly accept an
inference-verified artifact, with original processor bytes in a separate bundle,
rather than falsely marking the full training checkpoint complete. No bundle or
robot evaluation was created in this follow-up.

Audit-time free space: C8.493GiB, D186.608GiB. Repository unique file contents
54.240GiB; path-summed60.480GiB includes linked entries. Protected checkpoint
directories total40.808GiB. Review-only cleanup candidates: user pip cache
3.500GiB on C, downloaded Conda archives0.209GiB on D. No deletion occurred.
No demonstrably incomplete temporary checkpoint payload was found. A5GiB free
target allows a copied inference bundle,2GiB estimated matrix artifacts and2GiB
headroom; actual output size varies. The prior disk-full error's cause remains
unverified. Full inventory, exact cleanup paths/sizes and estimates:
`outputs/instruction-storage-audit/{report.md,cleanup-proposal.csv,storage.json}`.

## Measured training result and blocker

B independently reloaded all500 model tensors, original processor bytes and
its exact eight validation losses (mean0.5941866031). C's smoke checkpoint passed
the same independent reload checks. All300 C updates completed with finite
gradients in all155 intended tensors, using B's exact anchors, episode IDs and LR
sequence. Torch CPU/CUDA RNG states match B at saved steps1,150,300. Endpoint
RNG equality supports the retained protocol; it is not a logged per-update noise
fingerprint or a bitwise GPU determinism guarantee.

| C step | Fixed training loss | Fixed held-out loss |
|---:|---:|---:|
| 0 | 0.277560 | 0.448321 |
| 150 | 0.424329 | 0.571974 |
| 300 | 0.341997 | 0.591891 |

These probes use task-ID conditioning. B's historical probes use NL, so their
raw losses are not an instruction-controlled cross-model comparison. No robot
improvement, physical mechanism or research-direction decision follows yet.

The continuation worker took179.906s and reached step300. During final
`save_checkpoint`, saving optimizer tensors raised Windows I/O error112
(not enough disk space). A subsequent filesystem query reported6.44GiB free;
the discrepancy's cause is unverified. Do not label it a proven sandbox quota.
No automatic training replay or simulation followed.

Step300 contains all500 finite model tensors with expected names/shapes/dtypes:
122 intended trainable tensors changed,33 trainable tensors stayed unchanged,
and every frozen tensor matches A. Model file SHA256:
`e3cc9a2084f081778b2353bfc8e8c9d027e933e224eca37205be2d2c9715a3c8`.
Optimizer tensors/groups and scheduler state are absent. The runner's final
in-memory verification target and original-processor-byte restoration did not
execute. Processor JSON semantics and active state/action mean/std values match;
the raw reserialized preprocessor statistic file differs in other metadata.
This is not a fully verified or resumable step300 checkpoint. The intact step150
checkpoint is retained. All994 inventoried historical files remain hash-identical.

Prepared recovery (`scripts/instruction_control_recover.py`, not executed):
independently verify step150, replay151–300 into a new directory, require final
model-file SHA256 and all fixed step300 probe losses to match the failed-save
attempt exactly, then independently reload the recovered checkpoint. It never
trains past300, but would add150 executed updates (450 total compute), so explicit
approval was requested against the user's300-update budget. Failure of equality
must stop recovery; do not substitute a different candidate.

The matrix runner and raw-replan observation capture are prepared but unexecuted.
Two existing contact-instrumentation tests passed outside the sandbox after
temporary-directory access failed inside it; syntax checks passed for all seven
changed/new study scripts and `git diff --check` passed. No full matrix execution
test, fixed-observation prediction comparison, new video or success table exists.
No dependencies changed and no commit was made. Original untracked follow-up
audit files were left untouched.

Evidence: `outputs/agent-workflow-v1/instruction-control-v1/attempt02/`
contains `B-reload-verification.json`, `partial-training-audit.json`,
`serialized-C-tensor-hashes.json`, `preservation.json`, worker logs and C's
original checkpoints/probes. The earlier cache-path failure remains in the parent.

## Registered comparison

A is pinned pretrained SmolVLA (`f7af588d11e30b0069f200792fb285904b74c80b`),
SHA256 `96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b`.
B is original NL step300, SHA256
`0e011aa494e0d4e370b11feb5eec6e0c49af2fadd1f3f97ee03056909de0fe6d`.
C starts independently from A and receives exactly 300 updates. Its only learning
intervention is replacing raw `task` with `CloseFridge` before the original saved
newline/tokenization/normalization pipeline. This identifier is a conditioning
proxy, not verified upstream pretraining reproduction.

The original 461/52 episode split and 300 anchors are asserted equal, with batch1,
native BF16/FP32, frozen VLM, 99,880,992 trainable parameters, AdamW LR1e-4,
betas(0.9,0.95), epsilon1e-8, weight decay1e-10, clipping10 and the native
15/300 warmup/decay schedule. The native masked Flow Matching objective retains
50x12 actions padded to32, Gaussian noise and transformed Beta(1.5,1) times.
The smoke-step/save/independent-reload/resume sequence is preserved. C's step1
Torch CPU/CUDA RNG state must equal B's saved step1 RNG state before continuation.
Fixed eight-anchor train/validation probes use seeds42000–42007 at0/150/300;
intermediate historical probe cadence is retained to preserve execution order.
Step300 is primary regardless of loss or robot success. No precision correction,
longer training, alternate sampler, checkpoint selection or second candidate.

Historical B's file hash, all500 tensor hashes, four processor files and all300
logged anchors passed a fresh read-only audit. Original artifacts are inventoried
for preservation. Independent model loading and fixed validation are a further
gate. A first reload failed before model execution on a Windows cache lock path
over260 characters; the attempt is retained. A shorter generated cache under
`outputs/ic-cache` addresses filesystem layout only, without dependency changes.

## Robot protocol and resource bounds

Exactly ten diagnostic scenes42–51, three checkpoints and two instructions:
primary literal `CloseFridge`; secondary each scene's original native singular
or plural instruction. Freeze C's hash before simulation. Run all60 fresh episodes;
historical trajectories lack the current full contact/RNG contract. Per scene,
order A-NL, A-ID, B-NL, B-ID, C-NL, C-ID. A-ID checks reset against A-NL;
B/C check against A in the same instruction. Retain exact reset qpos/qvel/state,
scene and controller equality and existing sparse one-level camera tolerance.

Use existing owner-context renderer cleanup, original three256px cameras,16D
state and12D hybrid mobile/OSC control; no privileged policy input. H50 queue,
10 flow steps,600 actions,20Hz,25 physics substeps/action. Set Torch RNG to
scene_seed*100000+next_action_index before every native chunk refill. Record full
action chunks and raw observations at replans without modifying predictions.
This current protocol is shared by all arms, not bitwise historical reproduction.
The known seed43 discrepancy and any new numerical differences remain reportable.

Expected matrix cost roughly2–4 hours plus training/reload and fixed-state analysis;
hard cap600 seconds per episode and10 hours total worker time. Training/reload
has a separate30-minute total bound. At most60 new episode attempts, no automatic
episode retries, no seeds120–123. Stop on invalid camera, contract mismatch,
exception, runtime cap or disk reserve below2GiB. Guard-stopped robot episodes
remain visible as failures and physical regressions, never silently replaced.
Estimated checkpoint storage3.96GB plus approximately1–2GB logs/videos/observations.
No installation, environment repair, automatic commit or public-artifact editing.

Report actual task predicate, per-door initial/final/best angles, action count,
EEF trajectories, loaded robot–fridge contacts, peak forces by contact category,
self collisions, gripper/mode commands, inference and worker time for every pair.
Physical comparison uses existing progress allowance1degree and force ceiling
1.25*reference+25N as descriptive regression gates, not hardware safety claims.
Success counts are stratified by one/two relevant doors. Do not treat these
development scenes as population or generalization evidence.

## Fixed-state interpretation and decision

Compare A/B/C with both instruction conditions, identical RGB/state and fixed
noise, including historical successful B-seed44 and failed B-seed43 snapshots
and demonstration targets. Approach/contact/terminal labels require trace or
demonstration evidence; action changes alone are not improvements. Public actions
0:4 are base/torso,4 is achieved/desired goal mode,5:8 base-frame OSC translation
increments (controller clips and scales by0.05m),8:11 rotation increments (0.5rad),
11 gripper thresholded at0.5. They are commands, not guaranteed motion or torque.
Compare demonstration error only where aligned targets exist, masking padding.

If neither tuned model improves primary task-ID success, preserve the negative
finding and stop. A credible positive finding may justify proposing20 untouched
development seeds plus sensitive-scene repeats, but those runs require review
and are not authorized here. No recovery-data collection, residual RL or world
model training is part of this study; recommend among them only from evidence.
