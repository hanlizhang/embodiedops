# Resume Evidence

## Verified integrated workflow — 2026-10-19

This dated entry supersedes the old unverified checklist only for these measured
capabilities; it does not claim LLM reasoning or an accepted policy improvement.

- Implemented an executable deterministic evidence/tool loop with a registry,
  bounded subprocess adapters, versioned candidates and independent rejection gates.
- Ran four new matched SmolVLA/RoboCasa episodes. Instruction repair changed
  diagnostic success from 0/2 to 1/2 but was rejected for door-progress/contact-force
  regressions. Reserved final seeds were not used.
- Executed 512 mixed-instruction optimizer updates on 512 distinct anchors across
  461 training episodes. All intended gradients were finite; 122 tensors changed,
  frozen tensors stayed unchanged. A separate process verified the selected
  checkpoint's 500 model tensors, optimizer, scheduler, processors and validation.
- Rejected that learning candidate: held-out mean loss worsened 0.384626 -> 0.427864.
  No robot-performance claim is made for it. Original checkpoints were hash-preserved.
- Compared five adaptive offline probes with six fixed-order probes; no wall-time
  or GPU-cost advantage was established. External reasoning backend is implemented
  but untested and unused. The pre-update Windows cache-path recovery was operator
  assisted, not an agent achievement.
- Evidence: `environment/agent-workflow-v1.md`, `scripts/agent_workflow.py`,
  `tests/test_agent_workflow.py`, and `outputs/agent-workflow-v1/run-01/audit.json`.

This file tracks what has actually been implemented and verified in EmbodiedOps.

Do not convert planned work into resume claims.

## Verified setup / pipeline

- [x] NVIDIA RTX 3060 Laptop GPU available for CUDA workloads
- [x] PyTorch 2.11.0+cu128 with CUDA enabled
- [x] Hugging Face LeRobot 0.6.1
- [x] PushT simulation and RGB rendering
- [x] Diffusion Policy forward/backward training on GPU
- [x] Diffusion Policy checkpoint saving
- [x] PushT policy evaluation
- [x] PushT rollout-video generation
- [x] MuJoCo 3.3.7
- [x] robosuite 1.5.2
- [x] Panda `Lift` simulation

## Not yet verified

- [ ] RoboCasa integration
- [ ] SmolVLA baseline evaluation
- [ ] SmolVLA fine-tuning / post-training
- [ ] CloseFridge success-rate benchmark
- [ ] VLA inference-latency benchmark
- [ ] Controlled fault injection
- [ ] Diagnostic probe library
- [ ] NVIDIA / Nebius reasoning-agent integration
- [ ] Agentic root-cause diagnosis
- [ ] Automatic repair routing
- [ ] Targeted VLA post-training triggered by diagnosis
- [ ] Before/after repair benchmark
- [ ] pi0 integration
- [ ] RL-based policy improvement

## Resume claim status

Current status: environment, simulation, and robot-policy pipeline validation only.

It is accurate to say that the project has:
- run LeRobot-based Diffusion Policy training and evaluation on CUDA,
- generated simulation rollout videos,
- established a MuJoCo/robosuite Panda manipulation environment.

Do **not** yet claim:
- SmolVLA/VLA fine-tuning,
- RoboCasa policy training,
- agentic diagnosis,
- autonomous policy repair,
- measured policy improvement,

until those results are actually implemented and verified.

## Update rule

After each milestone:
1. add only facts supported by actual runs / artifacts,
2. include the measured metric or artifact where possible,
3. keep planned work in the "Not yet verified" section,
4. do not promote planned features into resume bullets before verification.

## Verified control evidence — 2026-10-19

This dated entry supersedes the earlier baseline/RoboCasa "not yet verified"
status for the narrowly measured work below; it does not claim an agent or
generalization improvement.

- Ran three pretrained SmolVLA/CloseFridge H20 episodes with fixed seeds 42–44,
  50-action predictions, saved processors and task-ID instructions. One succeeded,
  one reached the 600-action cap, and one stopped at the inherited motion guard.
- Audited/reused three historical H50 episodes: also one success. Prefix numerical
  differences and missing historical contact logs limit causal interpretation.
- Implemented per-physics-substep MuJoCo contact-force logging, physical trajectory,
  replan continuity and timing analysis without adding simulator state to policy
  inputs. Native queue/force preflight and two instrumentation tests passed.
- Evidence: `environment/control-horizon-v1.md`, its results JSON, and
  `outputs/control-horizon-v1/verification.json`. No measured success-rate gain,
  new training, autonomous diagnosis or repair claim follows from this experiment.

## Sustained VLA learning evidence — 2026-10-20

- Completed two independent 5,000-update SmolVLA fine-tunes from pinned original
  weights using 461 demonstration episodes, unchanged normalization and native
  masked flow matching. Compared episode-balanced sampling with a motion-proxy
  mixture; these inferred regimes are not verified contact labels.
- Used identical 104 validation anchors from 52 episode-disjoint episodes.
  Selected step5,000 in both arms by held-out task-ID loss: pretrained0.369275,
  uniform0.367256, motion0.370945. All ten saved checkpoints passed independent
  model/optimizer/scheduler/processor and validation reload checks.
- Executed new trained-policy RoboCasa rollouts. On valid matched seeds43–45:
  pretrained1/3, uniform1/3, motion0/3 success. Uniform increased door progress on
  seeds43/45 but regressed in seed44 peak contact force; motion lost seed44 success
  and regressed in contact force and door progress. Existing acceptance gate
  rejected both on the observed common scenes; no measured success gain.
- Ten valid episodes completed. Episode11 (uniform seed46) stopped after one
  action when all raw observation cameras became zero. The guard prevented a
  prediction on black images. Final seeds120–123 were not run. Full train-and-final-
  verification milestone remains incomplete; the camera cause is unresolved.
- Peak training CUDA allocation1.732GiB; serial GPU-worker wall-time2.269hours
  including CPU/I/O, not active kernel time. Energy not measured. All protected
  historical checkpoint hashes unchanged; no dependencies changed.
- Evidence: `environment/learning-repair-v1.md`, concise results JSON, and
  `outputs/agent-workflow-v1/learning-5000-v1/{trace.jsonl,partial-acceptance.json,audited-summary.json}`.
  This supports actual VLA fine-tuning and partial closed-loop repair verification,
  not accepted autonomous repair, official benchmark reproduction or robust
  generalization. Earlier broad "not yet fine-tuned" statements are historical.

## 2026-10-20 — Learning repair diagnostic evaluation completed

- Reproduced all-zero RGB at MuJoCo readback after garbage collection; validated
  owner-context renderer cleanup before policy evaluation. Exact historical
  collection timing remains inferred. Three cleanup unit tests and all 15
  lightweight tests passed.
- Executed only the two missing seed-46 candidates: both failed after 600 actions
  with valid cameras and no physical guard failures. All 12 diagnostic pairs
  are complete: pretrained 1/4, episode-balanced 1/4, motion-mixture 0/4.
- Both repairs rejected; final 120–123 not run because diagnostic gates failed.
  Better partial door closure does not establish better task success.
- Preserved all 808 original experiment files by hash; invalid attempt indexed
  in place and excluded. No retraining, model changes, dependency changes or commit.
- Evidence: `environment/learning-repair-v1-completion.md` and
  `outputs/agent-workflow-v1/learning-5000-resume-v1/completion-audit.json`.

## 2026-10-20 — Controlled FP32-master training completed; verification blocked

- Implemented differentiable FP32 masters and Adam moments with native BF16/FP32
  forward weights; initial hashes, samples, noise, schedule and processors match
  the archived episode-balanced run. All 17 lightweight tests passed.
- Preflight independently reloaded model, masters, optimizer, scheduler and
  processors, reproduced validation and the exact next update. One sustained
  5,000-update run then completed with finite gradients and unchanged frozen weights.
- All 33 previously unchanged normalization tensors changed in actual BF16
  forward weights. This resolved measured update accumulation, but validation
  worsened to 0.387121 versus native 0.367256; early deterioration also increased.
- Selected step 5,000 using the registered validation rule. All five serialized
  checkpoints passed CPU hash checks. Independent sustained-checkpoint reload
  timed out at step 1,000 after 300 seconds; later reloads and robot evaluation
  were not executed. No robot improvement or completed acceptance decision claimed.
- Training worker 132.70 min; peak CUDA allocation 2.648 GiB. No valid energy
  integral. All 957 protected historical files preserved. No commit or dependency
  change; final seeds 120–123 untouched.
- Evidence: `environment/optimizer-precision-v1.md` and
  `outputs/agent-workflow-v1/optimizer-precision-v1/audited-summary.json`.
