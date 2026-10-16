# Resume Evidence

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
