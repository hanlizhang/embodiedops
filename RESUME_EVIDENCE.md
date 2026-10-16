# Resume Evidence

This file tracks what has actually been implemented and verified in EmbodiedOps.

Do not convert planned work into resume claims.

## Verified setup / pipeline

- [x] NVIDIA RTX 3060 CUDA PyTorch environment
- [x] Hugging Face LeRobot 0.6.1
- [x] PushT simulation and RGB rendering
- [x] Diffusion Policy forward/backward training on GPU
- [x] Diffusion Policy checkpoint saving
- [x] PushT policy evaluation
- [x] PushT rollout-video generation
- [x] MuJoCo 3.3.7
- [x] robosuite 1.5.2
- [x] Panda Lift simulation

## Not yet implemented

- [ ] Full / meaningful Diffusion Policy baseline training
- [ ] Quantitative baseline benchmark across multiple episodes / seeds
- [ ] Robustness / perturbation evaluation suite
- [ ] Failure taxonomy
- [ ] Agent tool interface
- [ ] NVIDIA / Nebius reasoning agent integration
- [ ] Autonomous experiment-selection loop
- [ ] Policy-improvement / bounded retraining loop
- [ ] Before / after improvement benchmark
- [ ] SmolVLA integration
- [ ] pi0 integration
- [ ] RoboCasa integration

## Resume claim status

Current status: environment and robot-policy pipeline validation only.

Do **not** yet claim that EmbodiedOps autonomously diagnoses or improves robot policies.
