# Technical overview

EmbodiedOps connects SmolVLA optimization to physical execution and rejects
repairs that fail registered criteria. The executed backend is a bounded
deterministic evidence/tool loop, not autonomous LLM reasoning.

## Policy and data contract

| Item | Executed contract |
|---|---|
| Policy | `lerobot/smolvla_robocasa`, revision `f7af588d11e30b0069f200792fb285904b74c80b` |
| Backbone | SmolVLM2-500M-Video-Instruct, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Dataset | `pepijn223/robocasa_CloseFridge`, revision `b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4` |
| Data | 513 episodes, 155,443 frames, 20Hz; 461 train / 52 validation |
| Split | Last ceil(10%) episodes within each natural-instruction stratum; no shared local episodes |
| Cameras | left / right / wrist, mapped by saved processors to camera1 / camera2 / camera3 |
| Images | raw HWC uint8 256×256×3 → CHW float32 [0,1]; model resize/pad 512×512, then scale [-1,1] |
| State | 16D; saved mean/std normalization; padded internally to 32 |
| Actions | 12D; saved mean/std normalization; internal 50×32, public 50×12 |
| Language | Saved newline/tokenizer path, maximum 48 tokens; task-ID `CloseFridge` in sustained arms |
| Inference | 10 flow steps, 50 predicted actions; main evaluation executes 50 then reobserves |

State layout: base position[0:3], base quaternion[3:7], base-relative EEF
position[7:10], base-relative EEF quaternion[10:14] and gripper joints[14:16].
World-frame diagnostic EEF coordinates are separate; moving-base or moving-door
comparisons require the appropriate frame conversion.

Public action[0:4] is base/torso, [4] selects goal-update mode, [5:8] is EEF
translation, [8:11] rotation and [11] gripper. The wrapper reorders this into
arm/gripper/base/torso/mode for the hybrid mobile controller. OSC translation
clips to [-1,1] and scales by 0.05 m in the base frame; rotation scales by 0.5 rad.
These are target increments, not torques or guaranteed 50 ms displacements.
Gripper values below 0.5 command opening; values at least 0.5 command closing.
Measured gripper joints, contact and door motion must be examined separately.

One public control step is 0.05 s / 25 MuJoCo steps of 0.002 s. The runner imposes a
600-action cap and preserves actual task success before reset. Policy inputs
contain RGB, proprioception and language, not privileged door joints, contact
identities, forces or success labels.
The pinned CloseFridge predicate requires every checked fridge-door joint to have
normalized opening ≤0.005 of its joint range. This is not a 0.005-radian threshold;
partial or summed progress does not establish task completion.
See [Phase1A](../environment/phase1a.md) and [Phase1B](../environment/phase1b.md).

## Learning objective and reproducibility

For normalized demonstration actions `a`, Gaussian noise `ε` and sampled time `t`,
the pinned implementation uses `x_t = t ε + (1−t) a` and velocity target `ε−a`.
Training uses native Flow Matching squared error over real action dimensions
with padded timesteps masked. Inference integrates from noise toward actions
in ten steps. This is demonstration imitation, not RL or world-model training.

The VLM is frozen; the action expert and state projection train. Sustained native
training has 155 trainable tensors / 99,880,992 elements, with 129 BF16 and 26 FP32
tensors. It changes 122 tensors; 33 remain byte-identical despite finite gradients.
The FP32-master ablation addressed accumulation and exported native forward
weights, but worsened optimization.

Each sustained arm independently initializes from original pretrained weights:
5,000 updates, batch 1, AdamW betas(.9,.95), epsilon1e-8, weight decay1e-10,
clip norm 10, 250-update warmup and configured LR 5e-5 decaying to 5e-6.
Per-update LR and Gaussian/time draws are recorded. The native scheduler's
actual maximum used LR was 4.98008e-5.

Each episode-balanced cycle visits every training episode once. A fixed 32-anchor
training probe and 104-anchor global validation probe use fixed noise draws.
Validation anchors are two per held-out episode, correlated within episodes.
Checkpoints at 1000/2000/3000/4000/5000 are selected by lowest global task-ID loss,
earliest tie. No robot outcome selects a checkpoint.

Motion-mixture samples a uniform anchor half the time and a uniformly chosen
nonempty motion-proxy regime otherwise. Terminal-window replaces eligible
zero-based even draws with full 50-action chunks spanning first recorded reward.
Its 2,438 targeted replacements increased valid target rows by 11,739 because fewer
chunks needed padding: a coverage confound, not a pure phase intervention.
Neither sampler provides verified contact or recovery labels.

Independent reloads compare model/optimizer/scheduler state, saved processors
and fixed validation losses. FP32 sustained-checkpoint operational reload remains
unverified after timeout. Terminal content verification passed five checkpoints,
with one late reload; the corrected deadline guard was added after the run.

## Physical evidence and acceptance

The contact recorder observes all 25 physics substeps and records geometry IDs,
constraint activation and contact-frame forces. An active constraint and positive
normal component support loaded contact; proximity alone does not. Peak values
include self-contact and are not net EEF wrench or hinge torque.

Contracts include checkpoint/processor hashes, versions, instruction, reset
qpos/qvel/state, scene metadata, cameras and replan RNG rules. The corrected
renderer lifecycle frees obsolete resources with their own GL context current,
then restores the live context. A controlled probe localized black images first
to readback. Exact historical collection timing remains inferred; invalid episodes
are excluded and preserved.

The shared gate requires strict paired success gain, no lost reference success,
valid matched episodes, no new physical guard failures, no summed door regression
over 1° and peak normal force ≤ 1.25×reference+25N. These are comparative simulator
limits, not hardware safety certification. Terminal-window additionally requires
at least 2/4 successes, repair of 43 or 45, retained 44 success, no individual-door
regression over 1°, and both global/terminal offline criteria. It failed multiple
criteria. Final 120–123 stayed unused.

## Orchestration scope and limitations

`agent_workflow.py` registers observation-contract checks, instruction probes,
trajectory/contact analysis, configuration repair, training and diagnostic/final
evaluation. The deterministic backend branches on returned evidence, saves
decisions/results and stops under bounds. Its initial execution compared five
adaptive offline probes with six fixed-order probes; no compute advantage was
established. Later sustained ablations were human-registered dedicated runs that
reused these components, not autonomous discoveries.

Four diagnostic scenes, one training seed per arm and unknown pretraining overlap
cannot establish generalization. Worker-hours include CPU/I/O; active GPU-hours
and energy were not measured. A technically completed workflow can correctly
reject its candidate. No accepted autonomous repair or official success-rate
reproduction has been demonstrated.
