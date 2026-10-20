# Resume and interview material

These claims refer to executed simulation work. Use “built and evaluated,” not
“improved success rate.” The deterministic backend and negative outcomes should
remain explicit.

## English

**EmbodiedOps — VLA fine-tuning and manipulation repair verification**

Built a reproducible SmolVLA–RoboCasa pipeline connecting Flow Matching fine-tuning,
MuJoCo contact/trajectory diagnostics and matched-seed closed-loop evaluation.
Executed controlled learning ablations and rejected candidates when offline
optimization did not translate into better physical task completion.

Suggested resume bullets:

- Fine-tuned SmolVLA's action expert and state projection on 461 CloseFridge
  demonstration episodes, using 52 episode-disjoint validation episodes, fixed
  noise/sample ordering and independent checkpoint/processor verification.
- Integrated three-camera RGB, 16D proprioception and 12D mobile-manipulator
  control with per-substep MuJoCo contact logging, door-progress traces and
  observation-integrity guards.
- Executed four controlled 5,000-update runs covering episode-balanced,
  motion-proxy, FP32-master and terminal-window variants; separated numerical
  correctness, held-out imitation loss and actual robot task success.
- Implemented bounded deterministic repair orchestration and regression gates;
  rejected terminal-window fine-tuning despite 3.51% lower terminal held-out loss,
  because diagnostic success stayed 1/4 and door/contact metrics regressed.

For a short resume, keep the first, second and fourth bullets. The FP32 run
completed training but not operational final-checkpoint reload or robot evaluation.

## 中文

**EmbodiedOps — 面向机器人操作策略的 VLA 微调、故障分析与修复验证**

搭建 SmolVLA 与 RoboCasa/MuJoCo 的可复现实验链路，将 Flow Matching 模仿学习、
接触与轨迹分析、固定场景闭环评估及候选策略验收连接起来。通过受控消融区分
数值更新、离线损失与实际任务完成度，保留并分析未通过验收的实验结果。

简历要点：

- 基于461条 CloseFridge 示范轨迹微调 SmolVLA 动作专家与状态投影模块，使用
  52条按 episode 隔离的验证轨迹、固定采样与噪声种子，并独立验证检查点及预处理。
- 对接三路 RGB、16维本体状态与12维移动操作控制接口，实现 MuJoCo 物理子步
  接触记录、门关节进度分析及观测有效性检查。
- 完成 Episode-balanced、运动代理采样、FP32 主权重及终段窗口四组各5,000步
  训练实验；明确区分数值精度修正、验证损失变化与机器人闭环任务表现。
- 构建有预算约束的确定性诊断与修复流程：终段窗口验证损失降低3.51%，但固定
  诊断场景成功数仍为1/4且门角度与接触指标退化，因此依据预注册标准拒绝候选策略。

不写“自主大模型诊断”“世界模型训练”“真机部署”或“显著提升成功率”。
FP32 实验仅完成训练与序列化完整性检查，完整重载超时，未执行机器人评估。

## Main contributions and limits

The engineering contribution is an inspectable path from demonstration tensors
and controller semantics to actual robot outcomes, with reproducible optimization,
physical measurements and explicit rejection. The research contribution is the
controlled evidence: instruction sensitivity, ineffective partial closure, a
numerical-update correction that worsened learning, and a targeted imitation
improvement that failed to repair manipulation.

This is not a new VLA architecture or general-purpose autonomous diagnosis system.
Only four development scenes and one training seed per sustained arm were tested;
final 120–123 were unused. Local validation is episode-disjoint but pretraining
overlap is unknown. Contact forces are simulator solver values, not calibrated
hardware loads. GPU-worker time includes CPU/I/O; energy was not measured.

## Three technical interview questions

### 1. What exactly did you optimize, and how did you establish a real weight update?

**Answer:** I used the pinned SmolVLA Flow Matching objective. For normalized
action chunks `a`, noise `ε` and time `t`, the input is `tε+(1−t)a` and the
velocity target is `ε−a`. Padded timesteps are masked; the 12 real action dimensions
are distinguished from internal 32D padding. I froze the VLM and trained the action
expert/state projection with native AdamW, logging gradients, clipping and actual
parameter differences. Finite gradients alone were insufficient: 33 normalization
tensors stayed identical in native training. FP32 masters eventually changed all 33
in the real BF16 forward representation, yet validation worsened to 0.387121 versus
0.367256. That supports a numerical accumulation limitation but rejects the claim
that fixing it necessarily improves learning.

**Evidence:** [objective/contract](../environment/phase1a.md),
[learning controls](../environment/learning-repair-v1.md),
[negative precision result](../environment/optimizer-precision-v1.md).
Do not say the FP32 final checkpoint passed full independent reload: it timed out.

### 2. Why can terminal imitation loss improve while a robot leaves the door open?

**Answer:** The measured terminal loss is on states from successful demonstrations
and fixed noise draws; it does not test the distribution visited by the policy.
The proposed windows contained recorded reward transitions, not verified recovery
from the policy's residual openings. With matched initialization and training
settings, terminal loss fell 0.307188→0.296408, global loss worsened
0.367256→0.377230, and success remained 1/4. Seed 43 went from 13.11° residual opening
to 90.00°; seed 45 went from approximately 0°/28.27° to 40.76°/31.09°. Exposure to
successful terminal actions did not repair those cases.

Distribution mismatch and missing recovery coverage are plausible explanations,
not established causal mechanisms. Fewer padded chunks also increased valid
supervision by 11,739 action rows. I report that confound and do not label reward
windows as contact/recovery examples.

**Evidence:** [audited selection rule](../environment/terminal-window-repair-v1.md),
[paired results and gate](../environment/terminal-window-experiment-v1.md).

### 3. How do you distinguish a bad policy from a bad interface or misleading metric?

**Answer:** First I check camera tensors, state ordering, saved normalization and
controller conversion. A commanded 0.05 m OSC target increment is not guaranteed
0.05 m achieved movement in 50 ms. I compare frames correctly, and use geometry plus
active positive solver force to identify loaded contact instead of visual
proximity. Then I verify the selected weight hash, matched resets/cameras,
instruction, horizon and noise protocol before comparing task predicates.

A raw-camera fault first appeared at renderer readback; the invalid attempt was
excluded, preserved and replaced only after a controlled owner-context cleanup
test. The task-ID comparison has pretrained 1/4, episode-balanced 1/4,
motion-mixture 0/4 and terminal-window 1/4. These cannot be pooled with the earlier
NL baseline. The acceptance gate rejects lost successes, door/force regressions
and invalid observations; lower loss alone cannot pass it. The deterministic
router applies predefined evidence rules. It did not invent these experiments.

**Evidence:** [controller](../environment/phase1b.md),
[camera completion](../environment/learning-repair-v1-completion.md),
[deterministic loop](../environment/agent-workflow-v1.md).
