# Recovery Feasibility v1b — completed, neither branch recovered

2026-10-23. 本轮按预注册配置先 B 后 A、独立进程各执行一次，完成有效配对。
两路都执行满额外 **300 public actions / 15 秒仿真时间**，均未满足真实 all-door
CloseFridge success predicate。修正后的外部 Recovery Expert 已解决反向 approach
问题并建立门接触，但最终把门推得更开。原始冻结 SmolVLA 有明显部分关门进展，
仍未成功。**不能据此判定状态不可恢复，也不足以支持下一步纠正示范采集。**

## 冻结配置与预算

唯一状态为 `instruction-control-v1 / A-ID-48`，seed 48，历史 action 500 后，
初始门角 34.391893925°。它是已有开发状态，非未见测试集。
原完整任务上限 **600 actions**；本轮是独立 failure-state recovery benchmark，
在历史 500 步之后允许额外 300 步，总 action index 到 800。
不得将此结果称为原始 600-action 任务预算内成功。

A 使用 `lerobot/smolvla_robocasa` revision
`f7af588d11e30b0069f200792fb285904b74c80b`，model SHA256
`96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b`。
原 processor hashes、normalization 和 public action conversion 经检查保持一致。
instruction=`CloseFridge`，10-step Flow Matching，H=50。新加载模型并 `policy.reset()`，
通过显式 chunk 执行不使用任何历史队列。预先固定唯一噪声序列 seeds
931001–931006，每次 `torch.manual_seed(seed)` 后调用原生
`sample_noise((1,50,32), device)`；六个实际 tensor、seed、SHA256 均已保存。
没有筛选候选或重复尝试。

B 原样复用已修正的 `GeometryExpert`：最近表面 approach、12 mm 阈值后的切向推门、
最大 1 cm 目标增量、原 base/orientation/gripper 控制。未调增益或接触策略。
这是外部几何控制程序，**不是 SmolVLA 神经网络内部的 Action Expert**。

计划在物理执行前写入 [registered-plan.json](../outputs/agent-workflow-v1/recovery-feasibility-v1b/registered-plan.json)，
包含配置、源码 hashes、噪声生成方式、终止和验收规则。每路整进程 600 秒限制，
总任务目标 60 分钟；本轮交付在该预算内完成。两路 measured worker sections 合计
240.266 秒，不包含全部进程启动/退出、代码编写和结果整理；不将其称为总任务耗时。

## Horizon 与恢复有效性

检查了 installed LeRobot、RoboCasa Gym wrapper、env factory、robosuite base 和历史 runner：

- 历史 600 上限由 rollout runner 执行；LeRobot `_max_episode_steps` 是元数据，未传入低层。
- 原 RoboCasa factory 设置 `ignore_done=True`，实际低层 horizon=1000；Gym wrapper
  传递 low-level done，truncated=False，并调用真实 `_check_success()`。
- 本轮专用 runner 只 reset 一次，在首个 public action 前将低层 horizon=800、
  ignore_done=False，并设置 LeRobot episode_length=800。没有重置控制器、时间或门状态。
- 每次都保留 done/truncated、真实成功和硬停止处理。两路最后一步都在 index=800
  收到 horizon done，恰与恢复第 300 步一致。因此原始 `stop_reason=simulator_termination`
  表示预算边界正常结束，非提前异常；comparison 的 `stop_detail` 明确解释了这一点。

两路各通过完整历史 500-action 前缀重放；**每步 qpos、qvel、16D state 最大误差均为 0**。
场景、controller config、reset state 和 camera identity 检查通过。
相机 uint8 gate 原样保留：max≤1、MAE≤0.001。所有 reset 图像误差为 0；
B 边界三相机完全一致；A 边界 left max=1/MAE=0.000579834，right
max=1/MAE=0.000172933，wrist 误差为 0，均通过。

通过实际相机检查后，A 第一次推理使用 v1 已审计公共 RGB/state reference；
第 50/100/150/200/250 步后的五次重规划全部使用 A 的当前实际 RGB 和 16D 状态。
离线逐数组验证六次输入与各自规定来源精确相等。
没有用 reference 替代失败检查，也没有要求分支继续匹配历史图像。
所有恢复步观察（每路 301 份，含起点）、动作、物理状态和每子步接触记录均已保存。

## 两路结果

| 项目 | A — Original SmolVLA | B — Corrected Recovery Expert |
|---|---:|---:|
| Restoration validity | 有效 | 有效 |
| Execution status | completed | completed |
| 终止原因 | index 800 horizon / 恢复预算用尽 | index 800 horizon / 恢复预算用尽 |
| 实际恢复 actions | 300 | 300 |
| 恢复仿真时间 | 15.000 s | 15.000 s |
| 真实任务成功 | 否 | 否 |
| 初始角度 | 34.391894° | 34.391894° |
| 最终角度 | 15.916205° | 42.164566° |
| 最小角度（含起点） | 15.916205° | 34.383655° |
| 最终关门进展 | +18.475689° | −7.772672° |
| 首次有效手门接触 | 11.212 s（action 225 内） | 7.624 s（action 153 内） |
| 首次成功 | 无 | 无 |
| Peak all robot contact | 115.123817 N | 78.745787 N |
| Peak fridge contact | 30.748511 N | 78.745787 N |
| Peak gripper-door contact | 30.748511 N | 64.051479 N |
| Peak self-contact | 115.123817 N | 0 N |
| Hard guard | 未触发 | 未触发 |
| Relative-contact gate | 本轮参考 | 通过 |
| Prefix replay wall time | 40.547 s | 44.235 s |
| Inference wall time | 4.141 s（6 次） | 0 s |
| Recovery execution/recording wall time（不含 inference） | 26.953 s | 30.968 s |
| Worker measured wall time（含构建/关闭等） | 117.094 s | 123.172 s |

相对接触 gate 严格采用预注册的 v1 统计口径：恢复段所有 active robot contacts
的 per-physics-substep 正向法向力最大值，含自碰撞。
因此阈值为 **1.25 × 115.123817 + 25 = 168.904772 N**，B 78.745787 N 通过。
A 的参考最大值来自自碰撞；该定义不意味着 B 的冰箱接触小于 A，二者分项见表。
未事后改用 fridge-only、旧 50-step control 或其他候选。硬上限 1,500 N 保持不变。
这些是 MuJoCo 求解器接触力，不是真实力传感器读数或门铰链扭矩。
**接触 gate 通过不等于成功或修复批准；B 任务未成功，整体修复验收未通过。**

## 过程与五项结论

| 恢复步 | A 门角 | B 门角 | B clearance | B 阶段（该步后） |
|---:|---:|---:|---:|---|
| 50 | 34.389184° | 34.389184° | 0.118848 m | approach |
| 100 | 34.386473° | 34.386473° | 0.007400 m | tangential_push |
| 150 | 34.383763° | 34.383763° | 0.007815 m | tangential_push |
| 200 | 34.381054° | 42.171212° | 0.013321 m | approach |
| 250 | 31.742927° | 42.167889° | 0.011284 m | tangential_push |
| 300 | 15.916205° | 42.164566° | 0.013120 m | approach |

阶段摘要的 contact peaks 是对应单步值，不是整个前 50 步区间峰值；完整子步日志和
整段峰值保留了阶段采样点之间发生的接触。阶段 `push` 表示距离分支，不代表已经
建立有效载荷接触；B 在第 100 步进入该阶段，直到 7.624 秒才出现 ≥1 N 手门接触。

1. **反向 approach 是否修好：是。** 起点 clearance=0.250725 m，50 步降至
   0.118848 m、100 步降至 0.007400 m；每个 approach command 的世界方向检查通过。
   静态端点测试及真实 rollout 都支持该具体错误已修复。旧 v1 反向结果未被改写。
2. **B 是否关门成功：否。** 共 129 个动作使用 tangential_push 分支，产生接触后门
   开角反而增大，最终 42.164566°。本轮没有修改推门策略或追加尝试，不能把
   approach 修复等同于整个接触控制策略正确有效。
3. **A 是否在同等额外时间内成功：否。** 它在 15 秒内接触并部分关闭到 15.916205°，
   仍不满足真实全门成功判据；不能把部分进展称为完成。
4. **任务成功与验收：** 两路均无任务成功、均无硬停止；B 通过相对接触 gate，
   但缺少成功条件，整体修复未接受。
5. **是否支持小规模纠正示范采集：否。** 没有成功且通过验收的 Expert correction。
   只能说明这两个具体实现在本次预算内未恢复，不能证明该状态不可恢复。
   未取得 learned recovery、机器人成功率提升、泛化成功或硬件部署证据。

本轮到此结束，不启动采集、训练、其他场景、额外噪声候选或下一阶段。

## 实现、验证与证据保护

本轮新增源码仅 `scripts/recovery_v1b.py`（独立预算/重放/闭环/记录 runner）、
`scripts/recovery_v1b_report.py`（离线汇总与视频/输入检查）、
`tests/test_recovery_v1b.py`（方向拒绝、子步峰值和首次接触检查）。
没有编辑现有 Expert、环境包、controller gains 或历史 runner。
本报告为新增文档；原先工作区已有的修改和未跟踪文件保留，没有 Git commit/push/清理。

执行源码 SHA256：

| 源码 | SHA256 |
|---|---|
| recovery_v1b.py | `7bdcdd09b480e093f513763e935b2debfb27b9be7143ae814345f5a25b2e6aad` |
| recovery_branches.py（修正后 Expert） | `87d027b3e471340a09b0f1b51bacdf0b1adac078a03e0b740e35ab80a197ff08` |
| recovery_geometry_audit.py | `3e42cde77457151da9394529621e420503a1f869433756ae9c0d9f72bb3ba46d` |

完整 17 个执行/依赖源文件 hash 和逐文件快照保存在 plan 与 `executed-source/`。
执行前后 hashes 相同。沿用 v1 原 147 个受保护路径，并加入 v1 输出文件，
共 **260 个文件**前后 hashes 全部不变；这是枚举范围检查，不声称遍历所有历史输出。
原 Windows junction target 和新 junction target 均验证保持不变。

17 项相关检查通过：8 项 recovery、7 项 contact replanning、2 项 contact recorder。
初次 unittest 模块名调用未能发现 tests，改为 discover；两项 contact recorder
初次受沙箱临时目录权限阻止，沙箱外重跑通过。这些不是物理分支重试。
两个视频各 300 帧、20 fps、15 秒，全部解码核验通过；起止画面已检查。

新物理输出目录：`D:/EmbodiedOpsOutputs/recovery-feasibility-v1b`，新 junction：
`outputs/agent-workflow-v1/recovery-feasibility-v1b`。

- [预注册计划与 hashes](../outputs/agent-workflow-v1/recovery-feasibility-v1b/registered-plan.json)
- [Comparison CSV](../outputs/agent-workflow-v1/recovery-feasibility-v1b/comparison.csv)
- [完整 comparison JSON](../outputs/agent-workflow-v1/recovery-feasibility-v1b/comparison.json)
- [可播放 recordings 页面](../outputs/agent-workflow-v1/recovery-feasibility-v1b/recordings.html)
- [A 实际视频](../outputs/agent-workflow-v1/recovery-feasibility-v1b/A/rollout.mp4)
- [B 实际视频](../outputs/agent-workflow-v1/recovery-feasibility-v1b/B/rollout.mp4)
- [执行及保护检查](../outputs/agent-workflow-v1/recovery-feasibility-v1b/run-status.json)
- [视频及实时 policy 输入核验](../outputs/agent-workflow-v1/recovery-feasibility-v1b/delivery-verification.json)

大文件留在本地 ignored outputs 中，没有上传或公开发布。
