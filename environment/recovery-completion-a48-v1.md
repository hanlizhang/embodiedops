# A48 Frozen-Policy Completion Check

2026-10-23。有效执行但在固定上限内未成功。

本轮仅对开发状态 seed 48 做一次固定随机延续。v1b A 在其原额外 300 步预算内未完成的事实保持不变；本次超过原完整任务 600-action cap。没有学习、训练收益或泛化评测结论。

## 配置与恢复

重放原历史实际 public actions 1–500，再重放 v1b A 实际 actions 501–800；不重新推理前缀、不直接赋值 qpos/qvel。v1b 的 steps.index=300、historical_index=800 及 observations/300.npz 都是 action 800 **之后**，下一动作是 801。原始 RGB 无损数组存在，未使用视频帧作为像素参考。

同一 pinned checkpoint `lerobot/smolvla_robocasa`，revision `f7af588d11e30b0069f200792fb285904b74c80b`；model/processor/执行源码 hashes 在运行前验证。CloseFridge、10 flow steps、H=50、20 Hz、原 normalization、public action conversion 和 controller。

唯一允许的时间配置变化：从新环境 public actions 开始前设置 horizon=1000、ignore_done=False（且在唯一一次 reset 前设置）；无边界 reset。前 800 步仍检查所有真实 termination，原 action 800 的 budget done 现在应不出现。

策略上下文核验：n_obs_steps=1、RTC=null；predict_action_chunk 的 queues 只有 action 且该路径不填充它，运行时逐 chunk 断言队列为空。sample_actions 每次以 past_key_values=None 重新构建本 chunk KV cache，无 previous-action/leftover kwargs。因此重载冻结模型并 reset 不丢失此调用路径需要的跨 chunk 上下文。

新噪声预登记为 931007、931008、931009、931010，沿用 torch.manual_seed + 原生 sample_noise((1,50,32), device)。只执行达到停止条件前需要的 chunk；保存实际噪声与 hashes。每次使用当前实际三路 RGB/16D，不替换成旧 reference。

重放步数：800；最大误差：`{'qpos': 0.0, 'qvel': 0.0, 'state': 0.0}`。恢复有效性：**True**。

边界 camera gate 固定 uint8 max≤1、MAE≤0.001；实际结果：`{'robot0_agentview_left': {'max': 0.0, 'mean': 0.0}, 'robot0_agentview_right': {'max': 0.0, 'mean': 0.0}, 'robot0_eye_in_hand': {'max': 0.0, 'mean': 0.0}}`。

## 实测结果

| 项目 | 结果 |
|---|---|
| 运行有效性 | True |
| 停止原因 | global_horizon_1000 |
| 新增动作数 | 200 |
| 全局结束 action | 1000 |
| 开始门角 ° | [15.916204952046696] |
| 最终门角 ° | [7.8621295773003155] |
| 新增段最小门角 ° | [7.8621295773003155] |
| 真实 all-door success | False |
| 首次成功 global action | None |
| 首次成功新增 action | None |
| 新增仿真秒 | 10.000000000012221 |
| 原 failure state 起累计恢复仿真秒 | 25.0 |
| 全部 robot contact peak N | 207.4283080946359 |
| 冰箱接触 peak N | 207.4283080946359 |
| 手门接触 peak N | 22.770194884770994 |
| 自碰撞 peak N | 0.0 |
| 硬停止 | clear |
| 重放 wall 秒 | 70.14000000001397 |
| 推理 wall 秒 | 2.74900000001071 |
| 执行及记录 wall 秒（排除推理） | 16.18799999996554 |
| worker wall 秒 | 144.09399999998277 |

首次成功位置若为空，代表未成功，不是 0 步成功。未运行或无效不得计作机器人任务失败。worker wall 含构建/加载/关闭，不是整项工作的总耗时。

## 接触与末段行为

接触统计沿用每 physics substep 的 active robot contact 正法向力峰值；硬上限 1500 N。固定 v1b 相对参考是 **168.904772 N**，不由本次峰值抬高；两段观察预算不同，该比较仅作描述，不作为学习收益或安全认证。数据为 MuJoCo 求解器力，非硬件传感器或门铰链扭矩。

| Global action 区间 | 末角度 ° | ≥1 N 手门接触 actions | 冰箱接触 actions | 自碰撞 actions |
|---|---|---:|---:|---:|
| 801–850 | [7.863988711179913] | 8 | 24 | 0 |
| 851–900 | [7.863368951044761] | 0 | 3 | 0 |
| 901–950 | [7.862749239752845] | 0 | 15 | 0 |
| 951–1000 | [7.8621295773003155] | 0 | 5 | 0 |

实际 ≥1 N 接触 geom identities（按峰值排序）：

- `gripper0_right_hand_collision` ↔ `fridge_main_group_freezer_door_handle_2`：peak 207.428308 N；首次/末次 global actions 927/965。
- `gripper0_right_finger1_collision` ↔ `fridge_main_group_g34`：peak 64.584304 N；首次/末次 global actions 835/934。
- `gripper0_right_finger1_collision` ↔ `fridge_main_group_freezer_door_main`：peak 64.544592 N；首次/末次 global actions 835/934。
- `gripper0_right_hand_collision` ↔ `fridge_main_group_fridge_door_main`：peak 22.770195 N；首次/末次 global actions 803/819。
- `gripper0_right_hand_collision` ↔ `fridge_main_group_g67`：peak 17.200827 N；首次/末次 global actions 803/819。
- `gripper0_right_finger2_collision` ↔ `fridge_main_group_g34`：peak 10.243200 N；首次/末次 global actions 922/924。
- `gripper0_right_finger2_collision` ↔ `fridge_main_group_freezer_door_main`：peak 10.208028 N；首次/末次 global actions 922/924。

新增段净关门进展 8.054075°；具体阶段角度与接触见上表。精确部位按 geom 名称报告，不凭视频推测。

末段表现是**先继续关闭，再失去目标门的有效载荷接触并停滞**，不是持续有效推门。
目标 `fridge_door_main` 的 ≥1 N 手部接触最后出现在 global action 819；其后仍有冰箱
其他部位接触，包括 `freezer_door_main` 和 `freezer_door_handle_2`，不能把这些力都解释为
目标门上的有效关门力。最后 150 actions 的角度仅从 7.863989° 变为 7.862130°，
净变化约 0.001859°，没有显著反向打开，也没有完成真实任务。
新增段最大 207.428308 N 来自手部与 `freezer_door_handle_2` 的接触，
**超过固定 168.904772 N 参考线**；不通过提高参考线把它改判为通过。
上述是接触日志和角度的联合观察，不声称已确定策略失效的完整因果机制。

## 交付与保护

保护检查：944 个枚举文件，changed=[]；源码不变=True；旧 junction 不变=True。没有复制整套历史记录。新增唯一物理输出目录 `D:\EmbodiedOpsOutputs\recovery-completion-a48-v1`。

新增脚本 `scripts/recovery_completion_a48.py` 复用 v1b 核验/接触工具；`scripts/recovery_completion_a48_report.py` 仅离线汇总和拼接。已有代码和结果未改。运行前完成实际 800-action 顺序/数组一致性检查、源码身份检查和语法检查。

- [Manifest](D:/EmbodiedOpsOutputs/recovery-completion-a48-v1/manifest.json)
- [Registered plan](D:/EmbodiedOpsOutputs/recovery-completion-a48-v1/registered-plan.json)
- [Result](D:/EmbodiedOpsOutputs/recovery-completion-a48-v1/result.json)
- [新增实际视频](D:/EmbodiedOpsOutputs/recovery-completion-a48-v1/A_continuation.mp4)
- [完整恢复视频](D:/EmbodiedOpsOutputs/recovery-completion-a48-v1/A_full_recovery.mp4)

视频完整解码：新增 200 帧，拼接 500 帧，均 20 fps。拼接只重编码和添加标注，不变速、不插帧；从 15.00 秒起画面明确标注 NEW BUDGET。

本轮结束，不追加时间/噪声，不运行 B，不采集训练，不提交或推送 Git。
