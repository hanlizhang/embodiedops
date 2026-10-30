# Nemotron Agentic Verification & Final Demo

执行日期：2026-10-31（Europe/Berlin）。真实闭环已完成；独立验收拒绝已有候选，未训练或生成新的策略修复。

NVIDIA `nvidia/Nemotron-3_5-Lightning` 经 Nebius 原生 `tools/tool_calls/tool_call_id` 实际选择并调用：

`inspect_case → compare_candidates(archived) → inspect_trajectory(frozen) → inspect_trajectory(mix) → check_acceptance(archived) → run_bounded_verification → check_acceptance(new)`。

这不是 deterministic backend 的预写执行顺序。场景、工具边界、预算、模型与验收规则由人预先固定；具体工具顺序和是否追加验证由本次真实 API 响应决定。每个工具结果均实际回传给下一次模型请求，最后得到模型总结。

## 实际结果

模型先读取归档并得到拒绝，然后自主选择一次固定配对验证，以确认相同协议的新执行表现。两个独立 worker 均完成，场景/控制器/相机边界、实际噪声和独立接触检查通过；没有硬停止。每模型仅运行一次、最多600个 public actions，H=50、10-step Flow Matching、20Hz；复用原 checkpoint、processor、normalization 与物理 worker，没有重试或替换条件。

接触分项依次为目标手门／其他冰箱／自接触／全部机器人，单位 N；success 为真实 all-door predicate／通过相对验收的成功。

| 证据 | 模型 | actions | raw / qualified success | 初始 → 最终门角 ° | 接触峰值 N | 相对 gate / 硬停止 | worker 秒 |
|---|---|---:|---|---|---|---|---:|
| 归档 v2 | frozen | 439 | True / True | 86.036153 → -0.033344 | 189.833 / 0.000 / 0.000 / 189.833 | True / False | 76.828 |
| 归档 v2 | recontact_mix_300 | 600 | False / False | 86.036153 → 89.760425 | 449.547 / 188.610 / 0.000 / 449.547 | False / False | 85.781 |
| 本轮新验证 | frozen | 600 | False / False | 86.036153 → 2.227969 | 198.160 / 155.424 / 0.000 / 198.160 | True / False | 131.719 |
| 本轮新验证 | recontact_mix_300 | 600 | False / False | 86.036153 → 90.016391 | 298.552 / 0.000 / 0.000 / 298.552 | False / False | 91.015 |

本轮相对接触上限为 `1.25 × 198.160009 + 25 = 272.700011 N`，混合模型峰值 `298.551572 N` 超限。归档对应上限为 `262.291041 N`，归档混合模型同样超限。二者均未达到1500N硬停止上限，不能把这两个门槛混为一谈。

本轮独立 `agent_workflow.acceptance()` 结论：**rejected**；原因：`contact_force_regression, door_progress_regression, no_success_gain`。Nemotron无权限修改规则或覆盖结论。新验证后，原先缓存的归档验收被清空，必须重新用 `source=new` 验收。

**重复执行限制：**归档 frozen 成功，本轮 frozen 未成功；因此未复现完整成功模式，也不是位级相同轨迹。已有相机容差、噪声与状态匹配检查通过不等于整个闭环必然一致。本轮没有确定差异原因，不追加 rollout 或扩大诊断。旧 v2 结果保持不变。

## 模型文字错误与能力边界

原始最终总结被完整保留。它错误声称两组相对接触 gate 都通过，并用“确认归档模式”模糊了新 frozen 的失败；它也未遵循中文最终回复要求，返回了英文。`narrative-review.json` 与 Demo 明确纠正，结构化验收是最终权威。此次完成的是可审计的真实工具调用与验证闭环，不是证明模型叙述完全可靠。

已经验证：原生多轮工具选择、物理证据查询、模型自主请求固定 worker 配对、结果回传、独立拒绝与预算边界。未实现或未证明：自动生成有效策略修复、学习收益、泛化、硬件部署、任意场景的自主科研、可靠因果根因诊断。案例与工具由人限定，诊断得到的接触/门角现象不是已证明的唯一失效机制。

## 用量与检查

- 新 API 请求 `8/10`，工具决策 `7/8`，真实新 episodes `2/2`，训练0，自动重试0。
- 实际 API usage：input `78982`，output `1329`，total `80311`；提供商逐次报告 reasoning tokens 为0。每次 `max_tokens=2048` 包含 reasoning，不重复计费。
- 按登记公开费率计算 `$0.00505788`；不是账单金额，未假设缓存折扣，也不是提供商硬消费上限。
- API请求延迟合计 `10.273s`；真实闭环 `240.165s`（含工具与worker，非GPU纯计算时间）。
- 12项新 Agent 离线测试通过；覆盖真实选择顺序保留、tool_call_id回传、独立gate、参数白名单、8步/10请求上限、不重试、密钥脱敏。未重复基础 smoke test 或全量测试。

## 入口和交付

- Runner：`scripts/nemotron_agent_demo.py`；工具：`scripts/nemotron_case_tools.py`；独立物理适配：`scripts/nemotron_verification_worker.py`；测试：`tests/test_nemotron_agent_demo.py`。
- 已执行命令：`D:\Apps\miniconda\envs\embodiedops-robocasa\python.exe -B -u scripts/nemotron_agent_demo.py`。固定输出已存在时拒绝重跑；该命令不是额外计算授权。
- 完整本地记录：`C:\Users\Alan\Projects\embodiedops\outputs\nemotron-agent-demo-v1`。
- 本地 Demo：`C:\Users\Alan\Projects\embodiedops\outputs\nemotron-agent-demo-v1\index.html`，以本地文件方式打开，连接四段实际 seed64 视频。
- 公开 Demo：`https://hanlizhang.github.io/embodiedops/demo/#agentic`；源文件 `docs/demo/index.html`。公开版展示本轮数字与实际工具 trace；视频只使用原先已公开的历史 seed44/43链接，并明确它们不是 seed64 配对视频。
- 已提交的小型证据入口：`docs/evidence/nemotron-agent-demo-v1/README.md`。原始图像、逐子步日志、视频、checkpoint和`.env`保持本地。

| 视频 | 本地绝对路径 |
|---|---|
| 归档 frozen | `D:\EmbodiedOpsOutputs\recontact-self-imitation-v2\full\full-64-r1-frozen\rollout.mp4` |
| 归档 recontact_mix_300 | `D:\EmbodiedOpsOutputs\recontact-self-imitation-v2\full\full-64-r1-recontact_mix_300-attempt02\rollout.mp4` |
| 本轮 frozen | `C:\Users\Alan\Projects\embodiedops\outputs\nemotron-agent-demo-v1\physical\full-64-r1-frozen\rollout.mp4` |
| 本轮 recontact_mix_300 | `C:\Users\Alan\Projects\embodiedops\outputs\nemotron-agent-demo-v1\physical\full-64-r1-recontact_mix_300\rollout.mp4` |

交付检查：四段视频均实际解码首帧通过，20fps，帧数分别439/600/600/600，与实际动作数一致。公共HTML不含本地媒体链接。本地file://页面被内置浏览器安全策略阻止，因此不声称已在该浏览器播放；本地文件路径与媒体内容核验通过。
