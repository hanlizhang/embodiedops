# Recontact Self-Imitation v1

**恢复数据覆盖不足，未进入训练。**

新增收集 24/24 次；有效无硬停止成功 11；技术 invalid 0；硬停止 0。
合格来源 13 条，覆盖 3 个 reset 场景 [44, 47, 49]；合格再接触窗口 31 个，含完成段训练样本 39 个。
训练门槛固定为 8 条来源 / 4 个场景。本轮实际 optimizer updates：0。

## 冻结口径

原模型、原 processor、CloseFridge、H50、10-step Flow Matching、20Hz。每个新完整任务最多600 actions，单worker最多600秒；噪声规则和执行顺序见 registered-plan.json。
同一目标门先有有效手部/夹爪接触，再连续至少5步缺失接触，才构成因果锚点；未来最多50个实际动作内同门再接触，净关闭≥2°或窗口内真实成功，并且来源最终真实成功且无硬停止。所有门槛均未放宽。
额外观察捕获不触发推理。样本为原始三路RGB+16D state+指令到未来最多50个已执行的postprocessor12D动作；不要求未来每步RGB。时间padding与内部32D中的20个padding维度分别排除。
接触分项中的 other_fridge 指其余机器人—冰箱接触，包括机械臂连杆接触目标门；它不等于另一扇门，也不计作有效手部/夹爪再接触。精确部位以 geometry-catalog 和 contacts 中的 geom identity 为准。
历史47的pre101窗口101–150后续在163成功；pre151的13步是完成段，不是新的接触缺失锚点。历史49的pre301有29个实际动作并在窗口内成功。
普通对照已按配方与来源匹配，登记为历史task-ID step300推理权重；历史optimizer状态不完整，不作续训。新候选若训练则从原始权重开始。
测试60–65与本轮收集场景无冲突；原461条演示metadata没有环境reset seed，不能把dataset episode index当成seed，原数据/预训练场景重叠未知。120–123未启用。

## 所有新收集尝试

| Seed | 噪声序列 | 有效 | 真实成功 | Actions | 合格来源 | 再接触窗口 | 硬停止 | 峰值全部机器人 N |
|---|---:|---|---|---:|---|---:|---|---:|
| 42 | 1 | True | False | 600 | False | 0 | False | 150.111 |
| 44 | 1 | True | True | 299 | True | 3 | False | 277.766 |
| 45 | 1 | True | False | 600 | False | 0 | False | 84.858 |
| 46 | 1 | True | False | 600 | False | 0 | False | 134.715 |
| 47 | 1 | True | True | 166 | True | 1 | False | 100.163 |
| 49 | 1 | True | True | 366 | True | 6 | False | 182.281 |
| 42 | 2 | True | False | 600 | False | 0 | False | 203.326 |
| 44 | 2 | True | True | 218 | True | 1 | False | 253.147 |
| 45 | 2 | True | False | 600 | False | 0 | False | 227.519 |
| 46 | 2 | True | False | 600 | False | 0 | False | 78.152 |
| 47 | 2 | True | True | 292 | True | 3 | False | 87.459 |
| 49 | 2 | True | True | 287 | True | 4 | False | 48.463 |
| 42 | 3 | True | False | 600 | False | 0 | False | 87.760 |
| 44 | 3 | True | True | 325 | True | 2 | False | 86.715 |
| 45 | 3 | True | False | 600 | False | 0 | False | 169.096 |
| 46 | 3 | True | False | 600 | False | 0 | False | 122.322 |
| 47 | 3 | True | True | 282 | True | 2 | False | 234.150 |
| 49 | 3 | True | False | 600 | False | 0 | False | 240.894 |
| 42 | 4 | True | False | 600 | False | 0 | False | 280.715 |
| 44 | 4 | True | True | 326 | True | 2 | False | 104.530 |
| 45 | 4 | True | False | 600 | False | 0 | False | 72.690 |
| 46 | 4 | True | False | 600 | False | 0 | False | 137.713 |
| 47 | 4 | True | True | 277 | True | 2 | False | 83.762 |
| 49 | 4 | True | True | 252 | True | 3 | False | 134.929 |

## 合格再接触事件

| 来源 | 场景 | 锚点/动作窗口 | 连续缺失动作数 | 首次再接触 | 门角起→终 ° | 窗口内成功 |
|---|---:|---|---:|---:|---|---|
| historical-A-ID-47 | 47 | pre101: 101–150 | 5 | 107 | 78.363936→12.929204 | False |
| historical-A-ID-49 | 49 | pre301: 301–329 | 23 | 304 | 5.216629→0.034077 | True |
| collection-44-r1-frozen | 44 | pre248: 248–297 | 5 | 251 | 36.411177→1.250188 | False |
| collection-44-r1-frozen | 44 | pre251: 251–299 | 8 | 251 | 34.574509→-0.028828 | True |
| collection-44-r1-frozen | 44 | pre268: 268–299 | 5 | 274 | 18.647318→-0.028828 | True |
| collection-47-r1-frozen | 47 | pre135: 135–166 | 5 | 137 | 44.626532→0.045651 | True |
| collection-49-r1-frozen | 49 | pre162: 162–211 | 5 | 162 | 78.580937→39.978309 | False |
| collection-49-r1-frozen | 49 | pre206: 206–255 | 5 | 220 | 43.580030→22.565045 | False |
| collection-49-r1-frozen | 49 | pre227: 227–276 | 5 | 232 | 33.186796→13.588552 | False |
| collection-49-r1-frozen | 49 | pre286: 286–335 | 5 | 287 | 8.529710→0.563536 | False |
| collection-49-r1-frozen | 49 | pre293: 293–342 | 5 | 293 | 7.271844→0.563531 | False |
| collection-49-r1-frozen | 49 | pre351: 351–366 | 47 | 365 | 0.563526→0.065269 | True |
| collection-44-r2-frozen | 44 | pre146: 146–195 | 5 | 147 | 82.192470→22.651037 | False |
| collection-47-r2-frozen | 47 | pre196: 196–245 | 5 | 208 | 56.296420→35.300155 | False |
| collection-47-r2-frozen | 47 | pre201: 201–250 | 10 | 208 | 53.392303→32.714099 | False |
| collection-47-r2-frozen | 47 | pre284: 284–292 | 5 | 286 | 5.948186→-0.023489 | True |
| collection-49-r2-frozen | 49 | pre156: 156–205 | 5 | 163 | 67.486736→28.677194 | False |
| collection-49-r2-frozen | 49 | pre178: 178–227 | 5 | 178 | 55.514100→25.612253 | False |
| collection-49-r2-frozen | 49 | pre207: 207–256 | 5 | 225 | 28.170085→12.729797 | False |
| collection-49-r2-frozen | 49 | pre261: 261–287 | 5 | 261 | 11.341295→0.109402 | True |
| collection-44-r3-frozen | 44 | pre257: 257–306 | 5 | 261 | 37.680469→8.987352 | False |
| collection-44-r3-frozen | 44 | pre267: 267–316 | 5 | 267 | 35.714947→5.438674 | False |
| collection-47-r3-frozen | 47 | pre206: 206–255 | 5 | 209 | 28.827587→8.537736 | False |
| collection-47-r3-frozen | 47 | pre255: 255–282 | 5 | 263 | 8.978760→-0.075536 | True |
| collection-44-r4-frozen | 44 | pre273: 273–322 | 5 | 273 | 47.588801→4.703370 | False |
| collection-44-r4-frozen | 44 | pre309: 309–326 | 5 | 314 | 15.818848→0.295690 | True |
| collection-47-r4-frozen | 47 | pre179: 179–228 | 5 | 180 | 54.589399→27.443381 | False |
| collection-47-r4-frozen | 47 | pre206: 206–255 | 5 | 231 | 29.976957→15.642644 | False |
| collection-49-r4-frozen | 49 | pre174: 174–223 | 5 | 175 | 53.302683→19.168888 | False |
| collection-49-r4-frozen | 49 | pre182: 182–231 | 5 | 183 | 45.279917→14.258340 | False |
| collection-49-r4-frozen | 49 | pre191: 191–240 | 5 | 191 | 37.708201→9.226477 | False |

## 物理评测与回归

未进入新训练；恢复benchmark与完整任务回归均未执行。三个模型本轮成功数均为 N/A，不能写成0次成功/0次失败的评测结果。

## 解释与交付

这些窗口是成功episode中的再接触行为，不能自动视为长期故障恢复或由新视觉重规划触发。完成段和重叠窗口不增加来源数；同reset不同噪声不增加场景数。
本轮全部worker累计 2373.441s，推理累计 128.240s；总墙钟 3625.117s。
物理输出：`D:\EmbodiedOpsOutputs\recontact-self-imitation-v1`。冻结方案、数据索引、实际执行清单、所有attempt、result.json与video-index.html均在此目录。
未训练或未达到验收不产生成功剪辑。未开启几何教师、seed48加时、其他checkpoint搜索或额外训练。没有commit/push。
