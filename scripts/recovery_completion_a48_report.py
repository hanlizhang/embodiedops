"""Offline evidence and video assembly only; no policy or simulation execution."""
import json
from pathlib import Path
import numpy as np
from instruction_control_audit import ROOT,read,dump,sha
from recovery_completion_a48 import OUT,A


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def main():
    result=read(OUT/'result.json');plan=read(OUT/'registered-plan.json')
    steps=lines(OUT/'steps.jsonl');preservation=read(OUT/'preservation.json')
    media={};contact_pairs={};windows=[]
    if result['restoration_validity'] and steps:
        import av
        from PIL import ImageDraw
        assert len(steps)==result['new_actions']
        assert all(s['global_index']==801+i for i,s in enumerate(steps))
        catalog={g['id']:g for g in read(OUT/'geometry-catalog.json')}
        for row in lines(OUT/'contacts.jsonl'):
            if row['step']<=800:continue
            for sample in row['samples']:
                for contact in sample['robot_contacts']:
                    force=max(0.,contact['force_N_contact_frame'][0])
                    if contact['constraint_address']<0 or force<1:continue
                    ids=tuple(sorted(contact['geom_ids']));key=':'.join(map(str,ids))
                    if key not in contact_pairs:
                        contact_pairs[key]=dict(geom_ids=list(ids),names=[catalog[i]['name'] for i in ids],
                            peak_N=0.,first_global_action=row['step'],last_global_action=row['step'],samples=0)
                    pair=contact_pairs[key];pair['peak_N']=max(pair['peak_N'],force)
                    pair['last_global_action']=row['step'];pair['samples']+=1
        for i in range(0,len(steps),50):
            batch=steps[i:i+50]
            windows.append(dict(global_start=batch[0]['global_index'],global_end=batch[-1]['global_index'],
                final_deg=np.rad2deg(list(batch[-1]['fridge']['qpos'].values())).tolist(),
                door_loaded_actions=sum(r['contact_peaks']['door']>=1 for r in batch),
                fridge_loaded_actions=sum(r['contact_peaks']['fridge']>=1 for r in batch),
                self_loaded_actions=sum(r['contact_peaks']['self']>=1 for r in batch)))
        # Retain original frame count/order/rate: concatenate, label, and re-encode only.
        full=OUT/'A_full_recovery.mp4';assert not full.exists()
        writer=av.open(str(full),'w');stream=writer.add_stream('libx264',rate=20)
        stream.width=768;stream.height=288;stream.pix_fmt='yuv420p';counts=[]
        for segment,path in enumerate([A/'rollout.mp4',OUT/'A_continuation.mp4']):
            count=0
            with av.open(str(path)) as reader:
                assert reader.streams.video[0].average_rate==20
                for frame in reader.decode(video=0):
                    img=frame.to_image();draw=ImageDraw.Draw(img)
                    draw.rectangle((0,0,768,31),fill='white')
                    label='V1B: original 300-action recovery budget (global 501-800)' if segment==0 else 'NEW BUDGET FROM 15.00s: frozen-policy continuation (global 801+)'
                    draw.text((5,8),label,fill='black')
                    for packet in stream.encode(av.VideoFrame.from_image(img)):writer.mux(packet)
                    count+=1
            counts.append(count)
        for packet in stream.encode():writer.mux(packet)
        writer.close()
        assert counts==[300,result['new_actions']]
        for name,expected in [('A_continuation.mp4',len(steps)),('A_full_recovery.mp4',300+len(steps))]:
            path=OUT/name
            with av.open(str(path)) as reader:
                count=sum(1 for _ in reader.decode(video=0))
            assert count==expected
            media[name]=dict(frames=count,fps=20,duration_seconds=count/20,sha256=sha(path))
        for call in result['inferences']:
            assert sha(OUT/'observations'/f"{call['after_new_action']:03d}.npz")==call['observation_sha256']
        result['full_recovery_actions']=300+result['new_actions']
        result['full_recovery_simulation_seconds']=(300+result['new_actions'])/20
        result['fixed_v1b_contact_reference_N']=168.904772
        result['new_segment_within_fixed_reference']=result['peak_normal_N']<=168.904772
        result['contact_reference_note']='Descriptive comparison only: new segment vs prior 300-action segment have different observation budgets; reference was not recomputed.'
    result['preservation']=preservation
    result['media']=media;result['contact_pairs_ge1N']=sorted(contact_pairs.values(),key=lambda p:-p['peak_N'])
    result['windows']=windows
    dump(OUT/'result.json',result)
    valid=result.get('execution_validity',False)
    conclusion=('有效执行且成功。冻结原始策略在该开发状态的一次预登记延续中完成任务，获得一条真实成功执行记录。'
        if valid and result['true_all_door_success'] else '有效执行但在固定上限内未成功。' if valid
        else '技术无效：'+str(result['stop_reason']))
    report=['# A48 Frozen-Policy Completion Check', '', '2026-10-23。'+conclusion,'',
        '本轮仅对开发状态 seed 48 做一次固定随机延续。v1b A 在其原额外 300 步预算内未完成的事实保持不变；本次超过原完整任务 600-action cap。没有学习、训练收益或泛化评测结论。','',
        '## 配置与恢复','',
        '重放原历史实际 public actions 1–500，再重放 v1b A 实际 actions 501–800；不重新推理前缀、不直接赋值 qpos/qvel。v1b 的 steps.index=300、historical_index=800 及 observations/300.npz 都是 action 800 **之后**，下一动作是 801。原始 RGB 无损数组存在，未使用视频帧作为像素参考。','',
        '同一 pinned checkpoint `lerobot/smolvla_robocasa`，revision `f7af588d11e30b0069f200792fb285904b74c80b`；model/processor/执行源码 hashes 在运行前验证。CloseFridge、10 flow steps、H=50、20 Hz、原 normalization、public action conversion 和 controller。','',
        '唯一允许的时间配置变化：从新环境 public actions 开始前设置 horizon=1000、ignore_done=False（且在唯一一次 reset 前设置）；无边界 reset。前 800 步仍检查所有真实 termination，原 action 800 的 budget done 现在应不出现。','',
        '策略上下文核验：n_obs_steps=1、RTC=null；predict_action_chunk 的 queues 只有 action 且该路径不填充它，运行时逐 chunk 断言队列为空。sample_actions 每次以 past_key_values=None 重新构建本 chunk KV cache，无 previous-action/leftover kwargs。因此重载冻结模型并 reset 不丢失此调用路径需要的跨 chunk 上下文。','',
        '新噪声预登记为 931007、931008、931009、931010，沿用 torch.manual_seed + 原生 sample_noise((1,50,32), device)。只执行达到停止条件前需要的 chunk；保存实际噪声与 hashes。每次使用当前实际三路 RGB/16D，不替换成旧 reference。','',
        f"重放步数：{result.get('replayed_actions',0)}；最大误差：`{result.get('max_replay_errors')}`。恢复有效性：**{result['restoration_validity']}**。",'',
        '边界 camera gate 固定 uint8 max≤1、MAE≤0.001；实际结果：`'+str(result.get('boundary_cameras','未通过/未到达；见异常'))+'`。','',
        '## 实测结果','',
        '| 项目 | 结果 |','|---|---|']
    for label,key in [('运行有效性','execution_validity'),('停止原因','stop_reason'),('新增动作数','new_actions'),('全局结束 action','global_end_action_index'),('开始门角 °','initial_deg'),('最终门角 °','final_deg'),('新增段最小门角 °','min_deg'),('真实 all-door success','true_all_door_success'),('首次成功 global action','first_success_global_index'),('首次成功新增 action','first_success_new_action'),('新增仿真秒','new_simulation_seconds'),('原 failure state 起累计恢复仿真秒','full_recovery_simulation_seconds'),('全部 robot contact peak N','peak_normal_N'),('冰箱接触 peak N','peak_fridge_N'),('手门接触 peak N','peak_door_N'),('自碰撞 peak N','peak_self_N'),('硬停止','hard_guard_status'),('重放 wall 秒','replay_wall_seconds'),('推理 wall 秒','inference_wall_seconds'),('执行及记录 wall 秒（排除推理）','execution_wall_seconds'),('worker wall 秒','worker_wall_seconds')]:
        report.append(f'| {label} | {result.get(key,"未执行/不可用")} |')
    report+=['','首次成功位置若为空，代表未成功，不是 0 步成功。未运行或无效不得计作机器人任务失败。worker wall 含构建/加载/关闭，不是整项工作的总耗时。','',
        '## 接触与末段行为','',
        '接触统计沿用每 physics substep 的 active robot contact 正法向力峰值；硬上限 1500 N。固定 v1b 相对参考是 **168.904772 N**，不由本次峰值抬高；两段观察预算不同，该比较仅作描述，不作为学习收益或安全认证。数据为 MuJoCo 求解器力，非硬件传感器或门铰链扭矩。','',
        '| Global action 区间 | 末角度 ° | ≥1 N 手门接触 actions | 冰箱接触 actions | 自碰撞 actions |','|---|---|---:|---:|---:|']
    for w in windows:report.append(f"| {w['global_start']}–{w['global_end']} | {w['final_deg']} | {w['door_loaded_actions']} | {w['fridge_loaded_actions']} | {w['self_loaded_actions']} |")
    report+=['','实际 ≥1 N 接触 geom identities（按峰值排序）：','']
    for p in result['contact_pairs_ge1N'][:12]:report.append(f"- `{p['names'][0]}` ↔ `{p['names'][1]}`：peak {p['peak_N']:.6f} N；首次/末次 global actions {p['first_global_action']}/{p['last_global_action']}。")
    if steps:
        change=result['initial_deg'][0]-result['final_deg'][0]
        report+=['',f'新增段净关门进展 {change:.6f}°；具体阶段角度与接触见上表。精确部位按 geom 名称报告，不凭视频推测。']
    report+=['','## 交付与保护','',
        f"保护检查：{preservation['checked']} 个枚举文件，changed={preservation['changed']}；源码不变={preservation['sources_unchanged']}；旧 junction 不变={preservation['junctions_unchanged']}。没有复制整套历史记录。新增唯一物理输出目录 `{OUT}`。",'',
        '新增脚本 `scripts/recovery_completion_a48.py` 复用 v1b 核验/接触工具；`scripts/recovery_completion_a48_report.py` 仅离线汇总和拼接。已有代码和结果未改。运行前完成实际 800-action 顺序/数组一致性检查、源码身份检查和语法检查。','',
        f"- [Manifest]({OUT.as_posix()}/manifest.json)",f"- [Registered plan]({OUT.as_posix()}/registered-plan.json)",f"- [Result]({OUT.as_posix()}/result.json)"]
    if media:
        report+=[f"- [新增实际视频]({OUT.as_posix()}/A_continuation.mp4)",f"- [完整恢复视频]({OUT.as_posix()}/A_full_recovery.mp4)", '',
            f"视频完整解码：新增 {media['A_continuation.mp4']['frames']} 帧，拼接 {media['A_full_recovery.mp4']['frames']} 帧，均 20 fps。拼接只重编码和添加标注，不变速、不插帧；从 15.00 秒起画面明确标注 NEW BUDGET。"]
    else:report+=['','未生成新增执行或拼接视频：恢复无效或没有新增动作，不能冒充成功记录。']
    report+=['','本轮结束，不追加时间/噪声，不运行 B，不采集训练，不提交或推送 Git。','']
    (ROOT/'environment/recovery-completion-a48-v1.md').write_text('\n'.join(report),encoding='utf-8')
    dump(OUT/'delivery-audit.json',dict(media=media,report_script_sha256=sha(Path(__file__)),report_sha256=sha(ROOT/'environment/recovery-completion-a48-v1.md')))
    print(json.dumps(dict(conclusion=conclusion,result=result),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
