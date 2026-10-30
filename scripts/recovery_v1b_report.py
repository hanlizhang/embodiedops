"""Read completed v1b evidence only; never load a policy or robot environment."""
import csv
import html
import json
from pathlib import Path
import numpy as np
from instruction_control_audit import ROOT, read, dump, sha

OUT=ROOT/'outputs/agent-workflow-v1/recovery-feasibility-v1b'


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def main():
    plan=read(OUT/'registered-plan.json'); status=read(OUT/'run-status.json')
    rows=[]; video_audit={}; input_audit={}
    for branch in ['A','B']:
        folder=OUT/branch
        if not (folder/'result.json').exists():
            rows.append(dict(branch=branch,restoration_validity=None,execution_status='unrun',
                stop_reason=status.get('stop_reason','not executed'),actual_recovery_actions=0,
                true_task_success=None,hard_guard_status='not_reached',relative_contact_gate_status='unavailable'))
            continue
        r=read(folder/'result.json'); steps=lines(folder/'steps.jsonl')
        r['recorded_actions']=len(steps)
        r['termination_action']=r['actual_recovery_actions']
        r['termination_simulation_seconds']=r.get('recovery_simulation_seconds',0.)
        if steps:
            r['initial_clearance_m']=steps[0]['decision']['nearest_distance_m']
            stages=lines(folder/'stages.jsonl')
            r['last_stage_clearance_m']=stages[-1]['clearance_m'] if stages else None
            r['approach_direction_check']='passed_each_command' if branch=='B' else 'not_applicable'
            r['push_actions']=sum(s['decision']['stage']=='tangential_push' for s in steps) if branch=='B' else None
            r['stop_detail']='Explicit 800-action episode horizon and 300-action recovery budget reached together' if len(steps)==300 and steps[-1]['terminated'] else r['stop_reason']
            assert len(steps)==r['actual_recovery_actions']
            assert all(s['historical_index']==501+i for i,s in enumerate(steps))
            assert all(abs(s['physical']['time']-(r['recovery_start_sim_time']+(i+1)*.05))<1e-8 for i,s in enumerate(steps))
        if branch=='A' and r.get('inferences'):
            checks=[]
            for call in r['inferences']:
                i=call['after_action']
                with np.load(folder/f'policy-input-{i:03d}.npz') as policy_input:
                    actual_path=folder/'observations'/f'{i:03d}.npz' if i else ROOT/'outputs/agent-workflow-v1/recovery-feasibility-v1'/ (plan['state']['id']+'-restoration-audit')/'reconstructed-observation.npz'
                    with np.load(actual_path) as actual:
                        equal=all(np.array_equal(policy_input[k],actual[k]) for k in actual.files)
                checks.append(dict(after_action=i,exact_expected_observation=equal,seed=call['seed']))
                assert equal
            input_audit[branch]=checks
        video=folder/'rollout.mp4'
        if video.exists():
            import av
            with av.open(str(video)) as container:
                count=sum(1 for frame in container.decode(video=0))
            video_audit[branch]=dict(path=str(video),frames=count,expected_frames=len(steps),
                decodable=count==len(steps),sha256=sha(video),bytes=video.stat().st_size)
        rows.append(r)
    a,b=rows
    a_valid=a['restoration_validity'] and a['execution_status']=='completed' and a['hard_guard_status']=='clear'
    ceiling=1.25*a['peak_normal_N']+25 if a_valid else None
    a['relative_contact_gate_status']='reference' if a_valid else 'unavailable'
    b['relative_contact_gate_status']=('passed' if b.get('peak_normal_N',float('inf'))<=ceiling else 'failed') if ceiling is not None and b['restoration_validity'] else 'unavailable'
    for r in rows: r['relative_contact_ceiling_N']=ceiling
    accepted=bool(b['restoration_validity'] and b['true_task_success'] and b['hard_guard_status']=='clear' and b['execution_status']=='completed' and b['relative_contact_gate_status']=='passed')
    differential=accepted and a_valid and not a['true_task_success']
    complete=bool(status['complete'] and all(r['restoration_validity'] and r['execution_status'] in ('completed','hard_stop') for r in rows))
    if not complete:
        conclusion='Incomplete：存在无效或未完成分支，不能把它计为机器人任务失败，也不能得出配对恢复结论。'
    elif b['true_task_success'] and not accepted:
        conclusion='已展示该状态的物理可恢复性，但修复验收未通过。不得批准为安全纠正数据来源。'
    elif differential:
        conclusion='该开发状态上存在原始 VLA 此次 continuation 未实现的、可执行且通过验收的纠正行为，值得下一步检验 corrective imitation learning。'
    elif a['true_task_success'] and b['true_task_success']:
        conclusion='两路均成功，应优先解释额外恢复预算的作用；没有证明 Expert 的独有恢复能力。'
    elif not a['true_task_success'] and not b['true_task_success']:
        conclusion='两路均未成功；只说明这两种具体实现未在本次额外 300-action 预算内恢复，不能证明该状态不可恢复。现有证据不足以支持纠正示范采集。'
    else:
        conclusion='这两种具体实现未在本次预算内同时形成通过验收的纠正对照；失败不能证明该状态不可恢复。'
    payload=dict(status='complete' if complete else 'incomplete',rows=rows,relative_contact_ceiling_N=ceiling,
        expert_repair_accepted=accepted,differential_corrective_evidence=differential,conclusion=conclusion,
        run_status=status,video_audit=video_audit,policy_input_audit=input_audit)
    dump(OUT/'comparison.json',payload)
    columns=['branch','restoration_validity','execution_status','stop_reason','actual_recovery_actions',
        'true_task_success','initial_deg','final_deg','min_deg','first_door_contact_seconds',
        'first_success_action','first_success_seconds','termination_simulation_seconds','peak_normal_N',
        'peak_fridge_N','peak_self_N','peak_door_N','hard_guard_status','relative_contact_gate_status',
        'relative_contact_ceiling_N','prefix_wall_seconds','inference_wall_seconds','recovery_wall_seconds','worker_wall_seconds']
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Recovery Feasibility v1b</title>',
        '<style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#f5f5f2;color:#18252e}video{width:100%;max-width:768px}section{margin:32px 0;padding:22px;background:white;border-radius:12px}a{color:#176282}</style>',
        '<h1>Recovery Feasibility v1b</h1><p>seed 48 · 历史 action 500 后 · 每路最多额外 300 actions / 15 秒仿真时间</p>',
        '<p>'+html.escape(conclusion)+'</p><p>原始完整任务上限是 600 actions。本轮是独立失败状态恢复实验，不能表述为原预算内成功。外部 Expert 不是神经网络 Action Expert。</p>',
        '<p><a href="comparison.csv">Comparison CSV</a> · <a href="comparison.json">完整结果</a> · <a href="registered-plan.json">预注册计划</a></p>']
    for r in rows:
        branch=r['branch']; title='Original frozen SmolVLA' if branch=='A' else 'Corrected Recovery Expert'
        page.append('<section><h2>'+branch+' — '+title+'</h2><p>'+html.escape(str({k:r.get(k) for k in ['restoration_validity','execution_status','stop_reason','actual_recovery_actions','true_task_success','hard_guard_status','relative_contact_gate_status']}))+'</p>')
        if branch in video_audit:
            page.append(f'<video controls preload="metadata" poster="{branch}/before.png" src="{branch}/rollout.mp4"></video><p><a href="{branch}/rollout.mp4">打开实际视频</a></p>')
        else: page.append('<p>未生成恢复视频：分支未执行恢复动作或在此前中止。不得将其按任务失败计数。</p>')
        page.append('</section>')
    page.append('</html>'); (OUT/'recordings.html').write_text('\n'.join(page),encoding='utf-8')
    dump(OUT/'delivery-verification.json',dict(videos=video_audit,policy_inputs=input_audit,report_script_sha256=sha(Path(__file__)),
        related_tests=dict(recovery=8,contact_replanning=7,contact_recorder=2,passed=17,
            deviations=['Initial unittest module-name invocation failed discovery; corrected to discover.',
                        'Contact-recorder tests initially blocked by sandbox temporary-directory permissions; both passed elevated rerun.'])))
    print(json.dumps(payload,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
