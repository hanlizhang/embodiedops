"""Summarize the bounded study from saved evidence; never infer or simulate."""
import csv
import html
import json
import math
from pathlib import Path
import time
from instruction_control_audit import ROOT,read,dump,sha
from recontact_self_imitation import OUT,rows,contacts,noise_seed


def execution_audit(results):
    import av
    import hashlib
    import numpy as np
    import torch
    checked=[]
    for r in results:
        if not r['execution_validity']:continue
        p=Path(r['job']['folder']);steps=rows(p/'steps.jsonl');job=r['job']
        assert len(steps)==r['actions']<= (150 if job['kind']=='recovery' else 600)
        assert r['worker_seconds']<=600
        assert not any(x['terminated'] or x['truncated'] or x['fridge']['success'] for x in steps[:-1])
        assert bool(steps[-1]['fridge']['success'])==r['success']
        for inf in r['inferences']:
            local=inf['local_pre_action'];pre=inf['pre_action']
            assert inf['seed']==noise_seed(job['kind'],job['seed'],job['replicate'],local)
            noise=torch.load(p/f'noise-{local:04d}.pt',map_location='cpu',weights_only=True)
            assert tuple(noise.shape)==(1,50,32) and torch.isfinite(noise).all()
            assert hashlib.sha256(noise.numpy().tobytes()).hexdigest()==inf['noise_sha256']
            assert sha(p/'observations'/f'{pre:04d}.npz')==inf['observation_sha256']
            executed=steps[local-1:local+49];predicted=np.load(p/f'actions-{pre:04d}.npy')
            assert np.array_equal(predicted[:len(executed)],np.array([x['action'] for x in executed],np.float32))
        with av.open(str(p/'rollout.mp4')) as c:
            s=c.streams.video[0]
            assert s.frames==len(steps) and float(s.average_rate)==20 and (s.width,s.height)==(768,288)
        checked.append(dict(id=job['id'],actions=len(steps),noise_tensors=len(r['inferences']),
            video_frames=len(steps),public_actions_equal_postprocessor=True))
    dump(OUT/'execution-audit.json',dict(passed=True,checked=checked,
        video_semantics='New video frame i-1 is AFTER actual public action i;20fps; no reset frame inserted. Historical video references retain their original reset frame.'))


def contact_metrics(result):
    """Recompute recovery categories for the selected SAME door, excluding prefix."""
    p=Path(result['job']['folder']);target=result['job'].get('target_door')
    if not target:return result.get('peaks',{})
    if not all((p/n).exists() for n in ['geometry-spec.json','geometry-catalog.json','contacts.jsonl','steps.jsonl']) or not result.get('execution_validity'):
        return dict(result.get('peaks',{}),technical_incomplete=True)
    spec=read(p/'geometry-spec.json');catalog=read(p/'geometry-catalog.json')
    # The saved geom ancestry identifies the task-door body's owning fixture.
    # Include that fixture's complete descendants. Visual-only geoms cannot have
    # active force; cross-check every action against the online fxtr.contact_geoms
    # peak, which was recorded independently by the worker.
    body=spec['doors'][target]['body_name'];roots=set()
    for g in catalog:
        if g['id'] in spec['doors'][target]['geoms']:
            chain=g['body_ancestors'];roots.add(chain[chain.index(body)+1])
    assert len(roots)==1
    root=next(iter(roots));all_ids={g['id'] for g in catalog if root in g['body_ancestors']}
    robot={g['id'] for g in catalog if g['robot']}
    door_ids=set(spec['doors'][target]['geoms']);hand=set(spec['gripper_geoms'])
    peaks=dict(all=0.,self=0.,target_hand_door=0.,fridge=result['peaks']['fridge'],other_fridge=0.)
    first=None
    steps={r['step']:r for r in rows(p/'steps.jsonl')}
    for row in rows(p/'contacts.jsonl'):
        if row['step']<=result['job']['prefix']:continue
        con=contacts(row['samples'],robot,all_ids,{target:door_ids},hand)
        assert abs(con['peaks']['fridge']-steps[row['step']]['contact']['peaks']['fridge'])<1e-9
        for k in ['all','self','target_hand_door']:peaks[k]=max(peaks[k],con['peaks'][k])
        peaks['other_fridge']=max(peaks['other_fridge'],con['peaks']['other_fridge'])
        if first is None and con['first'][target] is not None:first=con['first'][target]-result['origin_time']
    peaks['first_target_recontact_seconds']=first
    return peaks


def main():
    plan=read(OUT/'registered-plan.json');index=read(OUT/'data-index.json')
    executions=rows(OUT/'execution-list.jsonl') if (OUT/'execution-list.jsonl').exists() else []
    results=[r['result'] for r in executions];collect=[r for r in results if r['job']['kind']=='collection']
    assert [(r['job']['seed'],r['job']['replicate']) for r in collect]==[(r['seed'],r['replicate']) for r in plan['collection_order'][:len(collect)]]
    assert len(collect)<=24
    execution_audit(results)
    collection_status=read(OUT/'collection-status.json') if (OUT/'collection-status.json').exists() else dict(status='incomplete')
    training=read(OUT/'training/complete.json') if (OUT/'training/complete.json').exists() else None
    models=['frozen','ordinary_300','recontact_mix_300']
    recovery=[r for r in results if r['job']['kind']=='recovery']
    for r in recovery:r['reported_contacts']=contact_metrics(r)
    groups={}
    for r in recovery:groups.setdefault((r['job']['seed'],r['job']['replicate']),{})[r['job']['model']]=r
    paired=[]
    for key,g in sorted(groups.items()):
        complete=all(m in g and g[m]['execution_validity'] and g[m]['restoration_validity'] for m in models)
        if complete:
            fingerprints=[{x['local_pre_action']:x['noise_sha256'] for x in g[m]['inferences']} for m in models]
            for pre in set.intersection(*(set(x) for x in fingerprints)):
                assert len({x[pre] for x in fingerprints})==1,'Paired noise differs'
        ref=g.get('frozen');limit=1.25*ref['peaks']['all']+25 if ref and ref['execution_validity'] else None
        for m,r in g.items():
            r['relative_ceiling_N']=limit;r['relative_pass']=limit is not None and r['peaks']['all']<=limit
            r['qualified_success']=bool(r['execution_validity'] and r['restoration_validity'] and r['success'] and not r['hard_stop'] and r['relative_pass'])
        paired.append(dict(seed=key[0],replicate=key[1],complete=complete,
            models={m:dict(valid=r['execution_validity'],raw_success=r['success'],qualified_success=r['qualified_success'],
                hard_stop=r.get('hard_stop',False),relative_pass=r['relative_pass'],relative_ceiling_N=limit,
                actions=r['actions'],initial_deg=r.get('initial_deg'),final_deg=r.get('final_deg'),min_deg=r.get('min_deg'),
                contacts=r['reported_contacts'],first_success_seconds=r.get('first_success_seconds'),
                inference_seconds=r.get('inference_seconds'),execution_seconds=r.get('execution_seconds'),worker_seconds=r.get('worker_seconds'),
                video=str(Path(r['job']['folder'])/'rollout.mp4')) for m,r in g.items()}))
    full=[r for r in results if r['job']['kind']=='full'];full_table=[]
    for seed in plan['test_seeds']:
        g={r['job']['model']:r for r in full if r['job']['seed']==seed}
        full_table.append(dict(seed=seed,models={m:dict(valid=r['execution_validity'],success=r['success'],hard_stop=r['hard_stop'],
            actions=r['actions'],final_deg=r.get('final_deg'),video=str(Path(r['job']['folder'])/'rollout.mp4')) for m,r in g.items()},
            anchor=g.get('frozen',{}).get('first_causal_anchor')))
    complete=[r for r in paired if r['complete']];N=len(complete)
    successes={m:sum(r['models'][m]['qualified_success'] for r in complete) for m in models}
    raw={m:sum(r['models'][m]['raw_success'] for r in complete) for m in models}
    gains={r['seed'] for r in complete if r['models']['recontact_mix_300']['qualified_success'] and
        any(not r['models'][m]['qualified_success'] for m in ['frozen','ordinary_300'])}
    unique=[r for r in complete if r['models']['recontact_mix_300']['qualified_success'] and
        all(not r['models'][m]['qualified_success'] for m in ['frozen','ordinary_300'])]
    regression_complete=all(all(m in r['models'] and r['models'][m]['valid'] for m in ['frozen','recontact_mix_300']) for r in full_table)
    lost=[r['seed'] for r in full_table if r['models'].get('frozen',{}).get('success') and
        not r['models'].get('recontact_mix_300',{}).get('success')]
    mix_hard=sum(r['hard_stop'] for r in results if r['job']['model']=='recontact_mix_300')
    accepted=bool(N>=8 and len({r['seed'] for r in complete})>=4 and
        all(successes['recontact_mix_300']-successes[m]>=math.ceil(.25*N) for m in ['frozen','ordinary_300']) and
        len(gains)>=2 and mix_hard==0 and regression_complete and not lost)
    if collection_status['status']=='data_insufficient':outcome='data_insufficient_no_training'
    elif (OUT/'evaluation-finished.json').exists() and training and all(r['execution_validity'] for r in recovery+full):outcome='completed_positive' if accepted else 'completed_negative'
    else:outcome='technical_or_budget_incomplete'
    if outcome=='data_insufficient_no_training':
        assert len(collect)==24 and not (OUT/'training').exists() and not recovery and not full
    result=dict(outcome=outcome,time=time.time(),elapsed_seconds=time.time()-plan['start_unix'],
        collection_status=collection_status,new_collection_attempts=len(collect),new_collection_successes=sum(r['execution_validity'] and r['success'] and not r['hard_stop'] for r in collect),
        new_invalid=sum(not r['execution_validity'] for r in collect),new_hard_stops=sum(r['hard_stop'] for r in collect),
        qualified_sources=index['qualified_source_count'],qualified_reset_scenes=index['qualified_seeds'],
        recontact_events=index['event_count'],training_samples=index['sample_count'],
        new_optimizer_updates=training['updates'] if training else (read(OUT/'training/incomplete.json')['updates'] if (OUT/'training/incomplete.json').exists() else 0),
        ordinary_control='Reused historical task-ID step300; no new ordinary updates.',
        recovery_paired_N=N,raw_successes=raw if training else {m:None for m in models},
        qualified_successes=successes if training else {m:None for m in models},paired=paired,full_task=full_table,
        recovery_evaluation_status='executed' if recovery else 'not_run_data_gate_failed' if outcome=='data_insufficient_no_training' else 'not_run',
        full_task_regression_status='executed' if full else 'not_run_data_gate_failed' if outcome=='data_insufficient_no_training' else 'not_run',
        gains_reset_seeds=sorted(gains),lost_baseline_success_seeds=lost,new_mix_hard_stops=mix_hard if training else None,
        prototype_acceptance=accepted if training else None,
        inference_total_seconds=sum(r.get('inference_seconds',0) for r in results),worker_total_seconds=sum(r.get('worker_seconds',0) for r in results),
        limitations=[index['caveat'],'No statistical significance or broad generalization claim.','Below1500N is not validated hardware safety.'])
    dump(OUT/'result.json',result)
    dump(OUT/'recovery-evaluation.json',paired);dump(OUT/'full-task-regression.json',full_table)
    fields=['id','seed','replicate','valid','success','hard_stop','actions','qualified_source','event_count','initial_deg','final_deg','peak_all_N','peak_target_hand_door_N','worker_seconds']
    by_id={s['id']:s for s in index['sources']}
    with (OUT/'collection-attempts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in collect:
            s=by_id.get(r['job']['id'],{});p=r.get('peaks',{})
            writer.writerow(dict(id=r['job']['id'],seed=r['job']['seed'],replicate=r['job']['replicate'],valid=r['execution_validity'],
                success=r['success'],hard_stop=r['hard_stop'],actions=r['actions'],qualified_source=s.get('qualified',False),event_count=len(s.get('events',[])),
                initial_deg=json.dumps(r.get('initial_deg')),final_deg=json.dumps(r.get('final_deg')),peak_all_N=p.get('all'),
                peak_target_hand_door_N=p.get('target_hand_door'),worker_seconds=r['worker_seconds']))
    title={'data_insufficient_no_training':'恢复数据覆盖不足，未进入训练',
        'completed_positive':'训练与物理评测完成，通过本轮原型推进门槛',
        'completed_negative':'训练与物理评测完成，未通过本轮原型推进门槛',
        'technical_or_budget_incomplete':'技术原因或预算导致 incomplete'}[outcome]
    lines=['# Recontact Self-Imitation v1','',f'**{title}。**','',
        f'新增收集 {len(collect)}/24 次；有效无硬停止成功 {result["new_collection_successes"]}；技术 invalid {result["new_invalid"]}；硬停止 {result["new_hard_stops"]}。',
        f'合格来源 {index["qualified_source_count"]} 条，覆盖 {len(index["qualified_seeds"])} 个 reset 场景 {index["qualified_seeds"]}；合格再接触窗口 {index["event_count"]} 个，含完成段训练样本 {index["sample_count"]} 个。',
        f'训练门槛固定为 8 条来源 / 4 个场景。本轮实际 optimizer updates：{result["new_optimizer_updates"]}。',
        '', '## 冻结口径','',
        '原模型、原 processor、CloseFridge、H50、10-step Flow Matching、20Hz。每个新完整任务最多600 actions，单worker最多600秒；噪声规则和执行顺序见 registered-plan.json。',
        '同一目标门先有有效手部/夹爪接触，再连续至少5步缺失接触，才构成因果锚点；未来最多50个实际动作内同门再接触，净关闭≥2°或窗口内真实成功，并且来源最终真实成功且无硬停止。所有门槛均未放宽。',
        '额外观察捕获不触发推理。样本为原始三路RGB+16D state+指令到未来最多50个已执行的postprocessor12D动作；不要求未来每步RGB。时间padding与内部32D中的20个padding维度分别排除。',
        '接触分项中的 other_fridge 指其余机器人—冰箱接触，包括机械臂连杆接触目标门；它不等于另一扇门，也不计作有效手部/夹爪再接触。精确部位以 geometry-catalog 和 contacts 中的 geom identity 为准。',
        '历史47的pre101窗口101–150后续在163成功；pre151的13步是完成段，不是新的接触缺失锚点。历史49的pre301有29个实际动作并在窗口内成功。',
        '普通对照已按配方与来源匹配，登记为历史task-ID step300推理权重；历史optimizer状态不完整，不作续训。新候选若训练则从原始权重开始。',
        '测试60–65与本轮收集场景无冲突；原461条演示metadata没有环境reset seed，不能把dataset episode index当成seed，原数据/预训练场景重叠未知。120–123未启用。','',
        '## 所有新收集尝试','',
        '| Seed | 噪声序列 | 有效 | 真实成功 | Actions | 合格来源 | 再接触窗口 | 硬停止 | 峰值全部机器人 N |',
        '|---|---:|---|---|---:|---|---:|---|---:|']
    for r in collect:
        s=by_id.get(r['job']['id'],{})
        lines.append(f'| {r["job"]["seed"]} | {r["job"]["replicate"]} | {r["execution_validity"]} | {r["success"]} | {r["actions"]} | {s.get("qualified",False)} | {len(s.get("events",[]))} | {r["hard_stop"]} | {r.get("peaks",{}).get("all",0):.3f} |')
    lines+=['','## 合格再接触事件','', '| 来源 | 场景 | 锚点/动作窗口 | 连续缺失动作数 | 首次再接触 | 门角起→终 ° | 窗口内成功 |', '|---|---:|---|---:|---:|---|---|']
    for s in index['sources']:
        for e in s['events']:
            lines.append(f'| {s["id"]} | {s["seed"]} | pre{e["start"]}: {e["start"]}–{e["end"]} | {e["prior_no_contact_actions"]} | {e["first_recontact"]} | {e["start_deg"]:.6f}→{e["end_deg"]:.6f} | {e["success_in_window"]} |')
    lines+=['','## 物理评测与回归','']
    if not training:
        lines+=['未进入新训练；恢复benchmark与完整任务回归均未执行。三个模型本轮成功数均为 N/A，不能写成0次成功/0次失败的评测结果。']
    else:
        lines += [f'完整匹配恢复条件 N={N}。原始成功数 {raw}；含相对接触验收的成功数 {successes}。',
            f'增益场景 {sorted(gains)}；候选新增硬停止 {mix_hard}；丢失基线成功场景 {lost}；推进门槛={accepted}。',
            '', '| 场景/重复 | 完整匹配 | frozen 原始/合格 | ordinary_300 原始/合格 | mix 原始/合格 |','|---|---|---|---|---|']
        for p in paired:
            values=[str(p['seed'])+'/'+str(p['replicate']),str(p['complete'])]
            values += [str(p['models'][m]['raw_success'])+'/'+str(p['models'][m]['qualified_success']) if m in p['models'] else 'N/A' for m in models]
            lines.append('| '+' | '.join(values)+' |')
        lines+=['','| 完整任务场景 | frozen | mix | 恢复起点 |','|---|---|---|---|']
        for p in full_table:lines.append('| '+str(p['seed'])+' | '+' | '.join(str(p['models'].get(m,{}).get('success','N/A')) for m in ['frozen','recontact_mix_300'])+' | '+str(p['anchor'])+' |')
    if outcome=='technical_or_budget_incomplete':lines+=['','未完成原因：`'+json.dumps(collection_status,ensure_ascii=False)+'`。训练/评测缺失项见 result.json，未补跑或更换噪声。']
    lines+=['','## 解释与交付','',
        '这些窗口是成功episode中的再接触行为，不能自动视为长期故障恢复或由新视觉重规划触发。完成段和重叠窗口不增加来源数；同reset不同噪声不增加场景数。',
        f'本轮全部worker累计 {result["worker_total_seconds"]:.3f}s，推理累计 {result["inference_total_seconds"]:.3f}s；总墙钟 {result["elapsed_seconds"]:.3f}s。',
        f'物理输出：`{OUT}`。冻结方案、数据索引、实际执行清单、所有attempt、result.json与video-index.html均在此目录。',
        '未训练或未达到验收不产生成功剪辑。未开启几何教师、seed48加时、其他checkpoint搜索或额外训练。没有commit/push。']
    report=ROOT/'environment/recontact-self-imitation-v1.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    videos=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Recontact Self-Imitation v1</title>',
        '<style>body{font:16px system-ui;max-width:1500px;margin:24px auto;background:#111;color:#eee}video{width:720px;max-width:100%}section{border-top:1px solid #555;margin:24px 0;padding:12px}.pair{display:flex;flex-wrap:wrap;gap:16px}a{color:#7ad}</style>',
        '<h1>'+html.escape(title)+'</h1><p>真实执行视频，原20Hz，无变速或插帧。历史来源单独标记。接触事件依据物理记录，不依赖视频猜测。</p>']
    if unique:
        first=sorted(unique,key=lambda p:(plan['test_seeds'].index(p['seed']),p['replicate']))[0]
        videos+=['<h2>首个预登记顺序的候选独有合格成功：匹配 Before / After</h2><div class="pair">']
        for m in ['frozen','recontact_mix_300']:
            video=Path(first['models'][m]['video']);videos.append('<div><p>'+m+'</p><video controls preload="none" src="'+html.escape(video.as_uri())+'"></video></div>')
        videos+=['</div><p>各视频在自身停止时结束；同一恢复起点、配对噪声、150-action预算。下方保留全部失败。</p>']
    for s in index['sources']:
        video=Path(s['video'])
        if video.exists():videos.append('<section><h2>'+html.escape(s['id'])+'</h2><p>'+html.escape(f'历史={s["historical"]}; success={s["success"]}; qualified_source={s["qualified"]}; actions={s["actions"]}')+'</p><video controls preload="none" src="'+html.escape(video.as_uri())+'"></video></section>')
    for r in results:
        if r['job']['kind']=='collection' and r['execution_validity']:continue
        video=Path(r['job']['folder'])/'rollout.mp4'
        if video.exists():videos.append('<section><h2>'+html.escape(r['job']['id'])+'</h2><p>'+html.escape(f'valid={r["execution_validity"]}; success={r["success"]}; actions={r["actions"]}')+'</p><video controls preload="none" src="'+html.escape(video.as_uri())+'"></video></section>')
    videos.append('</html>');(OUT/'video-index.html').write_text('\n'.join(videos),encoding='utf-8')
    before=read(OUT/'preserved-before.json');changed=[p for p,h in before.items() if not Path(p).is_file() or sha(Path(p))!=h]
    junctions={ROOT/'outputs/agent-workflow-v1/recovery-feasibility-v1':Path('D:/EmbodiedOpsOutputs/recovery-feasibility-v1'),
        ROOT/'outputs/agent-workflow-v1/recovery-feasibility-v1b':Path('D:/EmbodiedOpsOutputs/recovery-feasibility-v1b'),
        ROOT/'outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation':Path('D:/EmbodiedOpsOutputs/instruction-control-v1-evaluation')}
    junction_status={str(p):p.is_junction() and p.resolve()==target.resolve() for p,target in junctions.items()}
    dump(OUT/'preservation.json',dict(passed=not changed and all(junction_status.values()),checked=len(before),changed=changed,junctions=junction_status))
    assert not changed,'Protected inputs changed'
    assert all(junction_status.values()),'Historical junction changed'
    print(json.dumps({k:v for k,v in result.items() if k not in ['paired','full_task']},indent=2),flush=True)


if __name__=='__main__':main()
