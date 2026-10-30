"""Read saved pilot evidence, audit alignment, and deliver all fixed conditions."""
import csv
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import time
import numpy as np
from instruction_control_audit import ROOT, read, dump, sha
from recontact_v2 import OUT, MODELS, assert_sources
from recontact_self_imitation import rows


def audit_execution(r):
    import av
    import torch
    p=Path(r['job']['folder']);job=r['job']
    if not r.get('execution_validity'):
        return dict(id=job['id'],passed=None,reason='Execution invalid; preserved without successful-run assertion')
    steps=rows(p/'steps.jsonl')
    assert len(steps)==r['actions']<=job['cap']
    assert r['worker_seconds']<=600
    assert [s['step'] for s in steps]==list(range(job['prefix']+1,job['prefix']+len(steps)+1))
    assert not any(s['terminated'] or s['truncated'] or s['fridge']['success'] for s in steps[:-1])
    assert bool(steps[-1]['fridge']['success'])==r['success']
    for inf in r['inferences']:
        local=inf['local_pre_action'];pre=inf['pre_action']
        assert inf['seed']==job['noise_seeds'][(local-1)//50]
        noise=torch.load(p/f'noise-{local:04d}.pt',map_location='cpu',weights_only=True)
        assert tuple(noise.shape)==(1,50,32) and torch.isfinite(noise).all()
        assert hashlib.sha256(noise.numpy().tobytes()).hexdigest()==inf['noise_sha256']
        ob=Path(inf.get('observation_path',str(p/'observations'/f'{pre:04d}.npz')))
        assert sha(ob)==inf['observation_sha256']
        if 'actual_observation_path' in inf:
            assert sha(Path(inf['actual_observation_path']))==inf['actual_observation_sha256']
        policy_ob=Path(inf.get('policy_observation_path',str(ob)))
        if 'policy_observation_sha256' in inf:assert sha(policy_ob)==inf['policy_observation_sha256']
        executed=steps[local-1:local+49];predicted=np.load(p/f'actions-{pre:04d}.npy')
        assert np.array_equal(predicted[:len(executed)],np.asarray([s['action'] for s in executed],np.float32))
    with av.open(str(p/'rollout.mp4')) as c:
        stream=c.streams.video[0]
        assert stream.frames==len(steps) and float(stream.average_rate)==20
        assert (stream.width,stream.height)==(768,288)
    return dict(id=job['id'],passed=True,actions=len(steps),inferences=len(r['inferences']),
        video_frames=len(steps),actual_public_actions_equal_chunk_prefix=True)


def prepare_attempts(plan, ledger, invalidations):
    """Copy immutable attempts; select no outcome and overwrite no saved result."""
    conditions={c['id']:c for c in plan['conditions']}
    invalid_paths={str(Path(p).resolve()).casefold():reason for p,reason in invalidations.items()}
    assert all(isinstance(reason,str) and reason for reason in invalid_paths.values())
    attempts=[];previous={};first_seen=[];folders=set()
    for ledger_index,item in enumerate(ledger,1):
        r=deepcopy(item['result']);job=r['job'];identity=job['id']
        assert identity in conditions,f'Unregistered condition: {identity}'
        for key,value in conditions[identity].items():
            assert job.get(key)==value,f'Attempt changes registered {identity}.{key}'
        folder=str(Path(job['folder']).resolve()).casefold()
        assert folder not in folders,f'Duplicate attempt folder in ledger: {job["folder"]}'
        folders.add(folder)
        r['_original_execution_validity']=r.get('execution_validity')
        r['_original_restoration_validity']=r.get('restoration_validity')
        r['_original_status']=r.get('status');r['_original_stop_reason']=r.get('stop_reason')
        r['_ledger_index']=ledger_index
        r['_attempt_number']=previous[identity]['_attempt_number']+1 if identity in previous else 1
        if folder in invalid_paths:
            r['_invalidation_reason']=invalid_paths[folder]
            r.update(execution_validity=False,status='technical_invalid',stop_reason=invalid_paths[folder])
        if identity in previous:
            earlier=previous[identity]
            assert not (earlier.get('execution_validity') and earlier.get('restoration_validity')),f'Retry of a valid attempt prohibited: {identity}'
        else:first_seen.append(identity)
        previous[identity]=r;attempts.append(r)
    assert first_seen==[c['id'] for c in plan['conditions'][:len(first_seen)]],'First executions differ from registered order'
    assert set(invalid_paths)<=folders,'Invalidation refers to an unrecorded attempt'
    return attempts


def latest_attempts(plan, attempts):
    by_id={r['job']['id']:r for r in attempts}
    return [by_id[c['id']] for c in plan['conditions'] if c['id'] in by_id]


def attempt_records(plan, attempts, invalidations):
    current={r['job']['id']:r['_ledger_index'] for r in latest_attempts(plan,attempts)}
    records=[]
    for r in attempts:
        folder=Path(r['job']['folder']);is_current=current[r['job']['id']]==r['_ledger_index']
        records.append(dict(id=r['job']['id'],ledger_index=r['_ledger_index'],attempt_number=r['_attempt_number'],
            folder=str(folder),result_path=str(folder/'result.json'),
            current=is_current,superseded=not is_current,
            original_execution_validity=r['_original_execution_validity'],
            original_restoration_validity=r['_original_restoration_validity'],
            original_status=r['_original_status'],original_stop_reason=r['_original_stop_reason'],
            execution_validity=r.get('execution_validity'),restoration_validity=r.get('restoration_validity'),
            status=r.get('status'),reason=r.get('stop_reason'),
            invalidation_reason=r.get('_invalidation_reason'),
            literal_success=bool(r.get('success')),actions=r.get('actions',0),
            video=str(folder/'rollout.mp4') if (folder/'rollout.mp4').exists() else None))
    valid=lambda r:bool(r['execution_validity'] and r['restoration_validity'])
    summary=dict(total_attempts=len(records),distinct_conditions=len(current),planned_conditions=len(plan['conditions']),
        extra_infrastructure_attempts=len(records)-len(current),
        valid_attempts=sum(valid(r) for r in records),invalid_attempts=sum(not valid(r) for r in records),
        current_valid_conditions=sum(r['current'] and valid(r) for r in records),
        superseded_attempts=sum(r['superseded'] for r in records),
        explicitly_invalidated_attempts=sum(bool(r['invalidation_reason']) for r in records),
        ledger_path=str(OUT/'execution-list.jsonl'),invalidations_path=str(OUT/'invalidated-attempts.json') if invalidations else None,
        selection_rule='Latest ledger attempt for each fixed condition; retries require the preceding attempt to be infrastructure-invalid. No selection by success, force, angle or other policy outcome.')
    return records,summary


def io_retry_repeatability(attempts):
    failed=[r for r in attempts if 'WinError 5' in str(r.get('stop_reason'))]
    if not failed:return None
    old=failed[0];later=[r for r in attempts if r['job']['id']==old['job']['id'] and r['_ledger_index']>old['_ledger_index'] and r.get('execution_validity')]
    if not later:return dict(status='No completed valid retry')
    new=later[-1];a=Path(old['job']['folder']);b=Path(new['job']['folder'])
    aa=rows(a/'steps.jsonl');bb=rows(b/'steps.jsonl');errors={}
    for field in ['action','controller_input','state']:
        errors[field]=float(np.max(np.abs(np.asarray([r[field] for r in aa[:50]])-np.asarray([r[field] for r in bb[:50]]))))
    for field in ['qpos','qvel','time']:
        errors[field]=float(np.max(np.abs(np.asarray([r['physical'][field] for r in aa[:50]])-np.asarray([r['physical'][field] for r in bb[:50]]))))
    assert all(v==0 for v in errors.values())
    cameras={}
    with np.load(a/'observations/0051.npz') as x,np.load(b/'observations/0051.npz') as y:
        assert np.array_equal(x['state'],y['state'])
        for key in ['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']:
            d=np.abs(x[key].astype(np.int16)-y[key].astype(np.int16));cameras[key]=dict(max=int(d.max()),mean=float(d.mean()))
    old_noise={r['local_pre_action']:r['noise_sha256'] for r in old.get('inferences',[])}
    new_noise={r['local_pre_action']:r['noise_sha256'] for r in new.get('inferences',[])}
    assert all(new_noise.get(k)==v for k,v in old_noise.items())
    assert all(sha(a/f'noise-{k:04d}.pt')==sha(b/f'noise-{k:04d}.pt') for k in old_noise)
    return dict(failed_attempt=str(a),valid_retry=str(b),first50_max_errors=errors,
        consumed_noise_tensors_equal=len(old_noise),pre51_state_equal=True,pre51_rgb_difference=cameras,
        limitation='Exact noise and first50 actions/physics do not guarantee bitwise identical entire closed-loop continuation: tiny real RGB readback differences precede later policy divergence. No observations replaced or gates relaxed.')


def table_rows(plan, executions):
    by_id={r['job']['id']:r for r in executions}
    output=[]
    for condition in plan['conditions']:
        r=by_id.get(condition['id']);key=f'{condition["kind"]}-{condition["seed"]}-r{condition["replicate"]}-frozen'
        ref=by_id.get(key)
        limit=1.25*ref['peaks']['all']+25 if ref and ref.get('execution_validity') and ref.get('restoration_validity') else None
        row=dict(condition,attempted=r is not None,execution_validity=False,restoration_validity=False,
            literal_success=None,raw_success=False,qualified_success=False,relative_ceiling_N=limit,
            relative_pass=None,hard_stop=None,actions=0,status='not_executed',stop_reason='Not executed',
            initial_deg=None,final_deg=None,min_deg=None,peaks=None,video=None)
        if r:
            valid=bool(r.get('execution_validity') and r.get('restoration_validity'))
            raw=bool(valid and r['success'] and not r.get('hard_stop'))
            relative=bool(r['peaks']['all']<=limit) if limit is not None and 'peaks' in r else None
            row.update({k:r.get(k) for k in ['execution_validity','restoration_validity','hard_stop','actions',
                'status','stop_reason','initial_deg','final_deg','min_deg','min_abs_deg','peaks','per_door_peaks_N','first_contact_seconds',
                'first_recontact_seconds','first_recontact_action','recontact_events','first_target_recontact_seconds','first_success_action','first_success_global_action','first_success_seconds','global_end_action',
                'inference_seconds','execution_seconds','replay_seconds','worker_seconds','prefix_actions',
                'max_replay_errors','boundary_camera_errors']})
            row.update(literal_success=bool(r['success']),raw_success=raw,
                qualified_success=bool(raw and relative),relative_pass=relative,
                video=str(Path(r['job']['folder'])/'rollout.mp4') if (Path(r['job']['folder'])/'rollout.mp4').exists() else None,
                result_path=str(Path(r['job']['folder'])/'result.json'),
                attempt_number=r.get('_attempt_number',1),attempt_ledger_index=r.get('_ledger_index'),
                attempt_folder=r['job']['folder'])
        output.append(row)
    return output


def summarize(table, kind):
    selected=[r for r in table if r['kind']==kind]
    triples=[]
    keys=list(dict.fromkeys((r['seed'],r['replicate']) for r in selected))
    for seed,rep in keys:
        group={r['model']:r for r in selected if (r['seed'],r['replicate'])==(seed,rep)}
        triples.append(dict(seed=seed,replicate=rep,complete=all(r['execution_validity'] and r['restoration_validity'] for r in group.values()),
            raw={m:group[m]['raw_success'] for m in MODELS},qualified={m:group[m]['qualified_success'] for m in MODELS}))
    result=dict(conditions=len(selected),complete_triples=sum(t['complete'] for t in triples),triples=triples)
    for m in MODELS:
        rs=[r for r in selected if r['model']==m]
        result[m]=dict(planned=len(rs),attempted=sum(r['attempted'] for r in rs),
            valid=sum(bool(r['execution_validity'] and r['restoration_validity']) for r in rs),
            literal_success=sum(r['literal_success'] is True for r in rs),raw_success=sum(r['raw_success'] for r in rs),
            qualified_success=sum(r['qualified_success'] for r in rs),hard_stops=sum(r['hard_stop'] is True for r in rs),
            invalid=sum(r['attempted'] and not (r['execution_validity'] and r['restoration_validity']) for r in rs))
    result['gains']={m:[dict(seed=t['seed'],replicate=t['replicate']) for t in triples if t['complete'] and t['qualified']['recontact_mix_300'] and not t['qualified'][m]] for m in MODELS[:2]}
    result['regressions']={m:[dict(seed=t['seed'],replicate=t['replicate']) for t in triples if t['complete'] and not t['qualified']['recontact_mix_300'] and t['qualified'][m]] for m in MODELS[:2]}
    return result


def video_index(table, result):
    groups={}
    for r in table:groups.setdefault((r['kind'],r['seed'],r['replicate']),{})[r['model']]=r
    improvements=[];failures=[];regressions=[]
    for key,g in groups.items():
        mix=g['recontact_mix_300']
        for model in MODELS[:2]:
            if mix['qualified_success'] and g[model]['execution_validity'] and not g[model]['qualified_success']:
                improvements.append(dict(kind=key[0],seed=key[1],replicate=key[2],before=g[model]['id'],after=mix['id']))
            if g[model]['qualified_success'] and mix['execution_validity'] and not mix['qualified_success']:
                regressions.append(dict(kind=key[0],seed=key[1],replicate=key[2],before=g[model]['id'],after=mix['id']))
        if mix['attempted'] and not mix['qualified_success']:failures.append(mix['id'])
    result['first_qualified_improvement']=improvements[0] if improvements else None
    result['failure_or_regression_cases']=failures
    result['first_regression']=regressions[0] if regressions else None
    by_id={r['id']:r for r in table}
    def video(r):
        if not r['video']:return '<p>No valid video artifact available.</p>'
        relative=Path(r['video']).relative_to(OUT).as_posix()
        peaks=r.get('peaks') or {}
        detail=f'<p>True success: {r.get("literal_success")}; qualified: {r.get("qualified_success")}; robot peak: {fmt(peaks.get("all"))} N; matched ceiling: {fmt(r.get("relative_ceiling_N"))} N; hard stop: {r.get("hard_stop")}; success time: {fmt(r.get("first_success_seconds"))} s.</p>'
        return detail+f'<video controls preload="metadata" src="{html.escape(relative)}"></video>'
    parts=['<!doctype html><meta charset="utf-8"><title>Recontact v2 video index</title><style>body{font:16px system-ui;margin:24px;max-width:1500px} .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%}section{border-top:1px solid #aaa;margin-top:30px}code{overflow-wrap:anywhere}</style>',
        '<h1>Recontact Self-Imitation v2 — actual physical executions</h1>',
        '<p>20 fps, one frame per executed public action; no interpolation or time scaling. Raw and contact-qualified success are separate. Recovery rows are familiar-scene whole-held-out-source conditions.</p>',
        f'<p>{html.escape(result["conclusion"])}</p>']
    if improvements:
        first=improvements[0];parts.append('<h2>First qualified improvement in registered condition order</h2><div class="grid">')
        for side in ['before','after']:
            r=by_id[first[side]];parts.append(f'<div><h3>{side}: {html.escape(r["id"])}</h3>{video(r)}</div>')
        parts.append('</div>')
    else:parts.append('<p>No qualified improvement pair is available.</p>')
    if regressions:
        first=regressions[0];parts.append('<h2>First regression in registered condition order</h2><div class="grid">')
        for side in ['before','after']:
            r=by_id[first[side]];parts.append(f'<div><h3>{side}: {html.escape(r["id"])}</h3>{video(r)}</div>')
        parts.append('</div>')
    if failures:parts.append('<p>Failure/regression cases are retained below, including '+html.escape(failures[0])+'.</p>')
    for key,g in groups.items():
        parts.append(f'<section><h2>{key[0]} seed {key[1]}, noise replicate {key[2]}</h2><div class="grid">')
        for m in MODELS:
            r=g[m];parts.append(f'<div><h3>{m}</h3><p>valid={r["execution_validity"]}; raw={r["raw_success"]}; qualified={r["qualified_success"]}; actions={r["actions"]}</p>{video(r)}<p>{html.escape(str(r["stop_reason"]))}</p></div>')
        parts.append('</div></section>')
    preserved_attempts=[r for r in result.get('attempts',[]) if r['superseded'] or not (r['execution_validity'] and r['restoration_validity'])]
    if preserved_attempts:
        parts.append('<section><h2>Preserved infrastructure-invalid attempts</h2><p>These original attempts remain available and are excluded from valid policy scoring. Reference-only observation reads advanced observable clocks by 4ms; affected candidate attempts were invalidated before identical-condition retries. The current condition table uses the latest attempt, never the most successful attempt.</p>')
        for r in preserved_attempts:
            result_link=Path(r['result_path']).relative_to(OUT).as_posix()
            parts.append(f'<h3>{html.escape(r["id"])} / attempt {r["attempt_number"]}</h3><p>current={r["current"]}; superseded={r["superseded"]}; status={html.escape(str(r["status"]))}; original literal success={r["literal_success"]}</p><p>{html.escape(str(r["reason"]))}</p><p><a href="{html.escape(result_link)}">Original immutable result.json</a></p>{video(r)}')
        parts.append('</section>')
    (OUT/'video-index.html').write_text('\n'.join(parts),encoding='utf-8')


def fmt(value):
    if value is None:return '—'
    if isinstance(value,dict):return '; '.join(f'{k}={v:.4f}' if isinstance(v,(int,float)) else f'{k}={v}' for k,v in value.items())
    if isinstance(value,float):return f'{value:.4f}'
    return str(value)


def main():
    plan=read(OUT/'registered-plan.json')
    source_integrity_error=None
    try:assert_sources(plan)
    except Exception as exc:source_integrity_error=str(exc)
    ledger=rows(OUT/'execution-list.jsonl') if (OUT/'execution-list.jsonl').exists() else []
    invalidations=read(OUT/'invalidated-attempts.json') if (OUT/'invalidated-attempts.json').exists() else {}
    attempts=prepare_attempts(plan,ledger,invalidations)
    audits=[]
    for r in attempts:
        try:
            assert source_integrity_error is None,source_integrity_error
            item=audit_execution(r)
            if item.get('passed') is True:
                from recontact_v2_evidence import audit_contacts
                item['contact_and_boundary']=audit_contacts(r)
                assert item['contact_and_boundary']['passed'],str(item['contact_and_boundary'].get('errors'))
            item.update(folder=r['job']['folder'],attempt_number=r['_attempt_number'],ledger_index=r['_ledger_index'])
            if not r.get('execution_validity'):item['reason']=r.get('stop_reason')
            audits.append(item)
        except Exception as exc:
            reason=f'Post-run integrity audit failed: {type(exc).__name__}: {exc}'
            audits.append(dict(id=r['job']['id'],folder=r['job']['folder'],attempt_number=r['_attempt_number'],
                ledger_index=r['_ledger_index'],passed=False,reason=reason))
            r.update(execution_validity=False,status='technical_invalid',stop_reason=reason)
    results=latest_attempts(plan,attempts)
    by_group={}
    for r in results:
        if r.get('execution_validity') and r.get('restoration_validity'):
            by_group.setdefault((r['job']['kind'],r['job']['seed'],r['job']['replicate']),[]).append(r)
    for group in by_group.values():
        hashes=[{i['local_pre_action']:i['noise_sha256'] for i in r.get('inferences',[])} for r in group]
        for pre in set.union(*(set(h) for h in hashes)):
            if len({h[pre] for h in hashes if pre in h})!=1:
                for r in group:
                    r.update(execution_validity=False,status='technical_invalid',stop_reason='Paired native noise tensors differ')
                    audits.append(dict(id=r['job']['id'],folder=r['job']['folder'],attempt_number=r['_attempt_number'],
                        ledger_index=r['_ledger_index'],passed=False,reason='Paired native noise tensors differ'))
    attempt_list,attempt_summary=attempt_records(plan,attempts,invalidations)
    dump(OUT/'execution-audit.json',dict(passed=not any(a['passed'] is False for a in audits),checked=audits,
        attempts_summary=attempt_summary,invalidated_attempts=invalidations,
        video_semantics='frame i-1 is after local public action i;20fps; no inserted reset frame; lossy video is not raw camera reference'))
    table=table_rows(plan,results);full=summarize(table,'full');recovery=summarize(table,'recovery')
    complete=full['complete_triples']==6
    benefit=complete and all(full['recontact_mix_300']['qualified_success']>full[m]['qualified_success'] for m in MODELS[:2])
    secondary=recovery['complete_triples']==6 and all(recovery['recontact_mix_300']['qualified_success']>recovery[m]['qualified_success'] for m in MODELS[:2])
    gainseeds={g['seed'] for cases in full['gains'].values() for g in cases}
    preserved=all(not t['raw']['frozen'] or t['qualified']['recontact_mix_300'] for t in full['triples'])
    extra_hard=[]
    for kind,summary in [('full',full),('recovery',recovery)]:
        for t in summary['triples']:
            g={r['model']:r for r in table if (r['kind'],r['seed'],r['replicate'])==(kind,t['seed'],t['replicate'])}
            if g['recontact_mix_300']['hard_stop'] and not g['frozen']['hard_stop']:extra_hard.append([kind,t['seed'],t['replicate']])
    demo=complete and all(full['recontact_mix_300']['qualified_success']>=full[m]['qualified_success']+2 for m in MODELS[:2]) and len(gainseeds)>=2 and preserved and not extra_hard
    all_complete=complete and recovery['complete_triples']==6
    if not all_complete:conclusion='技术或预算不完整；逐条件保留已执行结果，不能形成完整学习收益结论。'
    elif benefit:conclusion='这批数据与固定混合后训练配方在六个完整任务条件中显示探索性物理收益；不代表已证明可靠泛化。'
    elif secondary:conclusion='完整任务未显示相对两个对照的合格收益；仅支持熟悉场景中留出轨迹条件的局部行为改善。'
    else:conclusion='未达到相对两个对照的探索性物理收益标准；关闭当前数据＋当前混合后训练配方的继续加码。'
    result=dict(version=plan['version'],created_unix=time.time(),elapsed_seconds=time.time()-plan['start_unix'],
        conclusion=conclusion,all36_valid=all_complete,full_task=full,recovery=recovery,
        exploratory_full_task_benefit=benefit,secondary_local_benefit=secondary,main_demo_gate_passed=demo,
        main_demo_components=dict(all6_valid_triples=complete,net_gain_at_least2_each=all(full['recontact_mix_300']['qualified_success']>=full[m]['qualified_success']+2 for m in MODELS[:2]),
            gain_seeds=sorted(gainseeds),preserve_frozen_successes=preserved,extra_hard_stops=extra_hard),
        conditions=table,attempts=attempt_list,attempts_summary=attempt_summary,
        invalidated_attempts=invalidations,
        invalidations_sha256=sha(OUT/'invalidated-attempts.json') if (OUT/'invalidated-attempts.json').exists() else None,
        training=read(OUT/'training/complete.json') if (OUT/'training/complete.json').exists() else None,
        training_reload=read(OUT/'training/reload-verification.json') if (OUT/'training/reload-verification.json').exists() else None,
        manifest_sha256=sha(OUT/'registered-plan.json'))
    if (OUT/'training/updates.jsonl').exists():
        sample_by_id={x['id']:x for x in read(OUT/'training-data-index.json')['samples']}
        exposure={}
        for update in rows(OUT/'training/updates.jsonl'):
            kind=sample_by_id[update['sample_id']]['kind'] if update['kind']=='recontact' else 'ordinary'
            item=exposure.setdefault(kind,dict(samples=0,valid_actions=0))
            item['samples']+=1;item['valid_actions']+=update['valid_actions']
        result['actual_exposure_by_sample_kind']=exposure
    result['io_retry_repeatability']=io_retry_repeatability(attempts)
    result['analysis_sources']={str(ROOT/'scripts'/name):sha(ROOT/'scripts'/name) for name in ['recontact_v2_report.py','recontact_v2_evidence.py']}
    video_index(table,result)
    protected=read(OUT/'preserved-before.json');changed=[p for p,h in protected.items() if not Path(p).exists() or sha(Path(p))!=h]
    junctions=read(OUT/'junctions-before.json');junction_changes=[p for p,target in junctions.items() if str(Path(p).resolve())!=target]
    preservation=dict(passed=not changed and not junction_changes,files=len(protected),changed=changed,junction_changes=junction_changes)
    dump(OUT/'preservation-audit.json',preservation);result['preservation']=preservation
    if not preservation['passed']:
        result['conclusion']='历史保护文件或 Junction 完整性核验失败；本次结果技术无效，详见 preservation-audit.json。'
        result['all36_valid']=False
        result['main_demo_gate_passed']=False
    dump(OUT/'result.json',result)
    with (OUT/'condition-results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for r in table for k in r));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in table:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    audit=read(OUT/'data-audit.json');split=read(OUT/'data-split.json')
    report=['# Recontact Self-Imitation v2 — bounded learning pilot','',conclusion,'',
        f'主 Demo 工程推进标准：{demo}。这不代表统计上已经证明可靠泛化。',
        'v1 三场景不足四场景的停止结论保持不变。v2 仅取消该训练入口门槛，未降低成功、重放或接触验收。',
        f'执行尝试：`{json.dumps(attempt_summary,ensure_ascii=False)}`。表内分母仍是每模型每层6个固定条件，额外基础设施尝试不增加条件分母。每条件仅使用账本中最后一次尝试；重试前一次必须已被判定技术无效，未按成功与否挑选结果。',
        '',f'采集分布：`{json.dumps(audit["distribution"],ensure_ascii=False)}`。11/24 为开发采集结果，不进入测试分母。',
        f'完整来源去重与划分：`{json.dumps(split["counts"],ensure_ascii=False)}`。各场景第一条合格新增来源整条留出；同源/完整重放分组不跨集合。',
        '筛选数据使用已保存锚点的三路原始无损 uint8 RGB、16D state 和后续实际执行的12D public actions；未执行尾部及内部20填充维度不参与有效损失。并非每个后续动作都有无损RGB，未复跑补齐。普通任务部分沿用原数据集和 PyAV 视频解码管线，不将视频解码帧称为原始无损观察。窗口重叠不增加独立来源数。',
        '登记配方：原任务461训练episodes；52验证episodes仅作固定诊断。ordinary_300复用匹配配方的历史权重；recontact_mix_300计划独立从原始checkpoint初始化，固定300updates，225普通样本＋75来源平衡筛选样本；不按测试或loss选择checkpoint。实际完成情况以下列训练记录为准。',
        f'训练实际记录：`{json.dumps(result["training"],ensure_ascii=False)}`；按样本类型曝光：`{json.dumps(result.get("actual_exposure_by_sample_kind"),ensure_ascii=False)}`。独立加载验证：`{json.dumps(result["training_reload"],ensure_ascii=False)}`。',
        '', '| 模型 | 完整任务有效 | 原始成功 | 接触合格成功 | 恢复有效 | 原始成功 | 接触合格成功 |', '|---|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:report.append(f'| {m} | {full[m]["valid"]}/6 | {full[m]["raw_success"]}/6 | {full[m]["qualified_success"]}/6 | {recovery[m]["valid"]}/6 | {recovery[m]["raw_success"]}/6 | {recovery[m]["qualified_success"]}/6 |')
    report+=['',f'基础设施 I/O 重试的重复性核验：`{json.dumps(result.get("io_retry_repeatability"),ensure_ascii=False)}`。',f'完整任务收益：`{json.dumps(full["gains"])}`；回归：`{json.dumps(full["regressions"])}`。',
        f'留出恢复收益：`{json.dumps(recovery["gains"])}`；回归：`{json.dumps(recovery["regressions"])}`。',
        '恢复状态来自成功轨迹，属于熟悉场景中的完整留出来源条件，不代表随机失败分布或跨场景恢复泛化。60–65未进入本轮恢复数据或调参；上游原任务数据没有reset seed元数据，不能声称已验证其跨场景独立性。',
        '所有模型在边界清空动作队列。完整前缀重放验证实际controller输入、qpos/qvel、16D state、物理时间/步数及原始相机；历史来源未保存内部controller快照，内部状态由完整重放重建并在分支间比较。首个共同策略输入只在真实相机gate通过后使用，之后逐chunk使用本分支实时观察。',
        '接触按实际geom identities和物理子步活动约束法向力计算：有效手门≥1N；硬上限1500N。相对上限固定为同条件frozen全部机器人接触峰值×1.25＋25N。other_fridge包括未归入目标手门的机器人—冰箱接触，可能是臂部接触同一门。',
        '', '| 条件 | 有效 | literal / raw / qualified | actions | 初始° | 最终° | 最小° | target / other-fridge / self / all N | 首接触 / 再接触 / 成功秒 | 推理 / 执行秒 | relative / hard | 停止原因 |',
        '|---|---|---|---:|---|---|---|---|---|---|---|---|']
    for r in table:
        peaks=r.get('peaks') or {};forces=' / '.join(fmt(peaks.get(k)) for k in ['target_hand_door','other_fridge','self','all'])
        times=' / '.join(fmt(r.get(k)) for k in ['first_contact_seconds','first_recontact_seconds','first_success_seconds'])
        report.append(f'| {r["id"]} | {r["execution_validity"]} | {r["literal_success"]}/{r["raw_success"]}/{r["qualified_success"]} | {r["actions"]} | {fmt(r["initial_deg"])} | {fmt(r["final_deg"])} | {fmt(r["min_deg"])} | {forces} | {times} | {fmt(r.get("inference_seconds"))}/{fmt(r.get("execution_seconds"))} | {r["relative_pass"]}/{r["hard_stop"]} | {str(r["stop_reason"]).replace("|","/")} |')
    if invalidations:
        report+=['',f'基础设施修复：reference分支的reset校验额外调用两次physical(low)，其force-update读取每次推进2ms观察采样计时，共4ms；物理qpos/qvel、controller输入及子步控制可完全一致，返回16D state仍会错相。首次恢复尝试在第1个前缀action处被原state gate拒绝，新增策略动作0。原{len(invalidations)}个完整任务candidate尝试据此判为技术无效，原始文件及视频保留。修复读取副作用后仅按同一模型、场景、噪声、历史前缀和动作预算重试；实际重试完成情况以以下尝试表为准。']
    report+=['','| 条件 / 尝试 | 当前 / 已替代 | 原始有效 → 审计有效 | literal成功 | 状态与原因 | 原始记录 |',
        '|---|---|---|---|---|---|']
    for a in attempt_list:
        valid=bool(a['execution_validity'] and a['restoration_validity'])
        reason=str(a['reason']).replace('|','/').replace('\n',' ')
        report.append(f'| {a["id"]} / {a["attempt_number"]} | {a["current"]} / {a["superseded"]} | {a["original_execution_validity"]} → {valid} | {a["literal_success"]} | {a["status"]}: {reason} | [{Path(a["folder"]).name}]({Path(a["result_path"]).as_posix()}) |')
    report+=['',f'输出目录：`{OUT}`；[视频索引]({(OUT/"video-index.html").as_posix()})；[完整结果]({(OUT/"result.json").as_posix()})；[逐条件表]({(OUT/"condition-results.csv").as_posix()})。',
        f'视频为真实动作后帧，20fps，不变速、不插帧；视频解码图像不作为历史像素参考。首次合格改善：`{json.dumps(result["first_qualified_improvement"])}`。失败/回归全部保留。',
        f'原始保护文件核验：{preservation["files"]}个，unchanged={preservation["passed"]}；Junction未改变。实际总用时{result["elapsed_seconds"]:.1f}秒。未提交或推送Git。']
    (ROOT/'environment/recontact-self-imitation-v2.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['conclusion','all36_valid','exploratory_full_task_benefit','main_demo_gate_passed','elapsed_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
