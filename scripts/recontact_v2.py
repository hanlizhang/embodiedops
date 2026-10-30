"""Pre-register and execute the single bounded v2 pilot, with immutable conditions."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from instruction_control_audit import ROOT, BASE, PROCESSORS, read, dump, sha
from recontact_self_imitation import STUDY, ORDINARY, rows, preserve

OUT = Path('D:/EmbodiedOpsOutputs/recontact-self-imitation-v2')
V1 = Path('D:/EmbodiedOpsOutputs/recontact-self-imitation-v1')
START = 1793217880.0  # 2026-10-28 20:04:40 UTC, before task implementation.
MODELS = ['frozen', 'ordinary_300', 'recontact_mix_300']


def assert_sources(plan):
    expected = dict(plan['source_hashes'])
    amendments = OUT/'runtime-source-amendments.jsonl'
    if amendments.exists():
        for item in rows(amendments):
            p = str(Path(item['path']))
            assert Path(p).parent == ROOT/'scripts' and Path(p).name.startswith('recontact_v2')
            assert expected[p] == item['old_sha256']
            expected[p] = item['new_sha256']
    for p, h in expected.items():
        assert sha(Path(p)) == h, f'Runtime source changed: {p}'


def register():
    from recontact_v2_data import prepare
    assert not OUT.exists(), 'Exclusive v2 output exists'
    assert time.time()+9000 < START+8*3600
    assert shutil.disk_usage(OUT.parent).free > 10*1024**3
    assert shutil.disk_usage(ROOT).free > 2*1024**3
    old = read(V1/'registered-plan.json')
    data = prepare()
    cm = read(STUDY/'C/manifest.json'); split = read(STUDY/'C/split.json')
    updates = rows(STUDY/'C/updates.jsonl')
    historical = rows(ROOT/'outputs/phase1d/training-v2/updates.jsonl')
    assert split == read(ROOT/'outputs/phase1d/training-v2/split.json')
    assert len(updates) == 300 and all(a['step'] == i+1 and a['finite'] for i,a in enumerate(updates))
    assert all(all(a[k] == b[k] for k in ['step','sample_index','episode','lr_used']) for a,b in zip(updates,historical))
    assert cm['instruction_intervention'] == 'CloseFridge' and cm['trainable_count'] == 99880992
    assert len(split['train_episodes']) == 461 and len(split['validation_episodes']) == 52
    assert not set(split['train_episodes']) & set(split['validation_episodes'])
    assert all(importlib.metadata.version(k) == v for k,v in cm['python_packages'].items())
    assert sha(BASE/'model.safetensors') == cm['base_model_sha256'] == old['model_hashes']['frozen']
    assert sha(ORDINARY/'model.safetensors') == sha(STUDY/'C/checkpoints/000300/pretrained_model/model.safetensors') == old['model_hashes']['ordinary_300']
    assert all(sha(BASE/n) == sha(ORDINARY/n) == cm['processor_sha256'][n] for n in PROCESSORS)
    assert read(ROOT/'outputs/instruction-storage-audit/reload/result.json')['passed']
    sources = dict(old['source_hashes'])
    for name in ['recontact_v2.py','recontact_v2_data.py','recontact_v2_train.py','recontact_v2_worker.py']:
        p = ROOT/'scripts'/name; sources[str(p)] = sha(p)
    assert all(sha(Path(p)) == h for p,h in sources.items())
    # Freeze the complete condition table before any training or model testing.
    conditions = []
    for seed in range(60,66):
        for model in MODELS:
            conditions.append(dict(id=f'full-{seed}-r1-{model}',kind='full',seed=seed,replicate=1,
                model=model,prefix=0,target_door=None,cap=600,
                noise_seeds=[42000000+seed*1000+i for i in range(1,601,50)]))
    for state in data['recovery_states']:
        for rep in [1,2]:
            for model in MODELS:
                conditions.append(dict(id=f'recovery-{state["seed"]}-r{rep}-{model}',kind='recovery',
                    seed=state['seed'],replicate=rep,model=model,prefix=state['prefix'],cap=150,
                    target_door=state['target_door'],reference=state['reference'],source_id=state['source_id'],
                    noise_seeds=[43000000+state['seed']*10000+rep*1000+i for i in [1,51,101]]))
    assert len(conditions) == 36 and len({c['id'] for c in conditions}) == 36
    OUT.mkdir()
    dump(OUT/'data-split.json',data['split'])
    dump(OUT/'training-data-index.json',data['training_index'])
    dump(OUT/'holdout-data-index.json',data['holdout_index'])
    dump(OUT/'data-audit.json',data)
    protected = read(V1/'preserved-before.json')
    protected.update(preserve([V1, ROOT/'environment/recontact-self-imitation-v1.md',
        ROOT/'scripts/recontact_self_imitation.py',ROOT/'scripts/recontact_data.py',
        ROOT/'scripts/recontact_train.py',ROOT/'scripts/recontact_report.py']))
    assert all(sha(Path(p)) == h for p,h in protected.items()), 'Historical preservation mismatch'
    dump(OUT/'preserved-before.json',protected)
    junctions = [ROOT/'outputs/agent-workflow-v1/recovery-feasibility-v1',
        ROOT/'outputs/agent-workflow-v1/recovery-feasibility-v1b',STUDY/'evaluation']
    dump(OUT/'junctions-before.json',{str(p):str(p.resolve()) for p in junctions})
    training = dict(old['training'],training_index=str(OUT/'training-data-index.json'),
        training_index_sha256=sha(OUT/'training-data-index.json'),effective_warmup_steps=15,
        no_additional_source_or_scene_count_gate=True)
    plan = dict(version='recontact-self-imitation-v2',start_unix=START,registered_unix=time.time(),
        deadline_unix=START+28800,min_remaining_before_training_seconds=9000,
        evaluation_delivery_reserve_seconds=7200,output=str(OUT),
        training_authorized_without_four_scene_gate=True,
        v1_conclusion='Preserved: 13 qualified sources across only 3 scenes; no v1 training/evaluation.',
        scope='Exploratory existing-data pilot; remove only four-scene training entrance requirement. No new positive collection; no alternative method, ratio, checkpoint, test or noise selection.',
        models=old['models'],model_hashes=old['model_hashes'],processors=old['processors'],
        revision=old['revision'],instruction='CloseFridge',H=50,flow_steps=10,public_hz=20,
        source_hashes=sources,training=training,ordinary_control=old['ordinary_control'],
        test_seeds=list(range(60,66)),reserved_seeds=list(range(120,124)),
        test_scene_audit='60-65 absent from recovery sources and tuning. V1 planned these but did not execute any. Original dataset metadata has no reset seed; episode indices are not scene seeds; upstream scene overlap unknown.',
        recovery_states=data['recovery_states'],conditions=conditions,
        split_file_sha256=sha(OUT/'data-split.json'),holdout_index_sha256=sha(OUT/'holdout-data-index.json'),
        noise_rule=dict(full='42000000 + seed*1000 + local_pre_action',
            recovery='43000000 + seed*10000 + replicate*1000 + local_pre_action',
            native='torch.manual_seed; policy.model.sample_noise((1,50,32),device); save actual tensors and hashes'),
        filtering={k:v for k,v in old['filtering'].items() if k not in ['gate_sources','gate_reset_scenes']},
        evaluation=dict(full_actions=600,recovery_actions=150,replicates=2,
            order='Full: seed60..65, frozen/ordinary_300/recontact_mix_300. Then held-out source44/47/49, replicate1/2, same model order.',
            source='Complete whole-source holdout; first causal saved anchor by recorded past contact/gap, regardless future student behavior. All18 full tasks execute independently of gap existence.',
            replay='Full actual public prefix; exact controller input; qpos/qvel<=1e-8,state<=1e-7; time/count maintained; same scene/controller/cameras. Raw uint8 camera max<=1 and MAE<=.001.',
            controller='Internal boundary states reconstructed through complete replay; historical source has no direct internal snapshot. Validate replayed observable history and record/compare branch boundary internal signatures.',
            horizon='Set full600 or prefix+150, ignore_done=False before sole reset. Honor true success, hard guard, premature termination and integrity failure.',
            policy_context='n_obs_steps1,RTC off; reset empty action queues for all3; predict_action_chunk. First common raw policy input only after actual restoration/camera gate; later inputs branch-current.',
            relative_gate='Per matching kind/seed/replicate: peak_all <=1.25*frozen_peak_all+25 N, using active robot contact normal max. No candidate-dependent reference.',
            scoring='Literal predicate, valid policy success, and qualified success reported separately. Hard stop is policy failure even if predicate true. Infrastructure invalid separate. Qualified requires valid success, no hard stop, valid matched frozen reference, relative gate.',
            inference_scope='Recovery conditions originate in successful trajectories: familiar-scene whole-held-out-source recovery, not random failure distribution or cross-scene recovery generalization.'),
        conclusion_rules=dict(primary='Exploratory task benefit only when mix qualified full-task count exceeds BOTH controls on all6 valid three-way comparisons; report each gain/regression regardless demo gate.',
            secondary='If only secondary improves, restrict claim to familiar-scene held-out trajectory local behavior.',
            demo='All6 full-task triples valid; mix qualified count >= each control+2; gains on>=2 seeds; preserve every frozen raw successful full-task seed; no extra mix hard-stop condition relative to matched frozen across either tier.',
            no_gain='Close additional escalation of current data plus current mixture recipe; do not append training or change test set.'),
        stops=dict(worker_seconds=600,total_hours=8,hard_normal_N=1500,disk_floor_GiB=2,
            max_training_runs=2,actual_planned_training_runs=1,max_conditions=36,
            technical='Preserve all attempts and invalid reasons. No automatic retries, alternate noise/scenes or checkpoint selection; routine infrastructure repair may rerun only the exact condition with full attempt preservation.',
            integrity='Runtime/data/model integrity failures stop dependent work. Implementation-only v2 source repair requires recorded old/new hash and reason; frozen experimental choices cannot change.'),
        cost_basis=dict(v1_collection_24_worker_seconds=2373.441,v1_max_worker_seconds=169.266,
            historical_300_training_seconds=179.906,planned_training_timeout=1800,
            expected_eval_36_seconds=6000,note='At least7200s reserved for physics/delivery; individual physical cap600s.'))
    dump(OUT/'registered-plan.json',plan)
    snapshot=OUT/'executed-source';snapshot.mkdir()
    for i,p in enumerate(sources): (snapshot/f'{i:02d}-{Path(p).name}.txt').write_bytes(Path(p).read_bytes())
    dump(OUT/'ordinary-control-match.json',dict(passed=True,reused=True,manifest=cm,
        matched=['base','instruction','processors','normalization','trainable_modules','precision','optimizer','LR_schedule','effective_batch1','461_train_52_val_split','frame_sampling_seed42','updates300','serialized_weights'],
        inference_reload_evidence=old['ordinary_control']['inference_reload_evidence'],
        optimizer_continuation_required=False,weights_sha256=plan['model_hashes']['ordinary_300']))
    print('V2 registered: fixed split, 300 updates, all36 conditions; no new collection.',flush=True)


def register_repair():
    plan=read(OUT/'registered-plan.json')
    assert time.time()+7200<plan['deadline_unix']
    assert not (OUT/'invalidated-attempts.json').exists()
    ledger=rows(OUT/'execution-list.jsonl')
    affected=[x for x in ledger if x['job']['kind']=='full' and x['job']['model']!='frozen']
    assert len(affected)==12 and all(x['job']['reference'] for x in affected)
    stopped=ledger[-1]
    assert stopped['job']['id']=='recovery-44-r1-frozen' and stopped['result']['actions']==0
    assert not stopped['result']['execution_validity']
    reason='Infrastructure-invalid: reset reference verification called physical() twice, advancing RoboSuite observation sampling clocks by 0.004s without physics. Full candidate observation schedule differed from frozen. Original attempt retained; repeat exact condition after cached-reset fix.'
    invalid={x['job']['folder']:reason for x in affected}
    dump(OUT/'invalidated-attempts.json',invalid)
    folder=OUT/'infrastructure-repair-01';folder.mkdir()
    changes=[]
    for name in ['recontact_v2.py','recontact_v2_worker.py']:
        path=ROOT/'scripts'/name
        change=dict(time=time.time(),path=str(path),old_sha256=plan['source_hashes'][str(path)],
            new_sha256=sha(path),reason='Remove side-effecting diagnostic reads and preserve exact-condition infrastructure retries; no experimental choice or acceptance threshold changed.')
        changes.append(change);(folder/(name+'.txt')).write_bytes(path.read_bytes())
    with (OUT/'runtime-source-amendments.jsonl').open('x',encoding='utf-8') as f:
        for change in changes:f.write(json.dumps(change)+'\n')
    dump(folder/'repair-manifest.json',dict(time=time.time(),cause=reason,
        evidence=dict(failed_attempt=stopped['job']['folder'],replay_errors=rows(Path(stopped['job']['folder'])/'replay.jsonl'),
            first_trace_source=str(Path(stopped['job']['reference'])/'first-action-trace.json'),
            root_cause='physical->_get_observations(force_update=True)->Observable.update increments _time_since_last_sample even for diagnostic forced reads.'),
        invalidated_full_attempts=[dict(id=x['job']['id'],folder=x['job']['folder'],result_sha256=sha(Path(x['job']['folder'])/'result.json')) for x in affected],
        retained_valid_frozen=[x['job']['id'] for x in ledger if x['job']['kind']=='full' and x['job']['model']=='frozen'],
        retry_rule='Revisit original fixed condition order; retain valid policy outcomes, rerun only infrastructure-invalid attempts in new attempt folders. Never retry valid failure or replace seeds/noise/checkpoints.',
        unchanged=dict(registered_plan_sha256=sha(OUT/'registered-plan.json'),models=read(OUT/'frozen-models.json'),
            training_index_sha256=sha(OUT/'training-data-index.json'),conditions=36,full_cap=600,recovery_cap=150,
            worker_seconds=600,deadline_unix=plan['deadline_unix'],all_acceptance_thresholds=plan['evaluation']),
        changes=changes,max_expected_attempts=49,additional_optimizer_updates=0))
    assert_sources(plan)
    print('Infrastructure repair registered; 12 prior full candidates invalidated, 6 frozen baselines retained; exact conditions only.',flush=True)


def register_io_repair():
    plan=read(OUT/'registered-plan.json');ledger=rows(OUT/'execution-list.jsonl')
    failure=ledger[-1]
    assert failure['job']['id']=='full-60-r1-ordinary_300' and failure['job']['attempt']==2
    assert not failure['result']['execution_validity'] and 'WinError 5' in failure['result']['stop_reason']
    expected=dict(plan['source_hashes'])
    for change in rows(OUT/'runtime-source-amendments.jsonl'):expected[change['path']]=change['new_sha256']
    folder=OUT/'infrastructure-repair-02';folder.mkdir()
    changes=[]
    for name in ['recontact_v2.py','recontact_v2_worker.py']:
        p=ROOT/'scripts'/name
        change=dict(time=time.time(),path=str(p),old_sha256=expected[str(p)],new_sha256=sha(p),
            reason='Bounded retry of Windows atomic result-file replacement on sharing violation; progress will be read from append-only logs. No model, physical, observation, action or acceptance change.')
        changes.append(change);(folder/(name+'.txt')).write_bytes(p.read_bytes())
    with (OUT/'runtime-source-amendments.jsonl').open('a',encoding='utf-8') as f:
        for change in changes:f.write(json.dumps(change)+'\n')
    dump(folder/'repair-manifest.json',dict(time=time.time(),failed_attempt=failure['job'],
        reason=failure['result']['stop_reason'],actions_before_infrastructure_failure=failure['result']['actions'],
        file_write_retry_seconds_max=.475,unchanged_registered_plan_sha256=sha(OUT/'registered-plan.json'),
        unchanged_model_manifest_sha256=sha(OUT/'frozen-models.json'),additional_optimizer_updates=0,
        fixed_conditions=36,max_expected_attempts=50,changes=changes))
    assert_sources(plan)
    print('Windows I/O repair registered; preserve failed attempt02 and repeat exact condition as attempt03.',flush=True)


def command_stage(name, args, timeout):
    plan=read(OUT/'registered-plan.json');assert_sources(plan)
    assert time.time()+timeout < plan['deadline_unix']
    meta=OUT/(name+'-process.json');assert not meta.exists()
    started=time.time();result=dict(started_unix=started,command=args,timeout_seconds=timeout)
    with (OUT/(name+'.log')).open('x',encoding='utf-8') as log:
        try:
            p=subprocess.run([sys.executable,'-B','-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            result['returncode']=p.returncode
        except Exception as exc:result.update(returncode=-1,error=str(exc))
    result.update(finished_unix=time.time(),seconds=time.time()-started);dump(meta,result)
    assert result['returncode']==0,f'{name} failed; inspect {name}.log'


def run_job(job):
    plan=read(OUT/'registered-plan.json');assert_sources(plan)
    assert time.time()+600 < plan['deadline_unix']-600, 'Delivery reserve before worker'
    folder=Path(job['folder']);assert not folder.exists()
    folder.parent.mkdir(parents=True,exist_ok=True)
    jobfile=folder.parent/(folder.name+'-job.json');dump(jobfile,job)
    started=time.time()
    with (folder.parent/(folder.name+'-worker.log')).open('x',encoding='utf-8') as log:
        try:
            p=subprocess.run([sys.executable,'-B','-u',str(Path(__file__)),'worker','--job',str(jobfile)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            assert p.returncode==0,'Worker process crashed'
        except Exception as exc:
            folder.mkdir(exist_ok=True)
            r=read(folder/'result.json') if (folder/'result.json').exists() else dict(actions=0,success=False)
            r.update(job=job,status='technical_invalid',execution_validity=False,stop_reason=str(exc),worker_seconds=time.time()-started)
            dump(folder/'result.json',r)
    r=read(folder/'result.json')
    with (OUT/'execution-list.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(job=job,result=r))+'\n')
    print('FINISHED',job['id'],'valid',r.get('execution_validity'),'success',r.get('success'),
        'actions',r.get('actions'),'stop',r.get('stop_reason'),flush=True)
    return r


def evaluate():
    plan=read(OUT/'registered-plan.json');assert_sources(plan)
    assert read(OUT/'training/reload-verification.json')['passed']
    models=read(OUT/'frozen-models.json')
    assert models['data_index_sha256']==sha(OUT/'training-data-index.json')
    started=OUT/'evaluation-started.json'
    if not started.exists():dump(started,dict(time=time.time(),conditions=36))
    invalidated=read(OUT/'invalidated-attempts.json') if (OUT/'invalidated-attempts.json').exists() else {}
    attempts=rows(OUT/'execution-list.jsonl') if (OUT/'execution-list.jsonl').exists() else []
    def previous(identity):return [x for x in attempts if x['job']['id']==identity]
    def valid(entry):
        return (entry['job']['folder'] not in invalidated and entry['result'].get('execution_validity')
            and entry['result'].get('restoration_validity'))
    def latest_valid(identity):
        entries=previous(identity)
        return entries[-1] if entries and valid(entries[-1]) else None
    for condition in plan['conditions']:
        prior=previous(condition['id'])
        if latest_valid(condition['id']):continue
        # Only infrastructure-invalid attempts may be repeated; policy failures remain final.
        assert not prior or not valid(prior[-1]), 'Cannot retry valid policy outcome'
        attempt=len(prior)+1
        suffix='' if attempt==1 else f'-attempt{attempt:02d}'
        job=dict(condition,folder=str(OUT/condition['kind']/(condition['id']+suffix)),attempt=attempt,
            checkpoint=models['paths'][condition['model']],checkpoint_sha256=models['hashes'][condition['model']])
        if condition['kind']=='full':
            baseline=latest_valid(f'full-{condition["seed"]}-r1-frozen')
            if condition['model']!='frozen':
                assert baseline,'Missing valid matched frozen reset reference'
                ref=Path(baseline['job']['folder']);job['reference']=str(ref)
            else:ref=None;job['reference']=None
        else:
            baseline=latest_valid(f'recovery-{condition["seed"]}-r1-frozen')
            ref=Path(baseline['job']['folder']) if baseline else None
        job['paired_boundary_reference']=str(ref) if ref and (ref/'boundary-controller.json').exists() else None
        result=run_job(job);attempts.append(dict(job=job,result=result))
        if not result.get('execution_validity'):
            raise RuntimeError(f"Infrastructure-invalid condition {job['id']}: {result.get('stop_reason')}; preserved for diagnosis before proceeding")
    dump(OUT/'evaluation-finished.json',dict(time=time.time(),conditions=len(plan['conditions']),attempts=len(attempts)))


def run():
    try:
        command_stage('ordinary-verify',[str(ROOT/'scripts/recontact_v2_train.py'),'--verify-ordinary'],600)
        command_stage('mix-train',[str(ROOT/'scripts/recontact_v2_train.py')],1800)
        command_stage('mix-verify',[str(ROOT/'scripts/recontact_v2_train.py'),'--verify'],600)
        evaluate()
    except Exception as exc:
        dump(OUT/'orchestration-stop.json',dict(time=time.time(),reason=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','register_repair','register_io_repair','run','evaluate','worker'])
    ap.add_argument('--job',type=Path);args=ap.parse_args()
    if args.mode=='worker':
        from recontact_v2_worker import worker
        worker(read(args.job))
    else:globals()[args.mode]()
