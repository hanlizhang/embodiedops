"""Bounded recontact study: registration, frozen collection and paired evaluation.

Only this study's exclusive output is written. Existing execution primitives keep
the public action conversion, physics/contact guards and renderer lifecycle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from instruction_control_audit import ROOT, BASE, PROCESSORS, read, dump, sha
from recovery_feasibility import max_error, camera_error

OUT = Path('D:/EmbodiedOpsOutputs/recontact-self-imitation-v1')
STUDY = ROOT/'outputs/agent-workflow-v1/instruction-control-v1/attempt02'
ORDINARY = STUDY/'evaluation/inference-C'
START = 1793130278.0  # 2026-10-27 19:44:38 UTC, before implementation/registration.
CAMERAS = ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
COLLECT_SEEDS = [42, 44, 45, 46, 47, 49]


def rows(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]


def degrees(fridge):
    return {k: float(np.rad2deg(v)) for k, v in fridge['qpos'].items()}


def noise_seed(kind, seed, replicate, pre_action):
    if kind == 'collection': return 1000000 + replicate*100000 + seed*1000 + pre_action
    if kind == 'full': return 20000000 + seed*1000 + pre_action
    assert kind == 'recovery'
    return 30000000 + seed*10000 + replicate*1000 + pre_action


def contacts(samples, robot, fridge, doors, hand):
    peaks = dict(all=0., fridge=0., other_fridge=0., self=0., target_hand_door=0.)
    loaded = {d: False for d in doors}; first = {d: None for d in doors}
    per_door = {d: 0. for d in doors}
    for sample in samples:
        for c in sample['robot_contacts']:
            if c['constraint_address'] < 0: continue
            ids = set(c['geom_ids']); f = max(0., c['force_N_contact_frame'][0])
            peaks['all'] = max(peaks['all'], f)
            if ids.issubset(robot): peaks['self'] = max(peaks['self'], f)
            hit = False
            for d, gs in doors.items():
                if ids & gs and ids & hand:
                    hit = True; per_door[d] = max(per_door[d], f)
                    peaks['target_hand_door'] = max(peaks['target_hand_door'], f)
                    if f >= 1:
                        loaded[d] = True
                        if first[d] is None: first[d] = sample['time']
            if ids & fridge:
                peaks['fridge'] = max(peaks['fridge'], f)
                if not hit: peaks['other_fridge'] = max(peaks['other_fridge'], f)
    return dict(peaks=peaks, loaded=loaded, first=first, per_door_N=per_door)


def eligible(seen, gap, success):
    return [] if success else [d for d in seen if seen[d] and gap[d] >= 5]


def advance(seen, gap, loaded):
    for d in seen:
        if loaded[d]: seen[d] = True; gap[d] = 0
        else: gap[d] += 1


def preserve(paths):
    result = {}
    for path in paths:
        for p in (path.rglob('*') if path.is_dir() else [path]):
            if p.is_file(): result[str(p)] = sha(p)
    return result


def register():
    import shutil
    import pyarrow.parquet as pq
    assert not OUT.exists(), 'Exclusive output already exists'
    assert time.time()-START < 8*3600
    assert shutil.disk_usage(OUT.parent).free > 10*1024**3
    assert shutil.disk_usage(ROOT).free > 2*1024**3
    cm = read(STUDY/'C/manifest.json'); split = read(STUDY/'C/split.json')
    assert split == read(ROOT/'outputs/phase1d/training-v2/split.json')
    updates = rows(STUDY/'C/updates.jsonl'); reference = rows(ROOT/'outputs/phase1d/training-v2/updates.jsonl')
    assert len(updates) == 300 and all(a['step'] == i+1 and a['finite'] for i,a in enumerate(updates))
    assert all(all(a[k] == b[k] for k in ['step','sample_index','episode','lr_used']) for a,b in zip(updates,reference))
    assert cm['instruction_intervention'] == 'CloseFridge' and cm['trainable_count'] == 99880992
    assert cm['base_model_sha256'] == sha(BASE/'model.safetensors')
    assert sha(ORDINARY/'model.safetensors') == sha(STUDY/'C/checkpoints/000300/pretrained_model/model.safetensors') == 'e3cc9a2084f081778b2353bfc8e8c9d027e933e224eca37205be2d2c9715a3c8'
    assert all(sha(BASE/n) == sha(ORDINARY/n) == cm['processor_sha256'][n] for n in PROCESSORS)
    assert read(ROOT/'outputs/instruction-storage-audit/reload/result.json')['passed']
    assert len(split['train_episodes']) == 461 and len(split['validation_episodes']) == 52
    assert not set(split['train_episodes']) & set(split['validation_episodes'])
    data = ROOT/'outputs/phase1a/hf-cache/lerobot/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots/b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4'
    meta = data/'meta/episodes/chunk-000/file-000.parquet'
    columns = pq.read_schema(meta).names
    # Episode indices are dataset row identities, NOT environment reset seeds.
    seed_columns = [k for k in columns if 'seed' in k.lower() or 'layout' in k.lower()]
    assert not seed_columns, 'Additional scene metadata requires split review before registration'
    OUT.mkdir()
    protected = read(Path('D:/EmbodiedOpsOutputs/recovery-completion-a48-v1/preserved-before.json'))
    protected = {str(ROOT/p): h for p,h in protected.items()}
    protected.update(preserve([BASE, ORDINARY, STUDY/'C/manifest.json', STUDY/'C/split.json',
        STUDY/'C/updates.jsonl', ROOT/'outputs/instruction-storage-audit/reload/result.json',
        STUDY/'evaluation/A-ID-47', STUDY/'evaluation/A-ID-49',
        Path('D:/EmbodiedOpsOutputs/recovery-completion-a48-v1')]))
    assert all(sha(Path(p)) == h for p,h in protected.items())
    dump(OUT/'preserved-before.json', protected)
    source_names = ['recontact_self_imitation.py', 'recontact_data.py', 'phase1d_rollout.py','phase1b_contract.py',
        'phase1c_replay.py','control_diagnostics.py','renderer_lifecycle.py','recovery_feasibility.py',
        'instruction_disk_guard.py','phase1d_train.py']
    sources = {str(ROOT/'scripts'/n):sha(ROOT/'scripts'/n) for n in source_names}
    env = Path(sys.prefix)/'Lib/site-packages'
    for n in ['lerobot/policies/smolvla/modeling_smolvla.py','lerobot/envs/robocasa.py',
              'lerobot/optim/factory.py','lerobot/optim/schedulers.py']:
        sources[str(env/n)] = sha(env/n)
    plan = dict(version='recontact-self-imitation-v1', start_unix=START, registered_unix=time.time(),
        deadline_unix=START+8*3600, collection_deadline_unix=START+4*3600,
        min_remaining_before_training_seconds=2.5*3600, evaluation_delivery_reserve_seconds=7200,
        cost_basis=dict(historical_mean_worker_seconds=8018.754/52, historical_300_training_seconds=179.906,
            expected_collection_24_seconds=24*8018.754/52,expected_eval_48_seconds=48*8018.754/52,
            note='Estimates exclude setup, I/O and verification; hard worker timeout remains 600s.'),
        output=str(OUT), models=dict(frozen=str(BASE),ordinary_300=str(ORDINARY)),
        model_hashes=dict(frozen=sha(BASE/'model.safetensors'),ordinary_300=sha(ORDINARY/'model.safetensors')),
        processors={n:sha(BASE/n) for n in PROCESSORS}, source_hashes=sources,
        revision=BASE.name, instruction='CloseFridge', H=50, flow_steps=10, public_hz=20,
        collection_order=[dict(id=f'collect-{seed}-r{r}',seed=seed,replicate=r) for r in range(1,5) for seed in COLLECT_SEEDS],
        historical_sources=[str(STUDY/f'evaluation/A-ID-{s}') for s in [47,49]],
        test_seeds=[60,61,62,63,64,65], excluded_seeds=[120,121,122,123],
        split_audit=dict(new_training_reset_seeds=COLLECT_SEEDS,test_conflicts=[],
            original_metadata_columns=columns,metadata_sha256=sha(meta),
            limitation='Original 461 demonstration episodes have no reset-seed/scene identity; upstream pretraining and demonstration scene overlap unknown. Dataset episode indices are not reset seeds.'),
        noise_rule=dict(generator='torch.manual_seed(seed); native model.sample_noise((1,50,32),device); save actual tensor and byte SHA256',
            collection='1000000 + replicate(1..4)*100000 + reset_seed*1000 + pre_action(1,51,..551)',
            full='20000000 + reset_seed*1000 + pre_action(1,51,..551)',
            recovery='30000000 + reset_seed*10000 + replicate(1..2)*1000 + local_pre_action(1,51,101)'),
        filtering=dict(contact='Active constraint normal force >=1 N at any physics substep; audited gripper0_right colliding geoms and same fridge task-door descendants.',
            anchor='Not success; prior same-door contact; immediately preceding >=5 executed public actions without same-door effective contact. Extra raw capture when gap first reaches 5 in EACH gap, plus all H50 boundaries.',
            positive='Same-door recontact within next <=50 executed actions; end net closure >=2 degrees (absolute joint angle toward zero), OR success in window; source true all-door success <=600, valid, no hard guards.',
            completion='For each qualified event, subsequent saved H50 boundaries through source success may be completion samples, without reclassifying them as causal anchors or increasing source counts. De-duplicate source/pre_action.',
            gate_sources=8,gate_reset_scenes=4, window=50,net_closure_deg=2.,
            samples='Raw anchor three uint8 RGB + float32 16D state + CloseFridge -> saved postprocessor 12D public actions actually executed. Future RGB not required; time padding and 20 internal padded action dimensions excluded from loss.'),
        ordinary_control=dict(reuse=True,path=str(ORDINARY),recipe=cm,
            split_sha256=sha(STUDY/'C/split.json'),config_sha256=sha(STUDY/'C/config/train_config.json'),
            inference_reload_evidence=str(ROOT/'outputs/instruction-storage-audit/reload/result.json'),
            optimizer_state='Incomplete historical optimizer state; inference use only, no continuation.'),
        training=dict(updates=300,seed=42,batch=1,ordinary_fraction=.75,recontact_fraction=.25,
            sampling='Retain historical torch.randperm frame-uniform indices seed42; replace exactly75 of300 slots using np.default_rng(731042).permutation(300)[:75]. Source-episode balanced shuffled cycles, then uniformly select saved sample using same independent NumPy RNG.',
            modules='155 tensors / 99,880,992 params: original Action Expert/state/action modules; VLM frozen',
            precision=cm['precision'],optimizer=cm['optimizer'],scheduler=cm['scheduler'],
            split=str(STUDY/'C/split.json'), checkpoint='step300 fixed; save independently reload/hash/probe validate',
            probes='Original8 train/8 val anchors; torch fork_rng seeds42000..42007 at0 and every50; descriptive only'),
        evaluation=dict(source='Six frozen normal600 runs AFTER weights/data frozen; first causal eligible saved anchor per scene, joint-name order for simultaneous ties. N/A without replacement.',
            order='Source full tasks seed order, then each scene/replicate1,2/model frozen,ordinary_300,recontact_mix_300; finally mix full tasks seed order.',
            recovery_actions=150,full_actions=600,replicates=2,
            replay='Every original public action through anchor-1; exact qpos/qvel<=1e-8,state<=1e-7 and scene/controller equality; raw camera uint8 max<=1,MAE<=0.001 at reset and anchor.',
            horizon='Before reset set horizon=600 for full tasks, prefix+150 for recovery, ignore_done=False; never reset at boundary. Honor real success/done/truncation.',
            policy_context='n_obs_steps1,RTC disabled,predict_action_chunk stateless; all three clear queues/replan from raw common input AFTER actual camera gate; subsequent inputs branch-current.',
            relative_gate='Per anchor/noise matched frozen all-robot active peak: candidate <=1.25*frozen+25N. Raw success and qualified success separate.',
            acceptance='>=4 valid scenes; >=8 complete paired3model conditions N; mix qualified successes exceeds BOTH controls by ceil(.25*N), gains in>=2 scenes, no new hardstops, no lost frozen fulltask-success scene.'),
        stops=dict(worker_seconds=600,total_hours=8,hard_normal_N=1500,disk_floor_GiB=2,
            collection='Success, hard guard,600 actions; no reseed/retry. Stop early only on source/scene gate;24 exhausted below gate => data insufficient, no training.',
            technical='Keep invalid attempt; no noise replacement. Integrity failure stops study. Budget interruption => incomplete.'),
        historical_limitations='Recontact evidence is not necessarily long fault or fresh visual replan; no learned/generalization claim before matched evidence.')
    dump(OUT/'registered-plan.json', plan)
    snapshot=OUT/'executed-source';snapshot.mkdir()
    for i,p in enumerate(sources): (snapshot/f'{i:02d}-{Path(p).name}.txt').write_bytes(Path(p).read_bytes())
    dump(OUT/'ordinary-control-match.json',dict(passed=True,reused=True,manifest=cm,updates=300,
        exact_historical_anchors_and_lr=True,split=plan['ordinary_control'],weights_sha256=plan['model_hashes']['ordinary_300']))
    print('Registered full study before new rollout',flush=True)


def worker(job):
    start=time.monotonic(); plan=read(OUT/'registered-plan.json'); folder=Path(job['folder'])
    folder.mkdir(parents=True,exist_ok=False)
    result=dict(job=job,execution_validity=False,restoration_validity=False,status='initializing',
        actions=0,prefix_actions=0,success=False,hard_stop=False,stop_reason=None,
        inference_seconds=0.,execution_seconds=0.,replay_seconds=0.,first_success_action=None,
        first_contact_seconds={},peaks=dict(all=0.,fridge=0.,other_fridge=0.,self=0.,target_hand_door=0.),inferences=[])
    env=video=None
    try:
        assert time.time()<plan['deadline_unix'] and Path(sys.prefix).name=='embodiedops-robocasa'
        assert all(sha(Path(p))==h for p,h in plan['source_hashes'].items())
        model=Path(job['checkpoint'])
        assert sha(model/'model.safetensors')==job['checkpoint_sha256']
        assert all(sha(model/n)==h for n,h in plan['processors'].items())
        import phase1d_rollout as rollout
        from phase1b_contract import physical, serial, sheet
        from phase1c_replay import geometry_spec
        from control_diagnostics import ContactRecorder
        from instruction_disk_guard import sample
        from lerobot.envs.robocasa import RoboCasaEnv,convert_action
        import renderer_lifecycle
        import av
        rollout.OUT=folder;os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba-cache');renderer_lifecycle.install()
        class SingleEpisode(RoboCasaEnv):
            def reset(self,*a,**kw):
                assert not getattr(self,'reset_once',False),'Second reset prohibited'
                self.reset_once=True;return super().reset(*a,**kw)
            def step(self,action):
                raw,reward,done,truncated,info=self._env.step(convert_action(action))
                return self._format_raw_obs(raw),reward,done or bool(info.get('success',False)),truncated,info
        prefix=job.get('prefix',0); cap=150 if job['kind']=='recovery' else 600
        horizon=prefix+cap
        env=SingleEpisode('CloseFridge',camera_name=CAMERAS,episode_length=horizon)
        seed=job['seed'];np.random.seed(seed);env._ensure_env();low=env._env.env
        low.horizon=horizon;low.ignore_done=False
        np.random.seed(seed);obs,_=env.reset(seed=seed)
        assert low.timestep==0 and low.horizon==horizon and not low.ignore_done
        recorder=ContactRecorder(low.sim,folder)
        spec=geometry_spec(low.sim.model._model,low.robots[0],low)
        dump(folder/'geometry-spec.json',spec)
        hand=set(spec['gripper_geoms']);doors={d:set(v['geoms']) for d,v in spec['doors'].items()}
        fridge={g['id'] for g in recorder.catalog if g['name'] in set(low.fxtr.contact_geoms)}
        assert all(hand.isdisjoint(ds) for ds in doors.values())
        scene=serial(dict(ep_meta=low.get_ep_meta(),controller=low.robots[0].composite_controller_config))
        dump(folder/'scene.json',scene)
        dump(folder/'reset.json',serial(dict(physical=physical(low),state=obs['agent_pos'],fridge=rollout.fridge(low))))
        obs_dir=folder/'observations';obs_dir.mkdir()
        saved={}
        def save_obs(pre):
            if pre not in saved:
                p=obs_dir/f'{pre:04d}.npz'
                np.savez_compressed(p,state=obs['agent_pos'],**obs['pixels']);saved[pre]=str(p)
            return saved[pre]
        save_obs(1)
        def camera_gate(ref):
            with np.load(ref) as f:
                errors={c:camera_error(obs['pixels'][c],f[c]) for c in CAMERAS}
                assert max_error(obs['agent_pos'],f['state'])<=1e-7
            assert all(e['max']<=1 and e['mean']<=.001 for e in errors.values()),str(errors)
            return errors
        def step(action,index):
            assert time.monotonic()-start<590 and time.time()<plan['deadline_unix'],'Worker/global budget'
            assert sample(ROOT)['above_floor'] and sample(OUT)['above_floor'],'Disk reserve'
            gate=rollout.action_gate(env,action)
            tick=time.monotonic()
            with recorder.step(index):ob,tr=rollout.traced_step(env,action,index)
            elapsed=time.monotonic()-tick
            for c in CAMERAS:camera_error(ob['pixels'][c],ob['pixels'][c])
            raw_peak=max((c['force_N_contact_frame'][0] for s in recorder.last_samples for c in s['robot_contacts']),default=0.)
            if raw_peak>1500:tr['guard_failures'].append('Hard force ceiling1500N')
            con=contacts(recorder.last_samples,recorder.robot_ids,fridge,doors,hand)
            return ob,tr,gate,con,elapsed
        if job.get('reference'):
            source=Path(job['reference']);old_reset=read(source/'reset.json');old_scene=read(source/'scene.json')
            assert rollout.canonical_scene(scene)==rollout.canonical_scene(old_scene)
            assert all(max_error(physical(low)[k],old_reset['physical'][k])<=1e-8 for k in ['qpos','qvel'])
            result['reset_camera_errors']=camera_gate(source/'observations/0001.npz')
            assert read(source/'geometry-spec.json')==spec
        if prefix:
            source=Path(job['reference']);history=rows(source/'steps.jsonl')[:prefix];assert len(history)==prefix
            maxima=dict(qpos=0.,qvel=0.,state=0.)
            tick=time.monotonic()
            with (folder/'replay.jsonl').open('x') as log:
                for row in history:
                    i=row['step'];obs,tr,gate,con,elapsed=step(np.array(row['action'],dtype=np.float32),i)
                    err={k:max_error(tr['after'][k],row['physical'][k]) for k in ['qpos','qvel']}
                    err['state']=max_error(obs['agent_pos'],row['state'])
                    for k,v in err.items():maxima[k]=max(maxima[k],v)
                    log.write(json.dumps(dict(step=i,errors=err,guards=tr['guard_failures']))+'\n');log.flush()
                    assert err['qpos']<=1e-8 and err['qvel']<=1e-8 and err['state']<=1e-7,'Replay divergence'
                    assert not tr['guard_failures'] and not tr['terminated'] and not tr['truncated'],'Prefix guard/termination'
                    result['prefix_actions']=i
            result['replay_seconds']=time.monotonic()-tick;result['max_replay_errors']=maxima
            result['boundary_camera_errors']=camera_gate(source/'observations'/f'{prefix+1:04d}.npz')
            save_obs(prefix+1)
        result['restoration_validity']=True
        result['initial_deg']=degrees(rollout.fridge(low));result['final_deg']=result['initial_deg'].copy();result['min_deg']=result['initial_deg'].copy()
        origin=float(low.sim.data.time);result['origin_time']=origin
        assert not rollout.fridge(low)['success']
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        policy=SmolVLAPolicy.from_pretrained(str(model));policy.eval();policy.reset()
        assert policy.config.n_obs_steps==1 and policy.config.rtc_config is None
        assert policy.config.chunk_size==policy.config.n_action_steps==50 and policy.config.num_steps==10
        pre,post=make_pre_post_processors(policy.config,pretrained_path=str(model))
        dump(folder/'manifest.json',dict(job=job,registered_plan_sha256=sha(OUT/'registered-plan.json'),
            checkpoint_sha256=job['checkpoint_sha256'],processors=plan['processors'],
            horizon=horizon,ignore_done=False,boundary=result,action_semantics='Native postprocessor output retained before irreversible controller threshold/reorder/clipping; training normalizes this output exactly once.',
            raw_observation='Lossless npz; pre-action t after t-1. Extra capture only, no extra inference.'))
        video=av.open(str(folder/'rollout.mp4'),'w');stream=video.add_stream('libx264',rate=20)
        stream.width=768;stream.height=288;stream.pix_fmt='yuv420p'
        seen={d:False for d in doors};gap={d:0 for d in doors};anchors=[]
        result.update(status='running',stop_reason='action_budget')
        with (folder/'steps.jsonl').open('x') as log:
            for local in range(1,cap+1):
                index=prefix+local
                if (local-1)%50==0:
                    save_obs(index)
                    batch=pre(dict(preprocess_observation(obs),task=['CloseFridge']))
                    ns=noise_seed(job['kind'],seed,job.get('replicate',1),local)
                    torch.manual_seed(ns);noise=policy.model.sample_noise((1,50,32),next(policy.parameters()).device)
                    torch.save(noise.cpu(),folder/f'noise-{local:04d}.pt')
                    torch.cuda.synchronize();tick=time.monotonic()
                    with torch.inference_mode():chunk=post(policy.predict_action_chunk(batch,noise=noise)).detach().cpu().numpy()[0]
                    torch.cuda.synchronize();elapsed=time.monotonic()-tick
                    assert chunk.shape==(50,12) and np.isfinite(chunk).all() and len(policy._queues['action'])==0
                    np.save(folder/f'actions-{index:04d}.npy',chunk)
                    info=dict(pre_action=index,local_pre_action=local,seed=ns,noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),seconds=elapsed,observation_sha256=sha(Path(saved[index])))
                    result['inferences'].append(info);result['inference_seconds']+=elapsed
                ds=eligible(seen,gap,False)
                if index in saved and ds:anchors.append(dict(pre_action=index,doors=ds,prior_gap={d:gap[d] for d in ds},observation=saved[index]))
                action=chunk[(local-1)%50];obs,tr,gate,con,elapsed=step(action,index)
                result['execution_seconds']+=elapsed;result['actions']=local
                result['final_deg']=degrees(tr['fridge_after'])
                for d,a in result['final_deg'].items():result['min_deg'][d]=min(result['min_deg'][d],a)
                for k,v in con['peaks'].items():result['peaks'][k]=max(result['peaks'][k],v)
                for d,t in con['first'].items():
                    if t is not None and d not in result['first_contact_seconds']:result['first_contact_seconds'][d]=t-origin
                result['success']=bool(tr['fridge_after']['success'])
                if result['success']:result['first_success_action']=local;result['first_success_seconds']=float(low.sim.data.time)-origin
                row=serial(dict(step=index,local_step=local,action=action,controller_input=tr['controller_input'],
                    state=obs['agent_pos'],physical=tr['after'],fridge=tr['fridge_after'],contact=con,
                    guard_failures=tr['guard_failures'],terminated=tr['terminated'],truncated=tr['truncated'],reward=tr['reward'],gate=gate))
                log.write(json.dumps(row)+'\n');log.flush()
                advance(seen,gap,con['loaded'])
                if not result['success'] and any(seen[d] and gap[d]==5 for d in doors):save_obs(index+1)
                for packet in stream.encode(av.VideoFrame.from_image(sheet(obs,f'{job["id"]} ACTION {index}'))):video.mux(packet)
                dump(folder/'anchors.json',anchors);dump(folder/'result.json',result)
                if local%50==0 or result['success'] or tr['guard_failures']:print(job['id'],local,result['final_deg'],'success',result['success'],flush=True)
                if tr['guard_failures']:
                    result.update(hard_stop=True,status='hard_stop',stop_reason=str(tr['guard_failures']));break
                if result['success']:result['stop_reason']='true_all_door_success';break
                if tr['terminated'] or tr['truncated']:
                    assert local==cap and low.timestep==horizon,'Premature termination'
                    result['stop_reason']='horizon';break
            if result['status']=='running':result['status']='completed'
            result['execution_validity']=True
        save_obs(prefix+result['actions']+1)
        result['first_causal_anchor']=anchors[0] if anchors else None
    except Exception as exc:
        result.update(status='technical_invalid',execution_validity=False,stop_reason=str(exc),exception=traceback.format_exc())
    finally:
        if video is not None:
            for packet in stream.encode():video.mux(packet)
            video.close()
        if env is not None:env.close()
        result['worker_seconds']=time.monotonic()-start;dump(folder/'result.json',result)
    print(json.dumps(result),flush=True)


def run_job(job):
    plan=read(OUT/'registered-plan.json')
    assert time.time()+600<plan['deadline_unix'],'Insufficient remaining worker budget'
    folder=Path(job['folder']);folder.parent.mkdir(parents=True,exist_ok=True)
    jobfile=folder.parent/(job['id']+'-job.json');dump(jobfile,job)
    tick=time.monotonic()
    with (folder.parent/(job['id']+'-worker.log')).open('x') as log:
        try:
            p=subprocess.run([sys.executable,'-B','-u',str(Path(__file__)),'worker','--job',str(jobfile)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            assert p.returncode==0,'Worker process crashed'
        except Exception as exc:
            folder.mkdir(exist_ok=True)
            result=read(folder/'result.json') if (folder/'result.json').exists() else dict(actions=0,success=False)
            result.update(job=job,status='technical_invalid',execution_validity=False,stop_reason=str(exc),worker_seconds=time.monotonic()-tick)
            dump(folder/'result.json',result)
    r=read(folder/'result.json')
    with (OUT/'execution-list.jsonl').open('a') as f:f.write(json.dumps(dict(job=job,result=r))+'\n')
    return r


def job_for(kind, seed, rep, model='frozen', reference=None, anchor=None):
    plan=read(OUT/'registered-plan.json');models=read(OUT/'frozen-models.json') if (OUT/'frozen-models.json').exists() else dict(paths=plan['models'],hashes=plan['model_hashes'])
    name=f'{kind}-{seed}-r{rep}-{model}'
    return dict(id=name,kind=kind,seed=seed,replicate=rep,model=model,folder=str(OUT/kind/name),
        checkpoint=models['paths'][model],checkpoint_sha256=models['hashes'][model],
        reference=str(reference) if reference else None,prefix=anchor['pre_action']-1 if anchor else 0,
        target_door=anchor['doors'][0] if anchor else None)


def collect():
    from recontact_data import build_index
    plan=read(OUT/'registered-plan.json')
    with (OUT/'collection-started.json').open('x') as f:json.dump(dict(time=time.time()),f)
    index=build_index()
    for entry in plan['collection_order']:
        if index['gate_passed']:break
        if time.time()+600>plan['collection_deadline_unix']:
            dump(OUT/'collection-status.json',dict(status='budget_incomplete'));return
        r=run_job(job_for('collection',entry['seed'],entry['replicate']))
        index=build_index()
        print('COLLECTION',entry,'sources',index['qualified_source_count'],'scenes',index['qualified_seeds'],'valid',r['execution_validity'],flush=True)
        if not r['execution_validity']:
            dump(OUT/'collection-status.json',dict(status='technical_incomplete',job=r['job'],reason=r['stop_reason']));return
    dump(OUT/'collection-status.json',dict(status='gate_passed' if index['gate_passed'] else 'data_insufficient',index_sha256=sha(OUT/'data-index.json')))


def evaluate():
    plan=read(OUT/'registered-plan.json')
    assert read(OUT/'collection-status.json')['status']=='gate_passed'
    assert read(OUT/'training/reload-verification.json')['passed']
    assert (OUT/'frozen-models.json').exists()
    with (OUT/'evaluation-started.json').open('x') as f:json.dump(dict(time=time.time()),f)
    sources=[]
    for seed in plan['test_seeds']:
        r=run_job(job_for('full',seed,1));sources.append(r)
    for r in sources:
        if not r['execution_validity'] or not r.get('first_causal_anchor'):continue
        for rep in [1,2]:
            for model in ['frozen','ordinary_300','recontact_mix_300']:
                run_job(job_for('recovery',r['job']['seed'],rep,model,r['job']['folder'],r['first_causal_anchor']))
    for r in sources:
        run_job(job_for('full',r['job']['seed'],1,'recontact_mix_300',r['job']['folder']))
    dump(OUT/'evaluation-finished.json',dict(time=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','collect','worker','evaluate']);ap.add_argument('--job',type=Path);args=ap.parse_args()
    if args.mode=='worker':worker(read(args.job))
    else:globals()[args.mode]()
