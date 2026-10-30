"""Single bounded continuation after the physically replayed v1b A action 800."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from instruction_control_audit import ROOT, BASE, read, dump, sha
from recovery_feasibility import max_error, camera_error
from recovery_v1b import peak_contacts

OUT=Path('D:/EmbodiedOpsOutputs/recovery-completion-a48-v1')
V1B=ROOT/'outputs/agent-workflow-v1/recovery-feasibility-v1b'
A=V1B/'A'


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def replay_records(plan):
    historical=records(ROOT/plan['state']['source']/'steps.jsonl')[:500]
    continuation=records(A/'steps.jsonl')
    assert len(historical)==500 and len(continuation)==300
    rows=[]
    for i,row in enumerate(historical+continuation,1):
        assert row.get('step',row.get('historical_index'))==i
        rows.append(dict(index=i,action=row['action'],physical=row['physical'],
            state=row['returned_state'] if i<=500 else row['state']))
    # Saved postprocessor output equals the actual public action logged by v1b.
    for i in range(0,300,50):
        chunk=np.load(A/f'actions-{i:03d}.npy')
        assert np.array_equal(chunk,np.asarray([r['action'] for r in continuation[i:i+50]],dtype=np.float32))
    with np.load(A/'observations/300.npz') as boundary:
        assert max_error(boundary['state'],rows[-1]['state'])==0
    assert abs(rows[-1]['physical']['time']-40.5)<1e-8
    return rows


def register():
    OUT.mkdir(exist_ok=False)
    old=read(V1B/'registered-plan.json')
    assert all(sha(Path(p))==h for p,h in old['source_hashes'].items())
    protected=read(V1B/'preserved-before.json')
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    for p in V1B.rglob('*'):
        if p.is_file(): protected[str(p)]=sha(p)
    protected[str(BASE/'config.json')]=sha(BASE/'config.json')
    dump(OUT/'preserved-before.json',protected)
    cfg=read(BASE/'config.json')
    assert cfg['n_obs_steps']==1 and cfg['rtc_config'] is None
    plan=dict(old,state=old['state'],created_unix=time.time(),noise_seeds=[931007,931008,931009,931010],
        recovery_actions=200,experiment_episode_cap=1000,total_seconds=1800,worker_seconds=600,
        task='A48 frozen-policy completion',original_failure_recovery_cap=500,
        input_paths=dict(historical=str(ROOT/old['state']['source']/'steps.jsonl'),
            v1b_actions=str(A/'steps.jsonl'),boundary_observation=str(A/'observations/300.npz')),
        boundary_semantics='v1b steps index=300/historical_index=800 and observations/300.npz are AFTER action 800; first new action=801.',
        action_semantics='v1b steps.action is postprocessor public 12D action passed to action_gate/traced_step; convert_action and key_converter perform original controller mapping. No raw model outputs replayed.',
        policy_context=dict(n_obs_steps=1,rtc_config=None,call='predict_action_chunk',
            queues='reset creates only action deque; predict_action_chunk excludes action and never fills it',
            cache='sample_actions builds prefix KV with past_key_values=None every call; local to this chunk',
            previous_action='No previous action or leftover kwargs supplied; RTC disabled',
            conclusion='Reload/reset preserves relevant context for this stateless chunk path; explicit noise defines one preregistered stochastic future.'),
        horizon_rule='After _ensure_env, before sole reset and before any public action, set low.horizon=1000 and ignore_done=False. Never reset at boundary; honor all done/truncated.',
        observation_rule='Every new chunk uses current actual RGB/16D; no common reference substitution.',
        relative_contact_reference_N=168.904772,
        source_hashes={**old['source_hashes'],str(Path(__file__).resolve()):sha(Path(__file__))},
        old_junctions={str(p):str(p.resolve()) for p in [V1B,V1B.parent/'recovery-feasibility-v1']})
    # Remove inherited v1b-only operational declarations rather than misregister them.
    for key in ['branch_order','acceptance','expert','common_invalidity','output_junction','original_junction']:
        plan.pop(key,None)
    plan['contact_rule']='Same per-physics-substep active robot normal-force peak; fridge/self/gripper-door separate. Hard ceiling=1500 N. Fixed v1b reference=168.904772 N is descriptive, not a new gate; unequal observation budgets.'
    plan['stops']=['true all-door success','200 new actions','any premature done/truncated','inherited guards and exceptions','1500 N hard force','disk floor 2 GiB','600s worker timeout']
    replay_records(plan)
    plan['input_hashes']={p:sha(Path(p)) for p in plan['input_paths'].values()}
    dump(OUT/'registered-plan.json',plan)
    (OUT/'executed-source.py.txt').write_bytes(Path(__file__).read_bytes())


def worker():
    start=time.monotonic(); plan=read(OUT/'registered-plan.json')
    result=dict(restoration_validity=False,execution_validity=False,execution_status='restoring',
        stop_reason=None,new_actions=0,global_end_action_index=800,replayed_actions=0,
        true_all_door_success=None,first_success_global_index=None,first_success_new_action=None,
        hard_guard_status='not_reached',inference_wall_seconds=0.,execution_wall_seconds=0.,
        replay_wall_seconds=0.,peak_normal_N=0.,peak_fridge_N=0.,peak_self_N=0.,peak_door_N=0.,
        first_new_door_contact_seconds=None,max_replay_errors=dict(qpos=0.,qvel=0.,state=0.),inferences=[])
    env=video=None
    try:
        assert Path(sys.prefix).name=='embodiedops-robocasa'
        assert all(sha(Path(p))==h for p,h in plan['source_hashes'].items())
        assert all(sha(Path(p))==h for p,h in plan['input_hashes'].items())
        assert sha(BASE/'model.safetensors')==plan['checkpoint_sha256']
        assert all(sha(BASE/n)==h for n,h in plan['processors'].items())
        import phase1d_rollout as rollout
        from phase1b_contract import CAMERAS, physical, serial, sheet
        from control_diagnostics import ContactRecorder
        from instruction_disk_guard import sample
        from lerobot.envs.robocasa import RoboCasaEnv,convert_action
        from PIL import Image
        import renderer_lifecycle
        import av
        rollout.OUT=OUT; os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba-cache')
        renderer_lifecycle.install()
        source=ROOT/plan['state']['source']
        import importlib.metadata as md
        versions=read(source/'manifest.json')['versions']
        assert {k:md.version(k) for k in versions}==versions
        class SingleEpisode(RoboCasaEnv):
            def reset(self,*args,**kwargs):
                assert not getattr(self,'reset_once',False),'Second reset prohibited'
                self.reset_once=True
                return super().reset(*args,**kwargs)
            def step(self,action):
                raw,reward,done,truncated,info=self._env.step(convert_action(action))
                return self._format_raw_obs(raw),reward,done or bool(info.get('success',False)),truncated,info
        env=SingleEpisode('CloseFridge',camera_name=CAMERAS,episode_length=1000)
        np.random.seed(48);env._ensure_env()
        low=env._env.env
        low.horizon=1000;low.ignore_done=False
        np.random.seed(48);obs,_=env.reset(seed=48)
        assert low.timestep==0 and low.horizon==1000 and not low.ignore_done
        result['horizon_configuration']=dict(horizon=low.horizon,ignore_done=low.ignore_done,timestep_before_public_actions=low.timestep)
        reset=physical(low);old=read(source/'reset.json')
        assert all(max_error(reset[k],old['physical'][k])==0 for k in ('qpos','qvel'))
        assert max_error(obs['agent_pos'],old['state'])==0
        assert rollout.canonical_scene(serial(low.get_ep_meta()))==rollout.canonical_scene(read(source/'scene.json')['ep_meta'])
        assert serial(low.robots[0].composite_controller_config)==read(source/'scene.json')['controller']
        def camera_gate(reference):
            errors={c:camera_error(obs['pixels'][c],reference[c]) for c in CAMERAS}
            assert all(e['max']<=1 and e['mean']<=.001 for e in errors.values()),'Camera gate failed: '+str(errors)
            return errors
        result['reset_cameras']=camera_gate({c:np.asarray(Image.open(source/f'reset-{c}.png')) for c in CAMERAS})
        recorder=ContactRecorder(low.sim,OUT)
        def step(action,index):
            assert time.monotonic()-start<600,'Worker deadline'
            for p in [ROOT,OUT]:assert sample(p)['above_floor'],'Disk reserve'
            rollout.action_gate(env,action)
            with recorder.step(index):observation,trace=rollout.traced_step(env,action,index)
            for c in CAMERAS:camera_error(observation['pixels'][c],observation['pixels'][c])
            peak=max((c['force_N_contact_frame'][0] for s in recorder.last_samples for c in s['robot_contacts']),default=0.)
            if peak>1500:trace['guard_failures'].append('Hard force ceiling')
            return observation,trace
        t=time.monotonic()
        with (OUT/'replay.jsonl').open('x') as log:
            for row in replay_records(plan):
                i=row['index'];obs,trace=step(np.asarray(row['action'],dtype=np.float32),i)
                result['replayed_actions']=i
                errors={k:max_error(trace['after'][k],row['physical'][k]) for k in ('qpos','qvel')}
                errors['state']=max_error(obs['agent_pos'],row['state'])
                for k,v in errors.items():result['max_replay_errors'][k]=max(v,result['max_replay_errors'][k])
                log.write(json.dumps(dict(index=i,errors=errors,guards=trace['guard_failures'],terminated=trace['terminated'],truncated=trace['truncated']))+'\n');log.flush()
                assert all(v<=plan['tolerances'][k] for k,v in errors.items()),'Physical/state replay divergence'
                assert not trace['guard_failures'],str(trace['guard_failures'])
                assert not trace['terminated'] and not trace['truncated'],'Unexpected prefix termination'
                if i==500:
                    with np.load(A/'observations/000.npz') as ref:result['action500_cameras']=camera_gate(ref)
                if i%100==0:print('Replay',i,flush=True)
        result['replay_wall_seconds']=time.monotonic()-t
        with np.load(A/'observations/300.npz') as ref:
            result['boundary_cameras']=camera_gate(ref)
            assert max_error(obs['agent_pos'],ref['state'])<=1e-7
        result['restoration_validity']=True
        result['boundary_time']=float(low.sim.data.time)
        result['initial_deg']=np.rad2deg(list(rollout.fridge(low)['qpos'].values())).tolist()
        result['final_deg']=result['initial_deg'].copy();result['min_deg']=result['initial_deg'].copy()
        assert low.timestep==800 and not rollout.fridge(low)['success']
        observation_dir=OUT/'observations';observation_dir.mkdir()
        def save_obs(i):np.savez_compressed(observation_dir/f'{i:03d}.npz',state=obs['agent_pos'],**obs['pixels'])
        save_obs(0);sheet(obs,'GLOBAL 800').save(OUT/'before.png')
        # Frozen registration plus completed restoration audit, before any new inference/action.
        dump(OUT/'manifest.json',dict(plan=plan,boundary_verification=result))
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        policy=SmolVLAPolicy.from_pretrained(str(BASE));policy.eval();policy.reset()
        assert policy.config.n_obs_steps==1 and policy.config.rtc_config is None and policy.rtc_processor is None
        assert policy.config.chunk_size==policy.config.n_action_steps==50 and policy.config.num_steps==10
        assert set(policy._queues)=={'action'} and len(policy._queues['action'])==0
        pre,post=make_pre_post_processors(policy.config,pretrained_path=str(BASE))
        geometry=read(A/'geometry-spec.json')
        door_ids=set(next(iter(geometry['doors'].values()))['geoms']);hand_ids=set(geometry['gripper_geoms'])
        fridge_ids={g['id'] for g in recorder.catalog if g['name'] in set(low.fxtr.contact_geoms)}
        assert recorder.catalog==read(A/'geometry-catalog.json')
        video=av.open(str(OUT/'A_continuation.mp4'),mode='w')
        stream=video.add_stream('libx264',rate=20);stream.width=768;stream.height=288;stream.pix_fmt='yuv420p'
        result.update(execution_status='running',hard_guard_status='clear',true_all_door_success=False,stop_reason='new_action_budget_200')
        execution_start=time.monotonic()
        with (OUT/'steps.jsonl').open('x') as log:
            for i in range(200):
                if i%50==0:
                    save_obs(i)
                    batch=pre(dict(preprocess_observation(obs),task=['CloseFridge']))
                    assert batch['observation.state'].shape==(1,16)
                    assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v,torch.Tensor))
                    seed=plan['noise_seeds'][i//50];torch.manual_seed(seed)
                    noise=policy.model.sample_noise((1,50,32),next(policy.parameters()).device)
                    torch.save(noise.cpu(),OUT/f'noise-{i:03d}.pt')
                    noise_hash=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
                    torch.cuda.synchronize();t=time.monotonic()
                    with torch.inference_mode():chunk=post(policy.predict_action_chunk(batch,noise=noise)).detach().cpu().numpy()[0]
                    torch.cuda.synchronize();elapsed=time.monotonic()-t
                    result['inference_wall_seconds']+=elapsed
                    result['inferences'].append(dict(after_new_action=i,seed=seed,noise_sha256=noise_hash,seconds=elapsed,observation_sha256=sha(observation_dir/f'{i:03d}.npz')))
                    assert chunk.shape==(50,12) and np.isfinite(chunk).all()
                    assert set(policy._queues)=={'action'} and len(policy._queues['action'])==0
                    np.save(OUT/f'actions-{i:03d}.npy',chunk)
                action=chunk[i%50];obs,trace=step(action,801+i)
                result.update(new_actions=i+1,global_end_action_index=801+i)
                peaks,first=peak_contacts(recorder.last_samples,recorder.robot_ids,fridge_ids,door_ids,hand_ids)
                for name,key in [('all','normal'),('fridge','fridge'),('self','self'),('door','door')]:result['peak_'+key+'_N']=max(result['peak_'+key+'_N'],peaks[name])
                if first is not None and result['first_new_door_contact_seconds'] is None:result['first_new_door_contact_seconds']=first-result['boundary_time']
                angles=np.rad2deg(list(trace['fridge_after']['qpos'].values())).tolist()
                result['final_deg']=angles;result['min_deg']=np.minimum(result['min_deg'],angles).tolist()
                result['new_simulation_seconds']=float(low.sim.data.time)-result['boundary_time']
                success=bool(trace['fridge_after']['success'])
                if success:result.update(true_all_door_success=True,first_success_global_index=801+i,first_success_new_action=i+1,first_success_new_simulation_seconds=result['new_simulation_seconds'],first_success_since_failure_seconds=15+result['new_simulation_seconds'])
                save_obs(i+1)
                log.write(json.dumps(serial(dict(index=i+1,global_index=801+i,action=action,physical=trace['after'],state=obs['agent_pos'],fridge=trace['fridge_after'],guards=trace['guard_failures'],terminated=trace['terminated'],truncated=trace['truncated'],contact_peaks=peaks)))+'\n');log.flush()
                for packet in stream.encode(av.VideoFrame.from_image(sheet(obs,f'NEW BUDGET G{801+i}'))):video.mux(packet)
                result['execution_wall_seconds']=time.monotonic()-execution_start-result['inference_wall_seconds']
                dump(OUT/'result.json',result)
                if (i+1)%50==0 or success or trace['guard_failures']:print('New action',i+1,'angle',angles,'success',success,'peaks',peaks,flush=True)
                if trace['guard_failures']:result.update(hard_guard_status='triggered',stop_reason=str(trace['guard_failures']),execution_status='hard_stop');break
                if success:result['stop_reason']='true_all_door_success';break
                if trace['terminated'] or trace['truncated']:
                    result['stop_reason']='global_horizon_1000' if i+1==200 else 'premature_termination';break
            if result['execution_status']=='running':result['execution_status']='completed'
            result['execution_validity']=result['execution_status']=='completed' and result['stop_reason']!='premature_termination'
    except Exception as exc:
        import traceback
        result.update(execution_status='technical_invalid',stop_reason=str(exc),exception=traceback.format_exc())
    finally:
        if video is not None:
            for packet in stream.encode():video.mux(packet)
            video.close()
        if env is not None:
            if result['restoration_validity']:sheet(obs,'FINAL').save(OUT/'after.png')
            env.close()
        result['worker_wall_seconds']=time.monotonic()-start
        dump(OUT/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


def run():
    register()
    with (OUT/'attempt-started.json').open('x') as f:json.dump(dict(time=time.time()),f)
    try:
        with (OUT/'worker.log').open('x') as log:
            process=subprocess.run([sys.executable,'-B','-u',str(Path(__file__)),'worker'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        assert process.returncode==0,'Worker crashed'
    except Exception as exc:
        r=read(OUT/'result.json') if (OUT/'result.json').exists() else dict(restoration_validity=False,new_actions=0,true_all_door_success=None)
        r.update(execution_validity=False,execution_status='technical_invalid',stop_reason=str(exc));dump(OUT/'result.json',r)
    finally:
        protected=read(OUT/'preserved-before.json');plan=read(OUT/'registered-plan.json')
        dump(OUT/'preservation.json',dict(checked=len(protected),changed=[p for p,h in protected.items() if sha(ROOT/p)!=h],
            sources_unchanged=all(sha(Path(p))==h for p,h in plan['source_hashes'].items()),
            junctions_unchanged=all(str(Path(p).resolve())==target for p,target in plan['old_junctions'].items())))
    print(json.dumps(read(OUT/'result.json'),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['run','worker']);args=ap.parse_args()
    run() if args.mode=='run' else worker()
