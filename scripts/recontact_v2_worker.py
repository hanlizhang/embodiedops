"""Execute one immutable v2 evaluation condition using inherited rollout primitives."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from instruction_control_audit import ROOT, read, sha
from recovery_feasibility import max_error, camera_error
from recontact_self_imitation import CAMERAS, contacts, degrees, rows, eligible, advance
from recontact_v2 import assert_sources
OUT = Path('D:/EmbodiedOpsOutputs/recontact-self-imitation-v2')


def dump(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt==19:raise
            time.sleep(.025)  # Windows readers may temporarily omit FILE_SHARE_DELETE.


def noise_seed(kind, seed, replicate, local_pre):
    assert kind in ('full', 'recovery')
    return (42000000 + seed*1000 + local_pre if kind == 'full'
            else 43000000 + seed*10000 + replicate*1000 + local_pre)


_UNSUPPORTED = object()


def plain_state(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.dtype.kind in 'biuf' else _UNSUPPORTED
    if isinstance(value, (list, tuple)):
        items = [plain_state(v) for v in value]
        return items if all(v is not _UNSUPPORTED for v in items) else _UNSUPPORTED
    if isinstance(value, dict):
        items = {str(k): plain_state(v) for k,v in value.items()}
        return items if all(v is not _UNSUPPORTED for v in items.values()) else _UNSUPPORTED
    return _UNSUPPORTED


def object_state(obj):
    # Stored attributes only: no property or controller-update invocation.
    values, omitted = {}, {}
    for key, value in vars(obj).items():
        plain = plain_state(value)
        if plain is not _UNSUPPORTED:
            values[key] = plain
        elif ('interpolator' in key or 'buf' in key) and hasattr(value, '__dict__'):
            values[key] = object_state(value)
        else:
            omitted[key] = type(value).__name__
    return dict(class_name=type(obj).__name__, values=values, omitted_nonstate_objects=omitted)


def controller_boundary(low):
    robot=low.robots[0]; cc=robot.composite_controller
    return dict(composite=object_state(cc),
        parts={k:object_state(v) for k,v in cc.part_controllers.items()},
        gripper_accumulator={k:g.current_action.tolist() for k,g in robot.gripper.items()},
        ctrl=low.sim.data.ctrl.tolist(),timestep=int(low.timestep),physics_time=float(low.sim.data.time))


def compare_state(actual, expected, tolerance=1e-8, path='controller'):
    if isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys()==expected.keys(), path+': keys differ'
        return max((compare_state(v,expected[k],tolerance,path+'.'+k) for k,v in actual.items()),default=0.)
    if isinstance(actual, list):
        assert isinstance(expected,list) and len(actual)==len(expected), path+': shape differs'
        return max((compare_state(a,b,tolerance,path+f'[{i}]') for i,(a,b) in enumerate(zip(actual,expected))),default=0.)
    if isinstance(actual,(int,float)) and not isinstance(actual,bool):
        assert isinstance(expected,(int,float)) and not isinstance(expected,bool), path+': type differs'
        if actual==expected: return 0.  # Includes equal static infinite limits; excludes NaN.
        error=abs(float(actual)-float(expected))
        assert np.isfinite(error) and error<=tolerance, path+f': error={error}'
        return error
    assert type(actual) is type(expected) and actual==expected,path+': value/type differs'
    return 0.


def load_observation(path):
    with np.load(path) as f:
        return dict(agent_pos=f['state'].copy(),pixels={c:f[c].copy() for c in CAMERAS})


def worker(job):
    start=time.monotonic(); plan=read(OUT/'registered-plan.json'); folder=Path(job['folder'])
    assert folder.resolve().is_relative_to(OUT.resolve()), 'Output outside v2 root'
    folder.mkdir(parents=True,exist_ok=False)
    result=dict(job=job,execution_validity=False,restoration_validity=False,status='initializing',
        actions=0,prefix_actions=0,success=False,hard_stop=False,stop_reason=None,
        inference_seconds=0.,execution_seconds=0.,replay_seconds=0.,first_success_action=None,
        first_contact_seconds={},first_recontact_seconds={},first_recontact_action={},recontact_events=[],
        raw_all_door_success=False,policy_success=False,first_success_seconds=None,first_success_global_action=None,
        per_door_peaks_N={},peaks=dict(all=0.,fridge=0.,other_fridge=0.,self=0.,target_hand_door=0.),inferences=[])
    env=video=None
    try:
        assert time.time()<plan['deadline_unix'] and Path(sys.prefix).name=='embodiedops-robocasa'
        assert_sources(plan)
        registered=next(c for c in plan['conditions'] if c['id']==job['id'])
        assert all(job[k]==v for k,v in registered.items()),'Condition differs from frozen manifest'
        if job['kind']=='recovery':
            assert sha(OUT/'holdout-data-index.json')==plan['holdout_index_sha256'],'Holdout index integrity'
            source=next(s for s in read(OUT/'holdout-data-index.json')['sources'] if s['id']==job['source_id'])
            provenance=read(OUT/'data-audit.json')['source_provenance']['sources'][job['source_id']]
            assert provenance['input_hashes']==source['input_hashes'],'Source provenance differs'
            assert all(sha(Path(p))==h for p,h in source['input_hashes'].items()),'Historical source integrity'
            anchor=next(s for s in plan['recovery_states'] if s['source_id']==job['source_id'])
            assert job['prefix']==anchor['prefix'] and job['target_door']==anchor['target_door']
            assert sha(Path(anchor['observation']))==anchor['observation_sha256'],'Historical boundary integrity'
            result['source_input_hashes_verified']=len(source['input_hashes'])
            result['registered_boundary_observation_sha256']=anchor['observation_sha256']
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
        prefix=int(job.get('prefix',0)); cap=150 if job['kind']=='recovery' else 600
        assert job['kind'] in ('full','recovery') and job.get('cap',cap)==cap
        assert (prefix>0 and job.get('reference') and job.get('target_door')) if job['kind']=='recovery' else prefix==0
        assert job['noise_seeds']==[noise_seed(job['kind'],job['seed'],job.get('replicate',1),i) for i in range(1,cap+1,50)]
        horizon=prefix+cap
        result.update(action_budget=cap,horizon=horizon,global_end_action=prefix)
        env=SingleEpisode('CloseFridge',camera_name=CAMERAS,episode_length=horizon)
        seed=job['seed'];np.random.seed(seed);env._ensure_env();low=env._env.env
        low.horizon=horizon;low.ignore_done=False
        np.random.seed(seed);obs,_=env.reset(seed=seed)
        assert low.timestep==0 and low.horizon==horizon and not low.ignore_done
        reset_time=float(low.sim.data.time)
        recorder=ContactRecorder(low.sim,folder)
        spec=geometry_spec(low.sim.model._model,low.robots[0],low)
        dump(folder/'geometry-spec.json',spec)
        hand=set(spec['gripper_geoms']);doors={d:set(v['geoms']) for d,v in spec['doors'].items()}
        fridge={g['id'] for g in recorder.catalog if g['name'] in set(low.fxtr.contact_geoms)}
        assert fridge and hand and all(hand.isdisjoint(ds) for ds in doors.values())
        target=job.get('target_door');assert target is None or target in doors
        metric_doors={target:doors[target]} if target is not None else doors
        dump(folder/'contact-geometry.json',dict(robot_geom_ids=sorted(recorder.robot_ids),hand_geom_ids=sorted(hand),
            fridge_geom_ids=sorted(fridge),fridge_geom_names=sorted(low.fxtr.contact_geoms),
            task_door_geom_ids={d:sorted(gs) for d,gs in doors.items()},selected_target_door=target,
            metric_target_geom_ids=sorted(set().union(*metric_doors.values())),
            effective_contact='Active solver constraint; >=1N normal force at any substep.',
            other_fridge='Robot-fridge except hand contact with selected target door(s).'))
        scene=serial(dict(ep_meta=low.get_ep_meta(),controller=low.robots[0].composite_controller_config))
        dump(folder/'scene.json',scene)
        reset_physical=physical(low)  # Exactly one original diagnostic read; forced observation reads advance sensor clocks.
        dump(folder/'reset.json',serial(dict(physical=reset_physical,state=obs['agent_pos'],fridge=rollout.fridge(low))))
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
            assert low.timestep==index-1,'Before-step counter mismatch'
            gate=rollout.action_gate(env,action)
            tick=time.monotonic()
            with recorder.step(index):ob,tr=rollout.traced_step(env,action,index)
            elapsed=time.monotonic()-tick
            assert low.timestep==index,'After-step counter mismatch'
            assert abs(float(low.sim.data.time)-(reset_time+index*.05))<=1e-8,'Physics-time mismatch'
            for c in CAMERAS:camera_error(ob['pixels'][c],ob['pixels'][c])
            raw_peak=max((c['force_N_contact_frame'][0] for s in recorder.last_samples for c in s['robot_contacts']),default=0.)
            if raw_peak>1500:tr['guard_failures'].append('Hard force ceiling1500N')
            con=contacts(recorder.last_samples,recorder.robot_ids,fridge,metric_doors,hand)
            all_doors=contacts(recorder.last_samples,recorder.robot_ids,fridge,doors,hand)
            con.update(task_door_loaded=all_doors['loaded'],task_door_first=all_doors['first'],task_door_peaks_N=all_doors['per_door_N'])
            return ob,tr,gate,con,elapsed
        if job.get('reference'):
            source=Path(job['reference']);old_reset=read(source/'reset.json');old_scene=read(source/'scene.json')
            assert rollout.canonical_scene(scene)==rollout.canonical_scene(old_scene)
            assert all(max_error(reset_physical[k],old_reset['physical'][k])<=1e-8 for k in ['qpos','qvel'])
            assert abs(reset_time-old_reset['physical']['time'])<=1e-8,'Reset physics time mismatch'
            result['reset_camera_errors']=camera_gate(source/'observations/0001.npz')
            assert read(source/'geometry-spec.json')==spec
            assert read(source/'geometry-catalog.json')==recorder.catalog,'Geometry catalog identity'
        seen={d:False for d in doors};gap={d:0 for d in doors}
        if prefix:
            source=Path(job['reference']);history=rows(source/'steps.jsonl')[:prefix];assert len(history)==prefix
            maxima=dict(qpos=0.,qvel=0.,state=0.,time=0.,controller_input=0.)
            tick=time.monotonic()
            with (folder/'replay.jsonl').open('x') as log:
                for i,row in enumerate(history,1):
                    assert row['step']==i,'Historical index mismatch'
                    assert not row['guard_failures'] and not row['terminated'] and not row['truncated'],'Historical prefix ended'
                    obs,tr,gate,con,elapsed=step(np.array(row['action'],dtype=np.float32),i)
                    err={k:max_error(tr['after'][k],row['physical'][k]) for k in ['qpos','qvel']}
                    err['state']=max_error(obs['agent_pos'],row['state'])
                    err['time']=abs(tr['after']['time']-row['physical']['time'])
                    err['controller_input']=max_error(tr['controller_input'],row['controller_input'])
                    assert np.array_equal(tr['controller_input'],np.asarray(row['controller_input'])),'Historical controller-input mismatch'
                    for k,v in err.items():maxima[k]=max(maxima[k],v)
                    log.write(json.dumps(dict(step=i,errors=err,guards=tr['guard_failures'],timestep=low.timestep,physics_time=tr['after']['time']))+'\n');log.flush()
                    assert err['qpos']<=1e-8 and err['qvel']<=1e-8 and err['state']<=1e-7 and err['time']<=1e-8,'Replay divergence'
                    assert not tr['guard_failures'] and not tr['terminated'] and not tr['truncated'],'Prefix guard/termination'
                    advance(seen,gap,con['task_door_loaded'])
                    result['prefix_actions']=i
            result['replay_seconds']=time.monotonic()-tick;result['max_replay_errors']=maxima
            assert seen[target] and gap[target]>=5 and not rollout.fridge(low)['success'],'Anchor causal contact gate'
            result['anchor_contact_history']=dict(seen=seen.copy(),preceding_gap_actions=gap.copy())
        assert low.timestep==prefix and abs(float(low.sim.data.time)-(reset_time+prefix*.05))<=1e-8
        common_path=Path(job['reference'])/'observations'/f'{prefix+1:04d}.npz' if job.get('reference') else None
        if common_path:result['boundary_camera_errors']=camera_gate(common_path)
        assert np.asarray(obs['agent_pos']).shape==(16,) and np.isfinite(obs['agent_pos']).all(),'Boundary 16D state integrity'
        for c in CAMERAS:camera_error(obs['pixels'][c],obs['pixels'][c])
        save_obs(prefix+1)
        np.savez_compressed(folder/'actual-boundary.npz',state=obs['agent_pos'],**obs['pixels'])
        first_input=load_observation(common_path) if common_path else deepcopy(obs)
        np.savez_compressed(folder/'common-policy-input.npz',state=first_input['agent_pos'],**first_input['pixels'])
        boundary=controller_boundary(low);dump(folder/'boundary-controller.json',boundary)
        sensor_clocks={name:{key:plain_state(vars(observable)[key]) for key in
            ['_enabled','_active','_sampling_timestep','_time_since_last_sample','_current_delay','_sampled']
            if key in vars(observable)} for name,observable in low._observables.items()}
        dump(folder/'boundary-observable-clocks.json',sensor_clocks)
        result['observation_schedule']='One reset physical() diagnostic, then original before/after traced_step calls. Reference comparisons reuse cached values; no extra force_update.'
        paired=job.get('paired_boundary_reference')
        if paired:
            paired_path=Path(paired)
            if paired_path.is_dir():paired_path=paired_path/'boundary-controller.json'
            result['paired_controller_max_error']=compare_state(boundary,read(paired_path))
            result['paired_controller_reference']=dict(path=str(paired_path),sha256=sha(paired_path))
            peer_clocks=paired_path.parent/'boundary-observable-clocks.json'
            if peer_clocks.exists():
                result['paired_observable_clock_max_error']=compare_state(sensor_clocks,read(peer_clocks))
            else:
                result['paired_observable_clock_basis']='Historical frozen baseline lacks clock snapshot; exact original diagnostic call schedule retained, with unchanged controller/physical/raw-camera gates.'
        result['controller_restore_basis']='Full historical public-action replay with exact converted inputs and per-step physical/state/time/counter checks; historical controller internal snapshot absent. Current boundary numeric state/targets/integrators/gripper saved and compared with first frozen restored boundary when supplied.'
        result['actual_boundary_sha256']=sha(folder/'actual-boundary.npz')
        result['common_policy_input_sha256']=sha(folder/'common-policy-input.npz')
        result['common_policy_input_reference']=str(common_path) if common_path else 'own first frozen reset'
        result['restoration_validity']=True
        result['initial_deg']=degrees(rollout.fridge(low));result['final_deg']=result['initial_deg'].copy();result['min_deg']=result['initial_deg'].copy()
        result['min_abs_deg']={d:abs(a) for d,a in result['initial_deg'].items()}
        origin=float(low.sim.data.time);result['origin_time']=origin
        assert not rollout.fridge(low)['success']
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        assert time.monotonic()-start<590 and time.time()<plan['deadline_unix'],'Worker/global budget'
        policy=SmolVLAPolicy.from_pretrained(str(model));policy.eval();policy.reset()
        assert policy.config.n_obs_steps==1 and policy.config.rtc_config is None
        assert getattr(policy,'rtc_processor',None) is None
        assert set(policy._queues)=={'action'} and not policy._queues['action']
        assert policy.config.chunk_size==policy.config.n_action_steps==50 and policy.config.num_steps==10
        pre,post=make_pre_post_processors(policy.config,pretrained_path=str(model))
        dump(folder/'manifest.json',dict(job=job,registered_plan_sha256=sha(OUT/'registered-plan.json'),
            checkpoint_sha256=job['checkpoint_sha256'],processors=plan['processors'],
            horizon=horizon,ignore_done=False,boundary=deepcopy(result),
            policy_context='n_obs_steps1, RTC disabled, empty action queue, direct stateless predict_action_chunk.',
            success_semantics='success/raw_all_door_success is observed predicate; policy_success excludes hard stops. Final report applies paired relative gate.',
            action_semantics='Native postprocessor output retained before irreversible controller threshold/reorder/clipping; training normalizes this output exactly once.',
            raw_observation='observations/t.npz is actual lossless RGB/state after t-1 actions. First common-policy-input.npz is loaded from reference only after current camera/state gate. Later policy inputs branch-current.'))
        video=av.open(str(folder/'rollout.mp4'),'w');stream=video.add_stream('libx264',rate=20)
        stream.width=768;stream.height=288;stream.pix_fmt='yuv420p'
        anchors=[]
        result.update(status='running',stop_reason='action_budget')
        with (folder/'steps.jsonl').open('x') as log:
            for local in range(1,cap+1):
                index=prefix+local
                if (local-1)%50==0:
                    assert time.monotonic()-start<590 and time.time()<plan['deadline_unix'],'Worker/global budget'
                    actual_path=save_obs(index)
                    policy_path=str(folder/'common-policy-input.npz') if local==1 else actual_path
                    current_input=first_input if local==1 else obs
                    batch=pre(dict(preprocess_observation(current_input),task=['CloseFridge']))
                    ns=noise_seed(job['kind'],seed,job.get('replicate',1),local)
                    assert ns==job['noise_seeds'][(local-1)//50]
                    torch.manual_seed(ns);noise=policy.model.sample_noise((1,50,32),next(policy.parameters()).device)
                    torch.save(noise.cpu(),folder/f'noise-{local:04d}.pt')
                    torch.cuda.synchronize();tick=time.monotonic()
                    with torch.inference_mode():chunk=post(policy.predict_action_chunk(batch,noise=noise)).detach().cpu().numpy()[0]
                    torch.cuda.synchronize();elapsed=time.monotonic()-tick
                    assert chunk.shape==(50,12) and np.isfinite(chunk).all() and len(policy._queues['action'])==0
                    np.save(folder/f'actions-{index:04d}.npy',chunk)
                    info=dict(pre_action=index,local_pre_action=local,seed=ns,noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),seconds=elapsed,observation_path=policy_path,policy_observation_path=policy_path,
                        observation_sha256=sha(Path(policy_path)),actual_observation_path=actual_path,actual_observation_sha256=sha(Path(actual_path)))
                    result['inferences'].append(info);result['inference_seconds']+=elapsed
                ds=sorted(eligible(seen,gap,False))
                if index in saved and ds:anchors.append(dict(pre_action=index,doors=ds,prior_gap={d:gap[d] for d in ds},observation=saved[index]))
                action=chunk[(local-1)%50];obs,tr,gate,con,elapsed=step(action,index)
                result['execution_seconds']+=elapsed;result['actions']=local;result['global_end_action']=index
                result['final_deg']=degrees(tr['fridge_after'])
                for d,a in result['final_deg'].items():
                    result['min_deg'][d]=min(result['min_deg'][d],a)
                    result['min_abs_deg'][d]=min(result['min_abs_deg'][d],abs(a))
                for k,v in con['peaks'].items():result['peaks'][k]=max(result['peaks'][k],v)
                for d,v in con['task_door_peaks_N'].items():result['per_door_peaks_N'][d]=max(result['per_door_peaks_N'].get(d,0.),v)
                for d,t in con['task_door_first'].items():
                    if t is None:continue
                    result['first_contact_seconds'].setdefault(d,t-origin)
                    if seen[d] and gap[d]>=5:
                        result['recontact_events'].append(dict(door=d,local_action=local,global_action=index,seconds=t-origin,prior_gap_actions=gap[d]))
                        result['first_recontact_seconds'].setdefault(d,t-origin)
                        result['first_recontact_action'].setdefault(d,local)
                result['success']=result['raw_all_door_success']=bool(tr['fridge_after']['success'])
                if result['success']:
                    result['first_success_action']=local;result['first_success_global_action']=index
                    result['first_success_seconds']=float(low.sim.data.time)-origin
                row=serial(dict(step=index,local_step=local,action=action,controller_input=tr['controller_input'],
                    controller_targets=tr['targets'],gripper_accumulator=tr['gripper_accumulator'],
                    state=obs['agent_pos'],physical=tr['after'],fridge=tr['fridge_after'],contact=con,
                    guard_failures=tr['guard_failures'],terminated=tr['terminated'],truncated=tr['truncated'],reward=tr['reward'],gate=gate))
                log.write(json.dumps(row)+'\n');log.flush()
                advance(seen,gap,con['task_door_loaded'])
                if not result['success'] and any(seen[d] and gap[d]==5 for d in doors):save_obs(index+1)
                for packet in stream.encode(av.VideoFrame.from_image(sheet(obs,f'{job["id"]} ACTION {index}'))):video.mux(packet)
                if tr['guard_failures']:result.update(hard_stop=True,status='hard_stop',stop_reason=str(tr['guard_failures']))
                result['policy_success']=bool(result['success'] and not result['hard_stop'])
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
        dump(folder/'final-controller.json',controller_boundary(low))
    except Exception as exc:
        result.update(status='technical_invalid',execution_validity=False,policy_success=False,stop_reason=str(exc),exception=traceback.format_exc())
    finally:
        close_errors=[]
        if video is not None:
            try:
                for packet in stream.encode():video.mux(packet)
                video.close()
            except Exception:close_errors.append(traceback.format_exc())
        if env is not None:
            try:env.close()
            except Exception:close_errors.append(traceback.format_exc())
        result['worker_seconds']=time.monotonic()-start
        if close_errors or result['worker_seconds']>600:
            result.update(status='technical_invalid',execution_validity=False,policy_success=False,
                finalization_errors=close_errors,stop_reason='Finalization error' if close_errors else 'Worker exceeded600seconds')
        dump(folder/'result.json',result)
    print(json.dumps(result),flush=True)
    return result



if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    worker(read(ap.parse_args().job))
