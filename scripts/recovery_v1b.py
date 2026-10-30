"""One registered seed48 paired recovery experiment. No retries or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from instruction_control_audit import ROOT, BASE, PROCESSORS, read, dump, sha
from recovery_feasibility import max_error, camera_error

OLD = ROOT / 'outputs/agent-workflow-v1/recovery-feasibility-v1'
OUT = ROOT / 'outputs/agent-workflow-v1/recovery-feasibility-v1b'
ENV = Path('D:/Apps/miniconda/envs/embodiedops-robocasa/Lib/site-packages')
SOURCES = ['recovery_v1b.py', 'recovery_branches.py', 'recovery_geometry_audit.py',
           'recovery_feasibility.py', 'phase1d_rollout.py', 'phase1b_contract.py',
           'phase1a_contract.py', 'phase1c_replay.py', 'control_diagnostics.py',
           'renderer_lifecycle.py', 'instruction_disk_guard.py', 'instruction_control_audit.py']


def peak_contacts(samples, robot_ids, fridge_ids, door_ids, hand_ids):
    peaks = dict(all=0., fridge=0., self=0., door=0.)
    first = None
    for sample in samples:
        for c in sample['robot_contacts']:
            if c['constraint_address'] < 0:
                continue
            force = max(0., c['force_N_contact_frame'][0])
            ids = set(c['geom_ids'])
            peaks['all'] = max(peaks['all'], force)
            if ids.issubset(robot_ids): peaks['self'] = max(peaks['self'], force)
            if ids & fridge_ids: peaks['fridge'] = max(peaks['fridge'], force)
            if ids & door_ids and ids & hand_ids:
                peaks['door'] = max(peaks['door'], force)
                if force >= 1 and first is None: first = sample['time']
    return peaks, first


def check_direction(action, decision, rotation):
    if decision['stage'] == 'approach':
        points = np.asarray(decision['nearest_points'])
        direction = points[3:] - points[:3]
        assert float(np.dot(rotation @ (action[5:8] * .05), direction)) > 0, 'Reversed approach command'


def geometry_observation(expert):
    """Unbounded read-only proximity measurement; never gates the VLA's policy."""
    from recovery_geometry_audit import ordered_geometry_distance
    low = expert.low
    distance = min(ordered_geometry_distance(low.sim.model._model, low.sim.data._data,
        h, d, 100.)[0] for h in expert.spec['gripper_geoms'] for d in expert.door['geoms'])
    return dict(nearest_distance_m=distance, stage='approach' if distance > .012 else 'tangential_push')


def register():
    assert OUT.is_junction() and OUT.resolve() == Path('D:/EmbodiedOpsOutputs/recovery-feasibility-v1b').resolve()
    assert not (OUT/'registered-plan.json').exists()
    protected = read(OLD/'preserved-before.json')
    assert all(sha(ROOT/p) == h for p, h in protected.items())
    # Include all v1 artifacts (including invalid records and executed snapshots).
    for p in OLD.rglob('*'):
        if p.is_file(): protected[str(p.relative_to(ROOT))] = sha(p)
    dump(OUT/'preserved-before.json', protected)
    frozen = {str(ROOT/'scripts'/n): sha(ROOT/'scripts'/n) for n in SOURCES}
    package_files = [ENV/'lerobot/envs/robocasa.py', ENV/'robosuite/environments/base.py',
                     ENV/'lerobot/policies/smolvla/modeling_smolvla.py']
    rc = ROOT/'outputs/phase1b/sources/robocasa-56e355ccc64389dfc1b8a61a33b9127b975ba681/robocasa'
    package_files += [rc/'wrappers/gym_wrapper.py', rc/'utils/env_utils.py']
    frozen.update({str(p): sha(p) for p in package_files})
    snapshot = OUT/'executed-source'; snapshot.mkdir()
    for i, (p, h) in enumerate(frozen.items()):
        target = snapshot/f'{i:02d}-{Path(p).name}.txt'
        target.write_bytes(Path(p).read_bytes())
        assert sha(target) == h
    old = read(OLD/'registered-plan.json')
    spec = next(s for s in old['states'] if s['seed'] == 48)
    plan = dict(version='v1b', created_unix=time.time(), state=spec,
        branch_order=['B', 'A'], attempts_per_branch=1, recovery_actions=300,
        public_control_hz=20, recovery_simulation_seconds=15, original_task_cap=600,
        experiment_episode_cap=800, worker_seconds=min(600, old['worker_seconds']),
        total_seconds=3600, instruction='CloseFridge', checkpoint=str(BASE),
        revision='f7af588d11e30b0069f200792fb285904b74c80b',
        checkpoint_sha256=old['checkpoint_sha256'], processors={n:sha(BASE/n) for n in PROCESSORS},
        tolerances=old['tolerances'], hard_force_ceiling_N=1500, disk_floor_GiB=2,
        noise_seeds=[931001,931002,931003,931004,931005,931006],
        noise_method='At each H50 boundary torch.manual_seed(seed), native sample_noise((1,50,32), policy device); save tensor and SHA256 of CPU numpy bytes before predict_action_chunk. Exactly one sequence.',
        flow_steps=10, execution_horizon=50,
        observation_rule='First input may use v1 reference only after actual camera gate passes; subsequent chunks use branch-current RGB and 16D state. Save all actual observations.',
        restoration='Fresh seeded scene and complete 500 action replay; compare every qpos/qvel/16D state and reset scene/controller/cameras. No simulator state assignment.',
        horizon_rule='Single reset only. After initial reset, before prefix, set low.horizon=800 and low.ignore_done=False; assert timestep=0. LeRobot episode_length=800. Preserve all done/truncated and success; no auto reset.',
        horizon_audit='Historical runner capped 600. LeRobot _max_episode_steps is metadata, not forwarded to RoboCasa. RoboCasa create_env sets ignore_done=True. Gym step forwards low done, truncated=False, actual _check_success. v1b explicitly enables low horizon stopping.',
        success_rule='First true all-door CloseFridge predicate ends either branch; log first success and all guards independently.',
        contact_rule='v1 peak: maximum positive normal force over active robot contacts at every physics substep, recovery only. Relative B ceiling=1.25*A measured peak+25 N; unavailable if A invalid/incomplete.',
        door_contact_rule='Active gripper-door contact >=1 N; first substep time minus recovery start sim time.',
        stops=['true task success','300 recovery actions','any done/truncated','inherited finite/action/motion/MuJoCo guards','1500 N ceiling','2 GiB disk reserve','600s whole process','reversed approach command'],
        common_invalidity='Stop study on replay or initial camera validity failure; no retry or alternate seed.',
        expert='Unchanged corrected GeometryExpert from recovery_branches; external controller, not neural action expert.',
        acceptance='B restored, true success, no hard guard, and relative contact gate passed; A restored and completed unsuccessfully required for differential corrective-behavior claim.',
        source_hashes=frozen, original_junction=str(OLD.resolve()), output_junction=str(OUT.resolve()))
    with (OUT/'registered-plan.json').open('x', encoding='utf-8') as f: json.dump(plan,f,indent=2)
    print('Registered plan, sources and preservation', flush=True)


def worker(branch):
    start = time.monotonic()
    plan = read(OUT/'registered-plan.json'); spec = plan['state']
    folder = OUT/branch; folder.mkdir(exist_ok=False)
    result = dict(branch=branch, restoration_validity=False, execution_status='initializing',
        stop_reason=None, actual_recovery_actions=0, prefix_actions=0, true_task_success=None,
        hard_guard_status='not_reached', relative_contact_gate_status='pending',
        prefix_wall_seconds=0., inference_wall_seconds=0., recovery_wall_seconds=0.,
        first_success_seconds=None, first_success_action=None, first_door_contact_seconds=None,
        peak_normal_N=0., peak_fridge_N=0., peak_self_N=0., peak_door_N=0.,
        max_replay_errors=dict(qpos=0.,qvel=0.,state=0.), inferences=[])
    env = video = None
    try:
        assert Path(sys.prefix).name == 'embodiedops-robocasa'
        assert all(sha(Path(p)) == h for p,h in plan['source_hashes'].items())
        assert sha(BASE/'model.safetensors') == plan['checkpoint_sha256']
        assert all(sha(BASE/n) == h for n,h in plan['processors'].items())
        import phase1d_rollout as rollout
        from phase1b_contract import CAMERAS, physical, serial, sheet
        from control_diagnostics import ContactRecorder
        from instruction_disk_guard import sample
        from lerobot.envs.robocasa import RoboCasaEnv, convert_action
        from recovery_branches import GeometryExpert
        from PIL import Image
        import renderer_lifecycle
        import importlib.metadata as md
        import av
        rollout.OUT = folder
        os.environ['NUMBA_CACHE_DIR'] = str(OUT/'numba-cache')
        renderer_lifecycle.install()
        source = ROOT/spec['source']
        versions = read(source/'manifest.json')['versions']
        assert {k:md.version(k) for k in versions} == versions
        class SingleEpisode(RoboCasaEnv):
            def reset(self, *args, **kwargs):
                assert not getattr(self, '_reset_once', False), 'Second episode prohibited'
                self._reset_once = True
                return super().reset(*args, **kwargs)
            def step(self, action):
                raw,reward,done,truncated,info = self._env.step(convert_action(action))
                return self._format_raw_obs(raw),reward,done or bool(info.get('success',False)),truncated,info
        env = SingleEpisode('CloseFridge', camera_name=CAMERAS, episode_length=800)
        np.random.seed(48); env._ensure_env()
        np.random.seed(48); obs,_ = env.reset(seed=48)
        low = env._env.env
        result['horizon_before'] = dict(horizon=low.horizon,ignore_done=low.ignore_done,timestep=low.timestep)
        assert low.timestep == 0
        low.horizon = plan['experiment_episode_cap']; low.ignore_done = False
        result['horizon_after'] = dict(horizon=low.horizon,ignore_done=low.ignore_done,timestep=low.timestep)
        reset = physical(low); old = read(source/'reset.json')
        assert all(max_error(reset[k],old['physical'][k]) == 0 for k in ('qpos','qvel'))
        assert max_error(obs['agent_pos'],old['state']) == 0
        assert rollout.canonical_scene(serial(low.get_ep_meta())) == rollout.canonical_scene(read(source/'scene.json')['ep_meta'])
        assert serial(low.robots[0].composite_controller_config) == read(source/'scene.json')['controller']
        result['reset_camera_errors'] = {}
        for cam in CAMERAS:
            e = camera_error(obs['pixels'][cam],np.asarray(Image.open(source/f'reset-{cam}.png')))
            result['reset_camera_errors'][cam] = e
            assert e['max'] <= 1 and e['mean'] <= .001, 'Reset camera gate failed'
        recorder = ContactRecorder(low.sim,folder)
        def step(action,index):
            assert time.monotonic()-start < plan['worker_seconds'], 'Worker deadline'
            for p in [ROOT,OUT]: assert sample(p)['above_floor'], 'Disk reserve'
            rollout.action_gate(env,action)
            with recorder.step(index): observation,trace = rollout.traced_step(env,action,index)
            failures = list(trace['guard_failures'])
            for cam in CAMERAS:
                try: camera_error(observation['pixels'][cam],observation['pixels'][cam])
                except AssertionError: failures.append('Invalid actual camera '+cam)
            peak = max((c['force_N_contact_frame'][0] for s in recorder.last_samples for c in s['robot_contacts']),default=0)
            if peak > 1500: failures.append('Hard force ceiling')
            trace['guard_failures'] = failures
            return observation,trace
        prefix_start = time.monotonic()
        with (folder/'replay.jsonl').open('x') as log:
            rows = [json.loads(s) for s in (source/'steps.jsonl').read_text().splitlines()]
            for row in rows[:500]:
                obs,trace = step(np.asarray(row['action'],dtype=np.float32),row['step'])
                result['prefix_actions'] += 1
                errors = {k:max_error(trace['after'][k],row['physical'][k]) for k in ('qpos','qvel')}
                errors['state'] = max_error(obs['agent_pos'],row['returned_state'])
                for k,v in errors.items(): result['max_replay_errors'][k] = max(v,result['max_replay_errors'][k])
                log.write(json.dumps(serial(dict(index=row['step'],errors=errors,guards=trace['guard_failures'],terminated=trace['terminated'],truncated=trace['truncated'])))+'\n'); log.flush()
                result['prefix_wall_seconds'] = time.monotonic()-prefix_start
                assert not trace['guard_failures'], str(trace['guard_failures'])
                assert all(v <= plan['tolerances'][k] for k,v in errors.items()), 'Historical prefix divergence'
                assert not trace['terminated'] and not trace['truncated'], 'Prefix termination'
                if row['step'] % 100 == 0: print(branch,'prefix',row['step'],flush=True)
        reference = np.load(OLD/(spec['id']+'-restoration-audit')/'reconstructed-observation.npz')
        result['boundary_camera_errors'] = {c:camera_error(obs['pixels'][c],reference[c]) for c in CAMERAS}
        assert all(e['max'] <= 1 and e['mean'] <= .001 for e in result['boundary_camera_errors'].values()), 'Boundary camera gate failed'
        assert max_error(obs['agent_pos'],reference['state']) <= 1e-7
        result.update(restoration_validity=True,execution_status='running',hard_guard_status='clear',true_task_success=False)
        dump(folder/'result.json',result)
        observation_dir = folder/'observations'; observation_dir.mkdir()
        def save_obs(i):
            np.savez_compressed(observation_dir/f'{i:03d}.npz',state=obs['agent_pos'],**obs['pixels'])
        save_obs(0); sheet(obs,'BEFORE '+branch).save(folder/'before.png')
        initial = rollout.fridge(low)
        result['initial_deg'] = np.rad2deg(list(initial['qpos'].values())).tolist()
        result['min_deg'] = result['initial_deg'].copy()
        expert = GeometryExpert(low)  # Read-only geometry diagnostics on A as well.
        dump(folder/'geometry-spec.json',expert.spec)
        door_ids = set(expert.door['geoms']); hand_ids = set(expert.spec['gripper_geoms'])
        fridge_ids = {g['id'] for g in recorder.catalog if g['name'] in set(low.fxtr.contact_geoms)}
        if branch == 'A':
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.envs.utils import preprocess_observation
            policy = SmolVLAPolicy.from_pretrained(str(BASE)); policy.eval(); policy.reset()
            assert policy.config.chunk_size == policy.config.n_action_steps == 50
            assert policy.config.num_steps == 10
            pre,post = make_pre_post_processors(policy.config,pretrained_path=str(BASE))
        video = av.open(str(folder/'rollout.mp4'),mode='w')
        stream = video.add_stream('libx264',rate=20); stream.width=768; stream.height=288; stream.pix_fmt='yuv420p'
        sim_start = float(low.sim.data.time); recovery_start = time.monotonic()
        result['recovery_start_sim_time'] = sim_start
        result['stop_reason'] = 'recovery_budget_300'
        with (folder/'steps.jsonl').open('x') as log:
            for i in range(300):
                if branch == 'A' and i % 50 == 0:
                    current = obs if i else {'agent_pos':reference['state'].copy(),'pixels':{c:reference[c].copy() for c in CAMERAS}}
                    np.savez_compressed(folder/f'policy-input-{i:03d}.npz',state=current['agent_pos'],**current['pixels'])
                    batch = pre(dict(preprocess_observation(current),task=['CloseFridge']))
                    assert batch['observation.state'].shape == (1,16)
                    assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v,torch.Tensor))
                    seed = plan['noise_seeds'][i//50]; torch.manual_seed(seed)
                    noise = policy.model.sample_noise((1,50,32),next(policy.parameters()).device)
                    torch.save(noise.cpu(),folder/f'noise-{i:03d}.pt')
                    noise_hash = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
                    torch.cuda.synchronize(); t = time.monotonic()
                    with torch.inference_mode(): chunk = post(policy.predict_action_chunk(batch,noise=noise)).detach().cpu().numpy()[0]
                    torch.cuda.synchronize(); elapsed = time.monotonic()-t
                    result['inference_wall_seconds'] += elapsed
                    result['inferences'].append(dict(after_action=i,seed=seed,noise_sha256=noise_hash,seconds=elapsed,input_sha256=sha(folder/f'policy-input-{i:03d}.npz'),input_source='audited_common_reference' if i==0 else 'current_branch_observation'))
                    assert chunk.shape == (50,12) and np.isfinite(chunk).all()
                    np.save(folder/f'actions-{i:03d}.npy',chunk)
                if branch == 'B':
                    geometry_action, decision = expert.action()
                    check_direction(geometry_action,decision,low.robots[0].composite_controller.part_controllers['right'].origin_ori)
                else: decision = geometry_observation(expert)
                action = geometry_action if branch == 'B' else chunk[i%50]
                obs,trace = step(action,501+i)
                result['actual_recovery_actions'] += 1
                peaks,first = peak_contacts(recorder.last_samples,recorder.robot_ids,fridge_ids,door_ids,hand_ids)
                for name,key in [('all','normal'),('fridge','fridge'),('self','self'),('door','door')]:
                    result['peak_'+key+'_N'] = max(result['peak_'+key+'_N'],peaks[name])
                if first is not None and result['first_door_contact_seconds'] is None: result['first_door_contact_seconds'] = first-sim_start
                angle = np.rad2deg(list(trace['fridge_after']['qpos'].values())).tolist()
                result['final_deg'] = angle; result['min_deg'] = np.minimum(result['min_deg'],angle).tolist()
                result['recovery_simulation_seconds'] = float(low.sim.data.time)-sim_start
                success = bool(trace['fridge_after']['success'])
                if success and result['first_success_action'] is None:
                    result.update(first_success_action=i+1,first_success_seconds=result['recovery_simulation_seconds'],true_task_success=True)
                save_obs(i+1)
                record = dict(index=i+1,historical_index=501+i,action=action,decision=decision,
                    physical=trace['after'],state=obs['agent_pos'],fridge=trace['fridge_after'],
                    guards=trace['guard_failures'],terminated=trace['terminated'],truncated=trace['truncated'],
                    warnings=trace['warnings'],contact_peaks=peaks)
                log.write(json.dumps(serial(record))+'\n'); log.flush()
                for packet in stream.encode(av.VideoFrame.from_image(sheet(obs,f'{branch} RECOVERY {i+1}'))): video.mux(packet)
                if (i+1)%50 == 0 or success or trace['guard_failures']:
                    after_geometry = geometry_observation(expert)
                    stage = dict(action=i+1,sim_seconds=result['recovery_simulation_seconds'],angle_deg=angle,
                        closure_deg=(np.asarray(result['initial_deg'])-angle).tolist(),clearance_m=after_geometry['nearest_distance_m'],
                        expert_stage=after_geometry['stage'] if branch=='B' else None,
                        effective_door_contact_established=result['first_door_contact_seconds'] is not None,
                        contact_peaks=peaks,guards=trace['guard_failures'])
                    with (folder/'stages.jsonl').open('a') as f: f.write(json.dumps(stage)+'\n')
                    print(branch,stage,flush=True)
                result['recovery_wall_seconds'] = time.monotonic()-recovery_start-result['inference_wall_seconds']
                dump(folder/'result.json',result)
                if trace['guard_failures']:
                    result.update(hard_guard_status='triggered',stop_reason='; '.join(trace['guard_failures']),execution_status='hard_stop'); break
                if success or trace['terminated'] or trace['truncated']:
                    result['stop_reason'] = 'true_task_success' if success else 'simulator_termination'
                    break
            if result['execution_status'] == 'running': result['execution_status'] = 'completed'
    except Exception as exc:
        import traceback
        result.update(execution_status='incomplete',stop_reason=str(exc),exception=traceback.format_exc())
        if result['restoration_validity']: result['hard_guard_status'] = 'exception_stop'
    finally:
        if video is not None:
            for packet in stream.encode(): video.mux(packet)
            video.close()
        if env is not None:
            if result['restoration_validity']: sheet(obs,'AFTER '+branch).save(folder/'after.png')
            env.close()
        result['worker_wall_seconds'] = time.monotonic()-start
        dump(folder/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['execution_status'] in ('completed','hard_stop') else 2


def run():
    plan = read(OUT/'registered-plan.json')
    with (OUT/'attempts-started.json').open('x') as f: json.dump(dict(time=time.time()),f)
    status = dict(attempts=[],complete=False)
    try:
        for branch in plan['branch_order']:
            remaining = plan['total_seconds']-(time.time()-plan['created_unix'])
            assert remaining > plan['worker_seconds']+120, 'Overall deadline: no new work'
            assert all(sha(Path(p))==h for p,h in plan['source_hashes'].items())
            status['attempts'].append(branch); dump(OUT/'run-status.json',status)
            with (OUT/f'{branch}.log').open('x') as log:
                process = subprocess.run([sys.executable,'-B','-u',str(Path(__file__)),'worker','--branch',branch],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=plan['worker_seconds'])
            result = read(OUT/branch/'result.json')
            print(branch,result['execution_status'],result['stop_reason'],flush=True)
            assert result['restoration_validity'], 'Common restoration invalidity; paired study stopped'
            assert process.returncode == 0, 'Worker exception; study stopped'
        status['complete'] = True
    except Exception as exc:
        status['stop_reason'] = str(exc)
        if isinstance(exc,subprocess.TimeoutExpired):
            p = OUT/branch/'result.json'
            r = read(p) if p.exists() else dict(branch=branch,restoration_validity=False,actual_recovery_actions=0,true_task_success=None)
            r.update(execution_status='incomplete',stop_reason='whole_process_timeout',hard_guard_status='timeout')
            dump(p,r)
    finally:
        protected = read(OUT/'preserved-before.json')
        status['preservation'] = dict(checked=len(protected),changed=[p for p,h in protected.items() if sha(ROOT/p)!=h])
        status['sources_unchanged'] = all(sha(Path(p))==h for p,h in plan['source_hashes'].items())
        status['junctions_unchanged'] = str(OLD.resolve())==plan['original_junction'] and str(OUT.resolve())==plan['output_junction']
        dump(OUT/'run-status.json',status)
    print(json.dumps(status,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode',choices=['register','run','worker'])
    parser.add_argument('--branch',choices=['A','B']); args = parser.parse_args()
    if args.mode == 'register': register()
    elif args.mode == 'run': run()
    else: sys.exit(worker(args.branch))
