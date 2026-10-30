"""Bounded failure-state registration and physical prefix-restoration gate.

Historical qpos/qvel are measurements, never assigned as a simulator snapshot.
Reconstruction executes the saved controller actions from a verified fresh reset.
No candidate branch is authorized unless this gate passes.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from instruction_control_audit import ROOT, BASE, PROCESSORS, read, dump, sha

LOGICAL = ROOT / 'outputs/agent-workflow-v1/recovery-feasibility-v1'
PHYSICAL = Path('D:/EmbodiedOpsOutputs/recovery-feasibility-v1')
HISTORY = ROOT / 'outputs/agent-workflow-v1/instruction-control-v1/attempt02/evaluation'


def output_root():
    assert LOGICAL.is_junction() and LOGICAL.resolve() == PHYSICAL.resolve()
    return LOGICAL


def max_error(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a - b))) if a.size else 0.


def camera_error(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape == (256, 256, 3)
    assert a.dtype == b.dtype == np.uint8 and a.max() > 0 and b.max() > 0
    difference = np.abs(a.astype(float) - b.astype(float))
    return dict(max=float(difference.max()), mean=float(difference.mean()))


def register():
    import numpy as np
    out = output_root()
    assert not (out/'registered-plan.json').exists()
    states = []
    preserved = {}
    for seed, action in [(48, 500), (43, 550)]:
        source = HISTORY/f'A-ID-{seed}'
        rows = [json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()]
        row = rows[action-1]
        assert len(rows) == 600 and row['step'] == action
        assert not read(source/'summary.json')['final_fridge']['success']
        assert not row['fridge']['success'] and not any(r['guard_failures'] for r in rows)
        assert len(row['fridge']['qpos']) == 1
        opening = np.rad2deg(list(row['fridge']['qpos'].values())).tolist()
        states.append(dict(id=f'A-ID-{seed}-after-{action}', seed=seed, action=action,
            source=str(source.relative_to(ROOT)), split='development', opening_deg=opening,
            physical=row['physical'], observation_state=row['returned_state'],
            preceding_50_progress_deg=np.rad2deg(np.array(list(rows[action-51]['fridge']['qpos'].values()))-np.array(list(row['fridge']['qpos'].values()))).tolist(),
            selection='Single-door incomplete closure in failed pretrained ID rollout; stalled scene48 and late incomplete scene43; no branch outcomes used'))
        for p in source.rglob('*'):
            if p.is_file(): preserved[str(p.relative_to(ROOT))] = sha(p)
    # Freeze all selected historical A/B/C model identities, without loading them.
    old_plan = read(HISTORY/'registered-plan.json')
    for arm, directory in old_plan['checkpoints'].items():
        p = Path(directory)/'model.safetensors'
        assert sha(p) == old_plan['hashes'][arm]
        preserved[str(p)] = old_plan['hashes'][arm]
    for n in PROCESSORS:
        preserved[str(BASE/n)] = sha(BASE/n)
    plan = dict(states=states, instruction='CloseFridge', checkpoint=str(BASE),
        checkpoint_sha256=old_plan['hashes']['A'], horizon=50,
        candidate_noise_seeds=[930001,930002,930003,930004],
        restoration='Fresh reset plus complete recorded action prefix; no qpos/qvel assignment',
        tolerances=dict(qpos=1e-8,qvel=1e-8,state=1e-7,camera_max=1,camera_mean=.001),
        replay_actions_cap=1200, worker_seconds=600, total_seconds=5400,
        branch_attempts_cap=12, reserved_seeds_excluded=[120,121,122,123],
        recovery_success='actual CloseFridge predicate, all doors',
        useful_progress='>=5 degrees closure, no guard and no force regression',
        force_regression='peak normal force <=1.25*matched control+25 N',
        hard_force_ceiling_N=1500, inherited_guards=True, disk_floor_GiB=2,
        no_training=True, created_unix=time.time())
    dump(out/'preserved-before.json',preserved)
    dump(out/'registered-plan.json',plan)
    print(json.dumps(plan,indent=2))


def audit(state_id):
    import numpy as np
    from PIL import Image
    import phase1d_rollout as rollout
    from phase1b_contract import CAMERAS, physical, serial
    from control_diagnostics import ContactRecorder
    from instruction_disk_guard import sample
    import renderer_lifecycle
    from lerobot.envs.robocasa import RoboCasaEnv, convert_action
    out = output_root()
    plan = read(out/'registered-plan.json')
    spec = next(s for s in plan['states'] if s['id']==state_id)
    folder = out/(state_id+'-restoration-audit')
    folder.mkdir(exist_ok=False)
    rollout.OUT = folder
    os.environ['NUMBA_CACHE_DIR'] = str(out/'numba-cache')
    renderer_lifecycle.install()
    source = ROOT/spec['source']
    rows = [json.loads(s) for s in (source/'steps.jsonl').read_text().splitlines()]
    start = time.monotonic()
    result = dict(passed=False, actions=0, state_id=state_id, max_errors={},
                  inference_calls=0, snapshot_assignment=False)
    env = None
    try:
        assert Path(sys.prefix).name == 'embodiedops-robocasa'
        import importlib.metadata as md
        manifest = read(source/'manifest.json')
        assert {k:md.version(k) for k in manifest['versions']} == manifest['versions']
        assert sha(BASE/'model.safetensors') == plan['checkpoint_sha256']
        for p in [ROOT, out]: assert sample(p)['above_floor']
        class SingleEpisode(RoboCasaEnv):
            def step(self, action):
                raw, reward, done, truncated, info = self._env.step(convert_action(action))
                success = bool(info.get('success', False))
                return self._format_raw_obs(raw), reward, done or success, truncated, info
        env = SingleEpisode('CloseFridge', camera_name=CAMERAS)
        np.random.seed(spec['seed'])
        env._ensure_env()
        np.random.seed(spec['seed'])
        obs, _ = env.reset(seed=spec['seed'])
        low = env._env.env
        baseline = read(source/'reset.json')
        reset = physical(low)
        for key in ('qpos','qvel'):
            assert max_error(reset[key],baseline['physical'][key]) == 0, 'Reset differs: '+key
        assert max_error(obs['agent_pos'],baseline['state']) == 0
        assert rollout.canonical_scene(serial(low.get_ep_meta())) == rollout.canonical_scene(read(source/'scene.json')['ep_meta'])
        assert serial(low.robots[0].composite_controller_config) == read(source/'scene.json')['controller']
        result['reset_cameras'] = {}
        for cam in CAMERAS:
            error = camera_error(obs['pixels'][cam], np.asarray(Image.open(source/f'reset-{cam}.png')))
            result['reset_cameras'][cam] = error
            assert error['max'] <= 1 and error['mean'] <= .001, 'Reset rendering differs'
        recorder = ContactRecorder(low.sim,folder)
        with (folder/'replay.jsonl').open('x') as log:
            for row in rows[:spec['action']+plan['horizon']]:
                assert time.monotonic()-start < plan['worker_seconds'], 'Runtime limit'
                for p in [ROOT,out]: assert sample(p)['above_floor'], 'Disk reserve'
                action = np.asarray(row['action'],dtype=np.float32)
                rollout.action_gate(env,action)
                with recorder.step(row['step']): obs, trace = rollout.traced_step(env,action,row['step'])
                result['actions'] += 1
                errors = {k:max_error(trace['after'][k],row['physical'][k]) for k in ('qpos','qvel')}
                errors['state'] = max_error(obs['agent_pos'],row['returned_state'])
                for k,v in errors.items(): result['max_errors'][k]=max(v,result['max_errors'].get(k,0))
                record = dict(action_index=row['step'],errors=errors,fridge=trace['fridge_after'],
                    physical=trace['after'],guards=trace['guard_failures'],targets=trace['targets'])
                log.write(json.dumps(serial(record))+'\n'); log.flush()
                assert not trace['guard_failures'], 'Inherited physical guard'
                for cam in CAMERAS: camera_error(obs['pixels'][cam],obs['pixels'][cam])
                assert all(v <= plan['tolerances'][k] for k,v in errors.items()), 'Historical physical replay diverged'
                peak=max((c['force_N_contact_frame'][0] for s in recorder.last_samples for c in s['robot_contacts']),default=0)
                assert peak <= plan['hard_force_ceiling_N'], 'Contact force ceiling'
                if row['step']==spec['action']:
                    np.savez_compressed(folder/'reconstructed-observation.npz',state=obs['agent_pos'],**obs['pixels'])
                    result['reached_registered_state']=True
                if row['step']%50==0: print('Replay',row['step'],errors,flush=True)
                assert not trace['terminated'] and not trace['truncated'], 'Unexpected replay termination'
        result['passed']=True
    except Exception as exc:
        import traceback
        result.update(reason=str(exc),exception=traceback.format_exc())
    finally:
        result['seconds']=time.monotonic()-start
        dump(folder/'result.json',result)
        if env is not None: env.close()
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['passed'] else 2


def run_audits():
    import subprocess
    out=output_root()
    plan=read(out/'registered-plan.json')
    marker=out/'audit-started.json'
    with marker.open('x') as f: json.dump(dict(started=time.time()),f)
    frozen={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/phase1d_rollout.py',ROOT/'scripts/phase1b_contract.py',ROOT/'scripts/control_diagnostics.py',ROOT/'scripts/renderer_lifecycle.py']}
    dump(out/'audit-source-hashes.json',frozen)
    status=dict(audits=[],branches_run=0,expert_attempts=0,training_updates=0)
    try:
        for spec in plan['states']:
            cmd=[sys.executable,'-B','-u',str(Path(__file__)),'audit','--state',spec['id']]
            with (out/(spec['id']+'.log')).open('x') as log:
                worker=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=plan['worker_seconds'])
            result=read(out/(spec['id']+'-restoration-audit')/'result.json')
            status['audits'].append(result)
            assert worker.returncode==0 and result['passed'], 'Restoration gate failed; causal branches prohibited'
        status['restoration_gate_passed']=True
    except Exception as exc:
        status.update(stopped=True,reason=str(exc),restoration_gate_passed=False)
    finally:
        changed=[p for p,h in read(out/'preserved-before.json').items() if sha(ROOT/p)!=h]
        status['preservation']=dict(checked=len(read(out/'preserved-before.json')),changed=changed)
        status['source_unchanged']=all(sha(ROOT/p)==h for p,h in frozen.items())
        dump(out/'audit-summary.json',status)
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['register','audit','run-audits'])
    ap.add_argument('--state')
    args=ap.parse_args()
    if args.mode=='register': register()
    elif args.mode=='audit': sys.exit(audit(args.state))
    else: run_audits()
