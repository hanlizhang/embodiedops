"""Preregistered 60-episode matrix. Fresh paired episodes, no retries or final seeds."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from instruction_control_audit import ROOT, HISTORY, BASE, PROCESSORS, read, dump, sha
from agent_workflow import episode_metrics
from instruction_storage_contract import evaluation_root

def detailed_metrics(folder):
    import numpy as np
    m = episode_metrics(folder)
    steps = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    reset = read(folder/'reset.json')
    cat = {g['id']:g for g in read(folder/'geometry-catalog.json')}
    fridge = set(read(folder/'study-fixture.json')['fridge_contact_geoms'])
    categories = {k:dict(peak_N=0., loaded_actions=set()) for k in ['fridge', 'self', 'other']}
    for line in (folder/'contacts.jsonl').read_text().splitlines():
        row = json.loads(line)
        for sample in row['samples']:
            for c in sample['robot_contacts']:
                force = c['force_N_contact_frame'][0]
                if c['constraint_address'] < 0 or force <= 1e-8:
                    continue
                geoms = [cat[i] for i in c['geom_ids']]
                category = 'self' if all(g['robot'] for g in geoms) else ('fridge' if any(g['name'] in fridge for g in geoms) else 'other')
                group = categories[category]
                group['peak_N'] = max(force, group['peak_N'])
                if force >= 1:
                    group['loaded_actions'].add(row['step'])
    for group in categories.values():
        group['loaded_action_indices'] = sorted(group['loaded_actions'])
        group['loaded_actions'] = len(group['loaded_actions'])
    joints = list(reset['fridge']['qpos'])
    angles = np.rad2deg([[r['fridge']['qpos'][j] for j in joints] for r in [reset]+steps])
    m.update(door_joints=joints, door_count=len(joints), per_door_progress_deg=(angles[0]-angles[-1]).tolist(),
         per_door_best_progress_deg=(angles[0]-angles.min(axis=0)).tolist(), contacts=categories,
         close_gripper_actions=sum(r['action'][11]>=.5 for r in steps),
         desired_goal_actions=sum(r['action'][4]>=.5 for r in steps),
         total_inference_seconds=sum(read(folder/'summary.json')['inference_seconds']))
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    args = ap.parse_args()
    study = args.study.resolve()
    verification=read(ROOT/'outputs/instruction-storage-audit/reload/result.json')
    assert verification['passed'] and verification['loaded_tensors']==500
    assert all(x['exact'] for group in verification['probes'].values() for x in group.values())
    out = evaluation_root()
    assert out.parent==study
    with (out/'evaluation-started.json').open('x') as f:
        json.dump(dict(started=time.time(),episode_cap=60),f)
    original_c=study/'C/checkpoints/000300/pretrained_model'
    assert sha(original_c/'model.safetensors')==verification['model_sha256']
    bundle=out/'inference-C'
    bundle.mkdir(exist_ok=False)
    for name in ['model.safetensors','config.json']:
        shutil.copyfile(original_c/name,bundle/name)
    for name in PROCESSORS:
        shutil.copyfile(BASE/name,bundle/name)
    dump(bundle/'inference-verification.json',dict(**verification,
         source=str(original_c),optimizer_scheduler_complete=False,training_replayed=False))
    models = {'A': BASE, 'B': HISTORY/'checkpoints/000300/pretrained_model',
              'C': bundle}
    hashes = {k:sha(p/'model.safetensors') for k,p in models.items()}
    assert hashes['A']=='96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b'
    assert hashes['B']=='0e011aa494e0d4e370b11feb5eec6e0c49af2fadd1f3f97ee03056909de0fe6d'
    assert hashes['C'] == verification['model_sha256']
    for p in models.values():
        assert all(sha(p/n) == sha(BASE/n) for n in PROCESSORS)
    scripts = ['phase1d_rollout.py','control_diagnostics.py','renderer_lifecycle.py',
               'phase1b_contract.py','instruction_control_evaluate.py','instruction_storage_contract.py',
               'agent_workflow.py','phase1a_contract.py']
    frozen = {n:sha(ROOT/'scripts'/n) for n in scripts}
    instructions = {}
    for seed in range(42,52):
        old = ROOT/'outputs/phase1d/evaluation'/f'trained-seed{seed}'
        instructions[str(seed)] = read(old/'reset.json')['task']
    plan = dict(checkpoints={k:str(v) for k,v in models.items()}, hashes=hashes,
        instructions=instructions, seeds=list(range(42,52)), episode_cap=60,
        actions_per_episode=600, episode_seconds=600, total_seconds=36000,
        rng='seed*100000+action_index at each native refill', execution_horizon=50,
        flow_steps=10, frozen_sources=frozen, registered_unix=time.time())
    plan.update(C_inference_verified=True,C_optimizer_scheduler_complete=False,
        training_replayed=False,preflight_seed=42,batch_size_scenes=1,storage_physical=str(out.resolve()),
        historical_scene_reset_hashes={str(s):read(ROOT/f'outputs/phase1d/evaluation/trained-seed{s}/reset.json')['hashes'] for s in range(42,52)})
    dump(out/'registered-plan.json', plan)
    for n in scripts:
        shutil.copyfile(ROOT/'scripts'/n, out/(n+'.txt'))
    started = time.monotonic()
    results = []
    try:
        for seed in range(42,52):
            assert evaluation_root()==out
            assert all(sha(p/'model.safetensors')==hashes[k] for k,p in models.items())
            dump(out/f'batch-{seed}-disk-before.json',dict(C=shutil.disk_usage(ROOT).free,D=shutil.disk_usage(out).free))
            for model in ['A','B','C']:
                for condition in ['NL','ID']:
                    assert len(results) < 60 and time.monotonic()-started < 36000
                    assert shutil.disk_usage(out).free > 2*1024**3
                    assert shutil.disk_usage(ROOT).free > 2*1024**3
                    assert all(sha(ROOT/'scripts'/n)==h for n,h in frozen.items())
                    folder = out/f'{model}-{condition}-{seed}'
                    cmd = [sys.executable,'-B','-u',str(ROOT/'scripts/phase1d_rollout.py'),
                        '--output',str(folder),'--seed',str(seed),'--instruction',
                        instructions[str(seed)] if condition=='NL' else 'CloseFridge',
                        '--checkpoint','lerobot/smolvla_robocasa' if model=='A' else str(models[model]),
                        '--workflow-diagnostics','--renderer-owner-cleanup','--study-observations']
                    if model=='A' and condition=='NL':
                        cmd += ['--reference',str(ROOT/f'outputs/phase1d/evaluation/trained-seed{seed}')]
                    elif model=='A' and condition=='ID':
                        cmd += ['--reference',str(out/f'A-NL-{seed}'),'--instruction-contrast']
                    elif model!='A':
                        cmd += ['--reference',str(out/f'A-{condition}-{seed}')]
                    dump(out/f'{folder.name}-command.json',cmd)
                    print(f'Starting {folder.name} ({len(results)+1}/60)',flush=True)
                    t = time.monotonic()
                    with (out/f'{folder.name}.log').open('x') as log:
                        p = subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                           timeout=min(600,36000-(time.monotonic()-started)))
                    elapsed = time.monotonic()-t
                    assert elapsed<=600 and p.returncode in (0,2)
                    m = detailed_metrics(folder)
                    m.update(model=model,condition=condition,seed=seed,worker_seconds=elapsed)
                    manifest = read(folder/'manifest.json')
                    assert manifest['model_sha256']==hashes[model]
                    assert read(folder/'paired-reset-validation.json')['passed']
                    assert manifest['rng_rule']=='manual_seed(seed * 100000 + action_index) before each new chunk'
                    assert read(folder/'reset.json')['hashes']==plan['historical_scene_reset_hashes'][str(seed)]
                    for name in PROCESSORS:
                        assert manifest['checkpoint_file_hashes'][name]==sha(BASE/name)
                    if model!='A':
                        reference_manifest=read(out/f'A-{condition}-{seed}/manifest.json')
                        for key in ['versions','simulator_sources','execution_horizon','rng_rule','renderer_owner_cleanup']:
                            assert manifest[key]==reference_manifest[key],key
                    from PIL import Image
                    import numpy as np
                    cameras=list(folder.glob('final-robot0_*.png'))
                    assert len(cameras)==3
                    for camera in cameras:
                        assert np.asarray(Image.open(camera)).max()>0,'Invalid final camera'
                    results.append(m)
                    dump(out/'results.json',results)
                    print(f'Finished {folder.name}: {m["termination"]}; final degrees {m["final_angles_deg"]}',flush=True)
                    assert m['valid'] or m['guard_failure'], 'Invalid episode; matrix stopped'
            assert len([m for m in results if m['seed']==seed])==6
            dump(out/f'batch-{seed}-complete.json',dict(contracts_passed=True,episodes=6,
                preflight=seed==42,C_free=shutil.disk_usage(ROOT).free,D_free=shutil.disk_usage(out).free))
        dump(out/'complete.json',dict(episodes=len(results),worker_seconds=time.monotonic()-started))
    except Exception as e:
        dump(out/'stopped.json',dict(reason=repr(e),completed=len(results),seconds=time.monotonic()-started))
        raise
    finally:
        assert all(sha(p/'model.safetensors')==hashes[k] for k,p in models.items())

if __name__=='__main__':
    main()
