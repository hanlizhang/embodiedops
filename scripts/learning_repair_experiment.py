"""Fixed sustained learning comparison using existing training/robotics adapters.

No diagnosis framework or model interface. Research evaluation is explicitly
separate from automatic promotion. All runs and checkpoints are exclusive.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from agent_workflow import ROOT, BASE, MODEL, read, rows, dump, sha, episode_metrics, acceptance


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT/'outputs/agent-workflow-v1')
    plan=read(ROOT/'environment/learning-repair-v1.json')
    assert not set(plan['diagnostic_seeds']) & set(plan['final_seeds'])
    assert len(plan['arms'])*plan['training']['steps']==plan['max_optimizer_updates_total']
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    dump(out/'registered-plan.json',plan)
    for name in ['learning_repair_experiment.py','learning_sampling.py','phase2_train_v1.py','phase1d_rollout.py','agent_workflow.py','control_diagnostics.py']:
        shutil.copyfile(ROOT/'scripts'/name,out/(name+'.txt'))
    protected=[BASE/'model.safetensors',ROOT/'outputs/phase1d/training-v2/checkpoints/000300/pretrained_model/model.safetensors',
               ROOT/'outputs/phase2/sanity-v1/checkpoints/000512/pretrained_model/model.safetensors',
               ROOT/'outputs/agent-workflow-v1/run-01/learning-candidate-v2/checkpoints/000512/pretrained_model/model.safetensors']
    preservation={str(p.relative_to(ROOT)):sha(p) for p in protected};dump(out/'preservation-before.json',preservation)
    process_seconds=0.;episodes=0

    def event(kind,**data):
        with (out/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event=kind,elapsed_seconds=time.monotonic()-started,**data))+'\n')
        print(kind,json.dumps(data)[:350],flush=True)

    def run(cmd,name,limit,allowed=(0,)):
        nonlocal process_seconds
        remaining=plan['max_wall_seconds']-(time.monotonic()-started)
        if remaining<=0:raise RuntimeError('Registered wall budget exhausted')
        event('worker_start',name=name,command=cmd,timeout=min(remaining,limit));tick=time.monotonic()
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(remaining,limit))
        duration=time.monotonic()-tick;process_seconds+=duration
        event('worker_end',name=name,seconds=duration,returncode=r.returncode)
        if r.returncode not in allowed:raise RuntimeError(f'{name} failed; preserve logs; no automatic rerun')

    try:
        selected={}
        for number,arm in enumerate(plan['arms'],1):
            settings=dict(plan['training'],sampling=arm)
            cfg=out/(arm+'.json');dump(cfg,settings);folder=out/arm
            cmd=[sys.executable,'-B','-u','scripts/phase2_train_v1.py','--config',str(cfg),'--output',str(folder),
                 '--dataset-cache',str(ROOT/f'outputs/lrc-v1-{number}')]
            event('training_arm',arm=arm,initialization='original pretrained',objective='native padding-masked flow matching',settings=settings)
            run(cmd,arm+'-train',plan['training_timeout_seconds'])
            split=read(folder/'split.json');updates=rows(folder/'updates.jsonl')
            assert len(updates)==5000 and len({r['episode'] for r in updates[:461]})==461
            assert all(r['all_trainable_gradients_finite'] and r['tensors_with_grad']==155 for r in updates)
            assert not set(split['train_episodes']) & set(split['validation_episodes'])
            for step in plan['checkpoint_steps']:
                path=folder/'checkpoints'/f'{step:06d}'
                run(cmd+['--verify',str(path)],arm+f'-reload-{step}',plan['reload_timeout_seconds'])
                assert read(folder/f'reload-{step:06d}.json')['passed']
            choice=read(folder/'selected-checkpoint.json');checkpoint=folder/choice['checkpoint']/'pretrained_model'
            selected[arm]=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint/'model.safetensors'),selection=choice,
                validation_improved=choice['score']<choice['pretrained_score'],research_only=True)
            event('checkpoint_frozen_before_simulation',arm=arm,**selected[arm])
        splits=[read(out/a/'split.json') for a in plan['arms']]
        assert [r['episode'] for r in splits[0]['training_order']]==[r['episode'] for r in splits[1]['training_order']]
        assert splits[0]['validation_samples']==splits[1]['validation_samples'] and splits[0]['training_probe']==splits[1]['training_probe']
        dump(out/'selected-checkpoints.json',selected)
        policy=read(ROOT/plan['acceptance_policy_source'])['acceptance']
        evaluations={}
        for stage in ('diagnostic','final'):
            metrics={};pairs={a:[] for a in plan['arms']}
            for seed in plan[stage+'_seeds']:
                reference=None
                for arm in ['pretrained']+plan['arms']:
                    episodes+=1;assert episodes<=plan['max_simulator_episodes']
                    folder=out/f'{stage}-{seed}-{arm}'
                    checkpoint=MODEL if arm=='pretrained' else selected[arm]['checkpoint']
                    if arm!='pretrained':assert sha(Path(checkpoint)/'model.safetensors')==selected[arm]['sha256']
                    cmd=[sys.executable,'-B','-u','scripts/phase1d_rollout.py','--output',str(folder),
                         '--seed',str(seed),'--checkpoint',checkpoint,'--instruction',plan['instruction'],'--workflow-diagnostics']
                    if reference:cmd+=['--reference',str(reference)]
                    event('research_evaluation',stage=stage,seed=seed,arm=arm,validation_warning=arm!='pretrained' and not selected[arm]['validation_improved'])
                    run(cmd,folder.name,plan['episode_timeout_seconds'],allowed=(0,2))
                    if not (folder/'summary.json').exists():raise RuntimeError('Missing episode summary')
                    m=episode_metrics(folder)
                    reset=read(folder/'reset.json');m['scene_type']='two_door' if len(reset['fridge']['qpos'])==2 else 'single_door'
                    metrics[f'{seed}-{arm}']=m
                    if arm=='pretrained':reference=folder
                    else:
                        matched=(folder/'paired-reset-validation.json').exists() and read(folder/'paired-reset-validation.json')['passed']
                        pairs[arm].append(dict(seed=seed,baseline=metrics[f'{seed}-pretrained'],candidate=m,matched_reset=matched))
                    event('robot_result',stage=stage,seed=seed,arm=arm,metrics=m)
                    # A physical motion guard ends that episode and rejects promotion;
                    # it is retained as a failed scene, not replaced or retried.
                    if not m['valid'] and not m['guard_failure']:
                        dump(out/'incomplete-evaluation.json',dict(stage=stage,metrics=metrics,pairs=pairs))
                        raise RuntimeError('Invalid observation or runtime termination; do not replace scene or continue automatically')
            evaluations[stage]={a:acceptance(pairs[a],policy) for a in plan['arms']}
            dump(out/(stage+'-evaluation.json'),evaluations[stage])
        decisions={}
        for arm in plan['arms']:
            reasons=[]
            if not selected[arm]['validation_improved']:reasons.append('validation_regression_blocks_promotion')
            for stage in ('diagnostic','final'):
                reasons.extend(stage+':'+r for r in evaluations[stage][arm]['reasons'])
            decisions[arm]=dict(accepted=not reasons,reasons=reasons,technically_completed=True)
            event('acceptance_decision',arm=arm,**decisions[arm])
        dump(out/'result.json',dict(selected=selected,evaluations=evaluations,decisions=decisions,episodes=episodes,
            optimizer_updates=10000,wall_seconds=time.monotonic()-started,gpu_worker_process_hours=process_seconds/3600,
            active_gpu_hours=None,energy_kWh=None,cost_note='Worker wall-hours include CPU/I/O; active GPU utilization and energy not measured'))
    except Exception as exc:
        event('blocked',error=str(exc),episodes=episodes)
        dump(out/'blocked.json',dict(error=str(exc),episodes=episodes,gpu_worker_process_hours=process_seconds/3600))
        raise
    finally:
        assert all(sha(ROOT/p)==digest for p,digest in preservation.items())
        dump(out/'preservation-after.json',dict(passed=True,weights=preservation))


if __name__=='__main__':main()
