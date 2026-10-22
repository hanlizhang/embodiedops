"""Summarize every registered cell, including missing/invalid attempts; no policy execution."""
import csv
import argparse
import html
import json
from pathlib import Path
from instruction_control_audit import ROOT, read, dump
from instruction_storage_contract import evaluation_root

def physical_phases(out,results):
    import numpy as np
    phases=[]
    for r in results:
        folder=ROOT/r['path']
        steps=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        states=[read(folder/'reset.json')]+steps
        joints=r['door_joints']
        angles=np.rad2deg([[s['fridge']['qpos'][j] for j in joints] for s in states])
        loaded=set(r['contacts']['fridge']['loaded_action_indices'])
        first=min(loaded) if loaded else len(steps)+1
        productive=[i for i in loaded if np.max(angles[i-1]-angles[i])>.05]
        eef=np.array([s['physical']['robot0_eef_pos'] for s in states])
        last=max(productive) if productive else None
        windows=[('approach_before_first_loaded_contact',1,min(first-1,len(steps))),
                 ('contact_and_subsequent_motion',first,len(steps)),
                 ('last_100_actions',max(1,len(steps)-99),len(steps))]
        for phase,lo,hi in windows:
            if lo>hi:continue
            rows=steps[lo-1:hi]
            phases.append(dict(seed=r['seed'],model=r['model'],condition=r['condition'],phase=phase,
                first_action=lo,last_action=hi,per_door_progress_deg=(angles[lo-1]-angles[hi]).tolist(),
                loaded_fridge_actions=sum(i in loaded for i in range(lo,hi+1)),
                loaded_actions_with_positive_door_motion=sum(lo<=i<=hi for i in productive),
                eef_path_m=float(np.linalg.norm(np.diff(eef[lo-1:hi+1],axis=0),axis=1).sum()),
                close_gripper_commands=sum(s['action'][11]>=.5 for s in rows),
                desired_goal_commands=sum(s['action'][4]>=.5 for s in rows),
                last_loaded_productive_action=last))
    dump(out/'physical-phase-windows.json',phases)
    return phases

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-directory',type=Path)
    args=ap.parse_args()
    evaluation=evaluation_root()
    out=args.results_directory or evaluation
    results=read(out/'results.json') if (out/'results.json').exists() else []
    by={(r['seed'],r['model'],r['condition']):r for r in results}
    ledger={(r['seed'],r['model'],r['condition']):r for r in read(out/'ledger.json')} if (out/'ledger.json').exists() else {}
    phases=physical_phases(out,results)
    columns=['seed','model','condition','door_count','status','success','actions','initial_angles_deg',
      'final_angles_deg','per_door_progress_deg','per_door_best_progress_deg','eef_path_m',
      'fridge_loaded_actions','fridge_peak_N','self_loaded_actions','self_peak_N','other_peak_N',
      'close_gripper_actions','desired_goal_actions','mean_inference_seconds','total_inference_seconds',
      'rollout_seconds','worker_seconds','artifact_directory']
    rows=[]
    for seed in range(42,52):
        for model in ['A','B','C']:
            for condition in ['ID','NL']:
                r=by.get((seed,model,condition))
                folder=ROOT/r['path'] if r else evaluation/f'{model}-{condition}-{seed}'
                row=dict(seed=seed,model=model,condition=condition,status='not_run',artifact_directory=str(folder))
                if r:
                    row.update({k:r.get(k) for k in columns if k in r})
                    row['status']=r['termination']
                    for cat in ['fridge','self','other']:
                        row[f'{cat}_peak_N']=r['contacts'][cat]['peak_N']
                        if cat!='other':row[f'{cat}_loaded_actions']=r['contacts'][cat]['loaded_actions']
                elif (folder/'run-started.json').exists():
                    row['status']='invalid_or_incomplete_attempt'
                    if (folder/'steps.jsonl').exists():
                        row['actions']=len((folder/'steps.jsonl').read_text().splitlines())
                    if (folder/'summary.json').exists():
                        s=read(folder/'summary.json')
                        row['status']=s['termination_reason']
                        row['actions']=s['simulator_steps']
                cell=ledger.get((seed,model,condition))
                if cell and cell['status']!='completed':row['status']=cell['status']
                rows.append(row)
    with (out/'all-60-cells.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    aggregates=[]
    for condition in ['ID','NL']:
        for model in ['A','B','C']:
            selected=[r for r in results if r['condition']==condition and r['model']==model]
            valid=[r for r in selected if r['valid']]
            aggregates.append(dict(model=model,condition=condition,recorded=len(selected),valid=len(valid),
                successes=sum(r['success'] for r in valid),
                single=dict(successes=sum(r['success'] for r in valid if r['door_count']==1),count=sum(r['door_count']==1 for r in valid)),
                double=dict(successes=sum(r['success'] for r in valid if r['door_count']==2),count=sum(r['door_count']==2 for r in valid)),
                worker_seconds=sum(r['worker_seconds'] for r in selected),
                rollout_seconds=sum(r['rollout_seconds'] for r in selected),
                inference_seconds=sum(r['total_inference_seconds'] for r in selected)))
    comparisons=[]
    video_pairs=[]
    for condition in ['ID','NL']:
        for model in ['B','C']:
            for seed in range(42,52):
                a,b=by.get((seed,'A',condition)),by.get((seed,model,condition))
                if a is None or b is None:continue
                comparisons.append(dict(seed=seed,model=model,condition=condition,
                    matched_valid=a['valid'] and b['valid'],success_change=int(b['success'])-int(a['success']),
                    door_progress_delta_deg=b['closing_deg']-a['closing_deg'],
                    force_regression=b['peak_contact_normal_N']>1.25*a['peak_contact_normal_N']+25,
                    self_peak_delta_N=b['contacts']['self']['peak_N']-a['contacts']['self']['peak_N']))
                if b['valid'] and b['success'] and a['valid'] and not a['success']:
                    video_pairs.append(dict(seed=seed,model=model,condition=condition,
                        before=str((ROOT/a['path']/'rollout.mp4').relative_to(evaluation)),after=str((ROOT/b['path']/'rollout.mp4').relative_to(evaluation))))
    complete=(out/'complete.json').exists() and len(results)==60
    summary=dict(complete=complete,aggregates=aggregates,comparisons=comparisons,new_success_video_pairs=video_pairs,
        physical_phase_windows=len(phases),
        stopped=read(out/'stopped.json') if (out/'stopped.json').exists() else None)
    dump(out/'comparison-summary.json',summary)
    lines=['# Instruction-controlled A/B/C robot study','',
      f'Status: {"complete 60-episode matrix" if complete else "INCOMPLETE — no full-matrix conclusion"}. '
      f'{len(results)} recorded episode metrics. All60 registered cells, including unrun/invalid cells, are in all-60-cells.csv.','',
      'A: pretrained; B: original300-update NL; C:300-update task-ID. C inference reload passed; '
      'optimizer/scheduler state is incomplete. No training replay. ID=CloseFridge; NL=native scene instruction.','',
      '| Model | Condition | Success / valid | Single-door | Double-door | Worker min | Inference s |',
      '|---|---|---:|---:|---:|---:|---:|']
    for a in aggregates:
        lines.append(f"| {a['model']} | {a['condition']} | {a['successes']}/{a['valid']} | "
          f"{a['single']['successes']}/{a['single']['count']} | {a['double']['successes']}/{a['double']['count']} | "
          f"{a['worker_seconds']/60:.2f} | {a['inference_seconds']:.2f} |")
    lines+=['','## All-scene outcomes','', '| Seed | A-ID | B-ID | C-ID | A-NL | B-NL | C-NL |','|---|---|---|---|---|---|---|']
    for seed in range(42,52):
        cells=[]
        for condition in ['ID','NL']:
            for model in ['A','B','C']:
                r=by.get((seed,model,condition))
                cells.append(ledger.get((seed,model,condition),{}).get('status','unrun/invalid') if not r else ('success' if r['success'] and r['valid'] else r['termination']))
        lines.append('| '+str(seed)+' | '+' | '.join(cells)+' |')
    lines+=['','Detailed CSV includes per-door final/best progress, EEF path, loaded fridge contact, '
      'self-contact, force peaks, gripper/goal commands and timings. Full EEF trajectories remain in steps.jsonl. '
      'Loaded counts use positive solver normal force at least1N sampled within an action; they do not establish grasping '
      'or continuous contact. Worker time includes loading; inference time is only the instrumented chunk calls. '
      'No energy or active GPU-kernel time was measured.','',
      'The current shared RNG/renderer protocol differs from historical Phase1D. Neither successful scenes '
      'nor action differences alone prove generalization or isolate a physical-learning mechanism. '
      'Known seed43 runtime sensitivity remains a limitation. No confirmation seeds were run.']
    if summary['stopped']:
        lines+=['','Stopped reason: `'+summary['stopped']['reason']+'`. No automatic retry.']
    if not complete:
        lines+=['','This is an incomplete descriptive comparison, not the final registered study result. '
          'Invalid and unrun cells are excluded from success-rate denominators.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    page=['<!doctype html><meta charset="utf-8"><title>Matched robot recordings</title>',
      '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto}video{width:48%}</style>',
      '<h1>Newly successful scenes: matched before / after</h1><p>See <a href="all-60-cells.csv">all60 registered cells</a> and <a href="report.md">report</a>. Each video is an actual recorded rollout.</p>']
    for p in video_pairs:
        prefix='../' if out!=evaluation else ''
        page.append(f"<h2>Seed {p['seed']}, {p['condition']}, A versus {p['model']}</h2>"+
          f'<video controls src="{prefix}{html.escape(p["before"])}"></video> <video controls src="{prefix}{html.escape(p["after"])}"></video>')
    if not video_pairs:page.append('<p>No verified newly successful pair is available.</p>')
    (out/'before-after.html').write_text('\n'.join(page),encoding='utf-8')
    print(json.dumps(dict(complete=complete,episodes=len(results),stopped=summary['stopped']),indent=2))

if __name__=='__main__':
    main()
