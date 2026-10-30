"""Summarize only executed recovery evidence; never execute a model/simulator."""
import csv
import json
from pathlib import Path
import numpy as np
from recovery_feasibility import output_root
from instruction_control_audit import ROOT,read,dump


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def metrics(folder,spec,control=False):
    start=spec['action']
    rows=lines(folder/('replay.jsonl' if control else 'steps.jsonl'))
    if control: rows=[r for r in rows if r['action_index']>start]
    initial=np.array(spec['opening_deg'])
    angles=np.array([list(r['fridge']['qpos'].values()) for r in rows])*180/np.pi
    catalog={g['id']:g for g in read(folder/'geometry-catalog.json')}
    source=ROOT/spec['source']
    fridge_names=set(read(source/'study-fixture.json')['fridge_contact_geoms'])
    contact_rows=[r for r in lines(folder/'contacts.jsonl') if r['step']>start]
    peaks=dict(all=0.,fridge=0.,self=0.); loaded=set(); self_actions=set()
    for r in contact_rows:
        for sample in r['samples']:
            for c in sample['robot_contacts']:
                if c['constraint_address']<0: continue
                force=max(0.,c['force_N_contact_frame'][0])
                geoms=[catalog[i] for i in c['geom_ids']]
                peaks['all']=max(peaks['all'],force)
                if all(g['robot'] for g in geoms):
                    peaks['self']=max(peaks['self'],force)
                    if force>=1: self_actions.add(r['step'])
                if any(g['name'] in fridge_names for g in geoms):
                    peaks['fridge']=max(peaks['fridge'],force)
                    if force>=1: loaded.add(r['step'])
    eef=np.array([spec['physical']['robot0_eef_pos']]+[r['physical']['robot0_eef_pos'] for r in rows])
    return dict(actions=len(rows),success=bool(rows[-1]['fridge']['success']) if rows else False,
        initial_deg=initial.tolist(),final_deg=angles[-1].tolist() if len(rows) else initial.tolist(),
        progress_deg=(initial-angles[-1]).tolist() if len(rows) else [0.]*len(initial),
        best_progress_deg=(initial-np.minimum(initial,angles.min(axis=0))).tolist() if len(rows) else [0.]*len(initial),
        peak_normal_N=peaks['all'],peak_fridge_N=peaks['fridge'],peak_self_N=peaks['self'],
        loaded_fridge_actions=len(loaded),self_contact_actions=len(self_actions),
        eef_path_m=float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum()))


def main():
    out=output_root(); plan=read(out/'registered-plan.json'); audit=read(out/'audit-summary.json')
    status=read(out/'branch-status.json'); results=read(out/'branch-results.json') if (out/'branch-results.json').exists() else []
    from recovery_geometry_audit import audit as geometry_audit
    dump(out/'geometry-endpoint-audit.json',geometry_audit())
    table=[]
    for spec in plan['states']:
        control=metrics(out/(spec['id']+'-restoration-audit'),spec,True)
        table.append(dict(state=spec['id'],branch='control',valid=True,**control))
        for branch in ['vla0','vla1','vla2','vla3','expert']:
            result=next((r for r in results if r['state_id']==spec['id'] and r['branch']==branch),None)
            if result is None:
                table.append(dict(state=spec['id'],branch=branch,valid=False,status='unrun')); continue
            folder=out/(spec['id']+'-'+branch)
            row=dict(state=spec['id'],branch=branch,valid=result['valid'],status=result['stop'],
                recorded_actions=result['actions'],worker_seconds=result['worker_seconds'],
                inference_seconds=result['inference_seconds'])
            if branch=='expert':
                row.update(execution_valid=result['valid'],valid=False,status='invalid_expert_endpoint_order')
            if (folder/'steps.jsonl').exists(): row.update(metrics(folder,spec))
            row['useful_progress']=bool(row.get('valid') and max(row.get('progress_deg',[0]))>=5 and row.get('peak_normal_N',float('inf'))<=1.25*control['peak_normal_N']+25)
            table.append(row)
    dump(out/'comparison.json',dict(rows=table,audit=audit,status=status))
    columns=['state','branch','valid','status','actions','success','initial_deg','final_deg','progress_deg',
             'best_progress_deg','peak_normal_N','peak_fridge_N','peak_self_N','loaded_fridge_actions',
             'self_contact_actions','eef_path_m','inference_seconds','worker_seconds','useful_progress']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(table)
    media=['<!doctype html><meta charset="utf-8"><title>Recovery feasibility: actual recordings</title>',
           '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}video{width:768px;max-width:100%}</style>',
           '<h1>Recovery feasibility: actual recordings</h1><p>No learned-policy improvement is established. Seed 43 stopped before recovery actions. The seed-48 expert has an invalid approach-direction implementation; its recording is retained as error evidence.</p>']
    for result in results:
        folder=result['state_id']+'-'+result['branch']
        if (out/folder/'rollout.mp4').exists():
            media.append(f'<h2>{folder}</h2><p>Success: {result["success"]}; actions: {result["actions"]}. '+('Expert implementation invalid.' if result['branch']=='expert' else 'Frozen pretrained policy.')+'</p>'+f'<video controls preload="metadata" poster="{folder}/before.png" src="{folder}/rollout.mp4"></video>')
    (out/'recordings.html').write_text('\n'.join(media),encoding='utf-8')
    print(json.dumps(dict(rows=table,status=status),indent=2))


if __name__=='__main__':main()
