"""Offline audit of three fixed seed43 recordings; no simulator or policy imports."""
import csv
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT / p for p in (
    'outputs/phase1d/evaluation/pretrained-seed43',
    'outputs/probes/instruction-distribution-v1/approved-retry-1/task-id-seed43',
    'outputs/control-horizon-v1/h20-seed43')]

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

def intervals(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [[int(a + 1), int(b)] for a, b in edges.reshape(-1, 2)]

def black(frame):
    a = np.asarray(frame)
    assert a.shape == (288, 768, 3)
    return [bool(a[32:, k*256:(k+1)*256].max() == 0) for k in range(3)]

def main():
    out = ROOT / 'outputs/seed43-interaction-v1'
    out.mkdir(exist_ok=False)
    stats, logs, sources = {}, [], {}
    for label, p in zip('ABC', PATHS):
        rs, reset, manifest = rows(p/'steps.jsonl'), read(p/'reset.json'), read(p/'manifest.json')
        logs.append(rs)
        a = np.array([r['action'] for r in rs])
        phys = [reset['physical']] + [r['physical'] for r in rs]
        eef = np.array([r['robot0_eef_pos'] for r in phys])
        quat = np.array([r['robot0_eef_quat'] for r in phys])
        quat /= np.linalg.norm(quat, axis=1)[:, None]
        gap = np.array([r['robot0_gripper_qpos'] for r in phys])
        gap = gap[:, 0] - gap[:, 1]
        door = np.rad2deg([next(iter(r['fridge']['qpos'].values())) for r in [reset]+rs])
        close = a[:, 11] >= .5
        assert len(rs) == 600 and [r['step'] for r in rs] == list(range(1, 601))
        assert all((r['gate']['gripper'] == 'close') == bool(c) for r,c in zip(rs,close))
        s = dict(path=str(p.relative_to(ROOT)), instruction=manifest.get('resolved_instruction'),
                 model=manifest.get('model'), revision=manifest.get('model_revision'),
                 model_sha256=manifest.get('model_sha256'), close_steps=int(close.sum()),
                 close_intervals=intervals(close), transitions=int(np.count_nonzero(np.diff(close))),
                 command_range=[float(a[:,11].min()),float(a[:,11].max())],
                 joint_separation_range_m=[float(gap.min()),float(gap.max())],
                 initial_final_door_deg=[float(door[0]),float(door[-1])],
                 best_door_deg=float(door.min()),best_step=int(door.argmin()),
                 success=bool(rs[-1]['fridge']['success']), segments=[])
        for lo, hi in [(1,76),(77,150),(151,250),(251,350),(351,450),(451,500),(501,600)]:
            s['segments'].append(dict(steps=[lo,hi],eef_start_m=eef[lo-1].tolist(),eef_end_m=eef[hi].tolist(),
                eef_path_m=float(np.linalg.norm(np.diff(eef[lo-1:hi+1],axis=0),axis=1).sum()),
                orientation_net_deg=float(np.rad2deg(2*np.arccos(np.clip(abs(quat[lo-1]@quat[hi]),0,1)))),
                door_closing_deg=float(door[lo-1]-door[hi]),gap_start_end_m=[float(gap[lo-1]),float(gap[hi])]))
        s['closure_measurements'] = [dict(steps=[lo,hi],gap_before_m=float(gap[lo-1]),
            gap_min_m=float(gap[lo:hi+1].min()),gap_end_m=float(gap[hi]),
            door_closing_deg=float(door[lo-1]-door[hi])) for lo,hi in intervals(close)]
        with (out/f'{label}-all-actions.csv').open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['step','elapsed_s','gripper_command','close','finger_q0','finger_q1','door_deg',
                'eef_world_x','eef_world_y','eef_world_z','eef_quat_x','eef_quat_y','eef_quat_z','eef_quat_w'])
            for i,r in enumerate(rs,1):
                w.writerow([i,i*.05,a[i-1,11],bool(close[i-1]),*phys[i]['robot0_gripper_qpos'],door[i],*eef[i],*quat[i]])
        for name in ['manifest.json','reset.json','scene.json','steps.jsonl','rollout.mp4','executed-script.py.txt']:
            with (p/name).open('rb') as f: sources[str((p/name).relative_to(ROOT))]=hashlib.file_digest(f,'sha256').hexdigest()
        stats[label]=s
    catalog={g['id']:g for g in read(PATHS[2]/'geometry-catalog.json')}
    pairs={}; loaded=set(); shrimp=[]
    for r in rows(PATHS[2]/'contacts.jsonl'):
        assert len(r['samples']) == 25
        for sample in r['samples']:
            for c in sample['robot_contacts']:
                names=[catalog[g]['name'] for g in c['geom_ids']]
                if any('door_obj' in str(catalog[g]) for g in c['geom_ids']): shrimp.append(r['step'])
                if c['constraint_address']<0 or c['force_N_contact_frame'][0]<=1e-8: continue
                loaded.add(r['step']); key=' / '.join(names)
                v=pairs.setdefault(key,dict(first=r['step'],last=r['step'],samples=0,peak_normal_N=0))
                v['last']=r['step'];v['samples']+=1;v['peak_normal_N']=max(v['peak_normal_N'],c['force_N_contact_frame'][0])
    stats['C']['contacts']=dict(loaded_steps=len(loaded),first=min(loaded),last=max(loaded),pairs=pairs,
        shrimp_contact_steps=sorted(set(shrimp)),shrimp_geometries=[g for g in catalog.values() if 'door_obj' in str(g)])
    for name in ['contacts.jsonl', 'geometry-catalog.json', 'execution-config.json']:
        p = PATHS[2]/name
        with p.open('rb') as f: sources[str(p.relative_to(ROOT))]=hashlib.file_digest(f,'sha256').hexdigest()
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    containers=[av.open(str(p/'rollout.mp4')) for p in PATHS]
    streams=[iter(c.decode(video=0)) for c in containers]
    encoder=av.open(str(out/'three-run-comparison.mp4'),'w');video=encoder.add_stream('libx264',rate=20)
    video.width=768;video.height=948;video.pix_fmt='yuv420p'
    black_steps={k:[[],[],[]] for k in 'ABC'}
    selected={0,76,150,250,350,450,486,600}
    selected.update(x for ab in stats['A']['close_intervals'] for x in ab)
    for step, frames in enumerate(zip(*streams)):
        canvas=Image.new('RGB',(768,948))
        for j,(label,frame) in enumerate(zip('ABC',frames)):
            im=frame.to_image();canvas.paste(im,(0,j*316+28))
            for k,b in enumerate(black(im)):
                if b:black_steps[label][k].append(step)
            desc=f'{label}: '+['NL H50','CloseFridge H50','CloseFridge H20'][j]+f' | step {step} | {step*.05:.2f}s'
            if step:desc+=f" | gripper {logs[j][step-1]['gate']['gripper']}"
            ImageDraw.Draw(canvas).text((5,j*316+3),desc,font=font,fill='white')
        if step in selected:canvas.save(out/f'comparison-step{step:03d}.jpg')
        for packet in video.encode(av.VideoFrame.from_image(canvas)):encoder.mux(packet)
    assert step==600
    for packet in video.encode():encoder.mux(packet)
    encoder.close()
    for c in containers:c.close()
    stats['black_video_frames_by_camera']=black_steps
    stats['video_frames']=step+1
    (out/'measurements.json').write_text(json.dumps(stats,indent=2)+'\n')
    (out/'source-sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps({k:{x:v for x,v in s.items() if x not in ['segments','contacts']} for k,s in stats.items() if k in 'ABC'},indent=2))

if __name__=='__main__':
    main()
