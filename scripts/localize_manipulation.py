"""Fixed offline CloseFridge localization: saved RGB, traces and seed42 kinematics.

No policy, environment construction, dynamics integration, training or network.
Outputs are exclusive. Missing contact/model/frame information stays unavailable.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
FONT = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def plain(x):
    if isinstance(x, (np.ndarray, np.generic)):
        return x.tolist()
    if isinstance(x, dict):
        return {k: plain(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def path_for(seed, h):
    if h == 20:
        return ROOT/f'outputs/control-horizon-v1/h20-seed{seed}'
    if seed == 42:
        return ROOT/'outputs/phase1d/language-control-native'
    return ROOT/f'outputs/probes/instruction-distribution-v1/approved-retry-1/task-id-seed{seed}'


def loaded_steps(contacts):
    return {r['step'] for r in contacts for s in r['samples'] for c in s['robot_contacts']
            if c['constraint_address'] >= 0 and c['force_N_contact_frame'][0] > 1e-8}


def black_views(frame):
    """Saved rollout layout: 32-pixel title, three 256x256 camera images."""
    a=np.asarray(frame)
    assert a.shape==(288,768,3)
    return [bool(np.max(a[32:,i*256:(i+1)*256])==0) for i in range(3)]


def motion_stats(logs, reset, selected):
    before = [reset['physical']]+[r['physical'] for r in logs[:-1]]
    ix = [i for i,r in enumerate(logs) if r['step'] in selected and r['action'][4] < .5]
    if not ix:
        return None
    command = np.array([np.clip(logs[i]['action'][5:8],-1,1)*.05 for i in ix])
    achieved = np.array([np.subtract(logs[i]['physical']['robot0_base_to_eef_pos'], before[i]['robot0_base_to_eef_pos']) for i in ix])
    return {'actions':len(ix), 'mean_requested_delta_norm_m':np.linalg.norm(command,axis=1).mean(),
            'mean_achieved_relative_displacement_m':np.linalg.norm(achieved,axis=1).mean(),
            'mean_delta_minus_displacement_norm_m':np.linalg.norm(command-achieved,axis=1).mean(),
            'opposed_motion_fraction':np.mean(np.einsum('ij,ij->i',command,achieved)<0)}


def analyze(seed,h):
    p=path_for(seed,h); logs=rows(p/'steps.jsonl'); reset=read(p/'reset.json')
    contacts=rows(p/'contacts.jsonl') if (p/'contacts.jsonl').exists() else []
    loaded=loaded_steps(contacts)
    names=list(reset['fridge']['qpos'])
    angles=np.rad2deg([[reset['fridge']['qpos'][n] for n in names]]+[[r['fridge']['qpos'][n] for n in names] for r in logs])
    # Minimum summed absolute joint opening, not a substitute for the task predicate.
    best=int(np.argmin(np.abs(angles).sum(axis=1)))
    events={'reset':0, 'approach_1s':min(20,len(logs)), 'first_loaded_contact':min(loaded) if loaded else None,
            'max_closure_progress':best, 'final':len(logs)}
    contact_summary=None
    if contacts:
        catalog=read(p/'geometry-catalog.json'); by_pair={}
        for r in contacts:
            for s in r['samples']:
                for c in s['robot_contacts']:
                    if c['constraint_address']<0 or c['force_N_contact_frame'][0]<=1e-8:
                        continue
                    key=' / '.join(catalog[g]['name'] for g in c['geom_ids'])
                    item=by_pair.setdefault(key,{'first':r['step'],'last':r['step'],'peak_normal_N':0.,'loaded_samples':0,
                                                'ancestors':[catalog[g]['body_ancestors'] for g in c['geom_ids']]})
                    item['last']=r['step']; item['loaded_samples']+=1
                    item['peak_normal_N']=max(item['peak_normal_N'],c['force_N_contact_frame'][0])
        contact_summary={'loaded_actions':len(loaded),'pairs':by_pair,
                         'net_door_change_on_loaded_actions_deg':np.diff(angles,axis=0)[[r['step'] in loaded for r in logs]].sum(axis=0)}
    result={'seed':seed,'horizon':h,'events':events,'door_names':names,'initial_deg':angles[0],
            'best_deg':angles[best],'final_deg':angles[-1],'success':logs[-1]['fridge']['success'],
            'contacts':contact_summary,
            'motion_loaded':motion_stats(logs,reset,loaded) if contacts else None,
            'motion_other':motion_stats(logs,reset,set(range(1,len(logs)+1))-loaded),
            'event_angles':{k:angles[v] if v is not None else None for k,v in events.items()}}
    # Validate commanded delta reconstruction at saved controller snapshots only.
    snapshots=[]
    for name in ('first-action-trace.json','unstable-step-trace.json'):
        if (p/name).exists():
            t=read(p/name); target=t['targets']['right']; a=np.array(t['raw_action'])
            if target['_goal_update_mode']=='achieved':
                origin=np.array(target['origin_ori']); pos=np.array(target['origin_pos'])
                current=origin.T@(np.array(target['ref_pos'])-pos)
                err=float(np.max(np.abs(np.array(target['goal_pos'])-current-np.clip(a[5:8],-1,1)*.05)))
                assert err<1e-6
                snapshots.append({'step':t['step'],'goal_reconstruction_max_error_m':err})
    result['saved_target_checks']=snapshots
    return result,logs,reset


def seed42_geometry(out, all_logs):
    import mujoco
    from phase1c_replay import load_bundle, apply_state, proximity
    def reject(*args,**kwargs):
        raise AssertionError('Dynamics integration prohibited')
    mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=reject
    model,spec=load_bundle(ROOT/'outputs/phase1c/retry-20261018/replay-ready')
    data=mujoco.MjData(model); result={}
    for h in (50,20):
        logs,reset=all_logs[(42,h)]
        old_reset=read(ROOT/'outputs/phase1c/retry-20261018/reset.json')
        assert reset['physical']['qpos']==old_reset['physical']['qpos']
        assert reset['physical']['qvel']==old_reset['physical']['qvel']
        states=[reset['physical']]+[r['physical'] for r in logs]
        records=[]; errors=[]
        for i,state in enumerate(states):
            apply_state(model,data,state)
            eef=data.site_xpos[spec['eef_site']].copy()
            errors.append(np.linalg.norm(eef-state['robot0_eef_pos']))
            doors={}
            for name,d in spec['doors'].items():
                bid,jid=d['body_id'],d['joint_id']; rot=data.xmat[bid].reshape(3,3)
                hinge=data.xanchor[jid].copy(); axis=data.xaxis[jid].copy(); delta=eef-hinge
                # Door-body axes rotate with the door; values are not camera coordinates.
                doors[name]={'eef_in_moving_door_frame_m':rot.T@(eef-data.xpos[bid]),
                             'hinge_world_m':hinge,'hinge_axis_world':axis,
                             'eef_hinge_radial_m':np.linalg.norm(delta-axis*np.dot(delta,axis)),
                             'eef_hinge_axial_m':np.dot(delta,axis)}
                angle=data.qpos[d['qpos_address']]
                expected=reset['fridge']['qpos'][name] if i==0 else logs[i-1]['fridge']['qpos'][name]
                assert angle==expected
            records.append({'step':i,'doors':doors,'clearance':proximity(model,data,spec)})
        assert max(errors)<.005, 'Replay geometry exceeds declared 5 mm observation-cache tolerance'
        result[str(h)]={'max_eef_replay_vs_observation_m':max(errors),'records':records}
    (out/'seed42-geometry.json').write_text(json.dumps(plain(result),indent=2))
    return {h:{'max_eef_replay_vs_observation_m':v['max_eef_replay_vs_observation_m']} for h,v in result.items()}


def tangent_analysis(out,all_logs):
    geometry=read(out/'seed42-geometry.json'); result={}
    for h in (50,20):
        logs,reset=all_logs[(42,h)]; states=[reset['physical']]+[r['physical'] for r in logs]
        for name in reset['fridge']['qpos']:
            values=[]
            for i,r in enumerate(logs):
                geo=geometry[str(h)]['records'][i]; door=geo['doors'][name]
                radius=np.array(states[i]['robot0_eef_pos'])-door['hinge_world_m']
                tangent=-np.cross(door['hinge_axis_world'],radius); tangent/=np.linalg.norm(tangent)
                request=Rotation.from_quat(states[i]['robot0_base_quat']).apply(np.clip(r['action'][5:8],-1,1)*.05)
                achieved=np.array(states[i+1]['robot0_eef_pos'])-states[i]['robot0_eef_pos']
                if geo['clearance'][name]['gripper_to_door_signed_clearance_m']<.01 and geo['clearance'][name]['angle_deg']>5 and r['gate']['mode']=='achieved':
                    values.append({'step':i+1,'requested_closing_tangent_m':float(np.dot(request,tangent)),
                                   'achieved_closing_tangent_m':float(np.dot(achieved,tangent))})
            result[f'{h}/{name}']={'near_open_door_actions':len(values),'values':values}
    (out/'tangent-analysis.json').write_text(json.dumps(result,indent=2))


def replay_images(out):
    """Explicitly reconstructed seed42 views; never substitute them for policy inputs."""
    import mujoco
    from phase1c_replay import load_bundle,apply_state
    def reject(*args,**kwargs):
        raise AssertionError('Dynamics integration prohibited')
    mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=reject
    model,spec=load_bundle(ROOT/'outputs/phase1c/retry-20261018/replay-ready')
    data=mujoco.MjData(model); p=path_for(42,20)
    states=[read(p/'reset.json')['physical']]+[r['physical'] for r in rows(p/'steps.jsonl')]
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    apply_state(model,data,states[0])
    cam.lookat[:]=(data.site_xpos[spec['eef_site']]+data.xpos[next(iter(spec['doors'].values()))['body_id']])/2
    cam.distance=1.8; cam.azimuth=0; cam.elevation=-20
    opt=mujoco.MjvOption(); opt.geomgroup[0]=False;opt.geomgroup[2]=False
    canvas=Image.new('RGB',(1280,776),'white')
    with mujoco.Renderer(model,height=360,width=640) as renderer:
        for j,index in enumerate((0,45,281,285)):
            apply_state(model,data,states[index]); renderer.update_scene(data,camera=cam,scene_option=opt)
            pixels=renderer.render(); assert np.ptp(pixels)>0
            im=tile(Image.fromarray(pixels),f'DISPLAY-ONLY reconstructed seed42 H20 #{index}',640)
            canvas.paste(im,((j%2)*640,(j//2)*388))
    canvas.save(out/'seed42-display-replay.jpg',quality=95)


def decode(path):
    with av.open(str(path)) as c:
        return [f.to_image().convert('RGB') for f in c.decode(video=0)]


def tile(frame,label,width=768):
    size=(width,round(frame.height*width/frame.width))
    image=Image.new('RGB',(width,size[1]+28),'white')
    image.paste(frame.resize(size),(0,28)); ImageDraw.Draw(image).text((5,3),label,font=FONT,fill='black')
    return image


def visuals(out,results):
    decoded={}; integrity=[]
    for r in results:
        key=(r['seed'],r['horizon']); frames=decode(path_for(*key)/'rollout.mp4')
        assert len(frames)==r['events']['final']+1
        decoded[key]=frames
        flags=[black_views(frame) for frame in frames]
        integrity.append({'seed':r['seed'],'horizon':r['horizon'],
                          'all_black_frames':[i for i,x in enumerate(flags) if all(x)],
                          'any_black_camera_frames':[i for i,x in enumerate(flags) if any(x)]})
        black=integrity[-1]['all_black_frames']
        if black:
            index=black[0]; canvas=Image.new('RGB',(1536,316),'white')
            canvas.paste(tile(frames[index-1],f'ORIGINAL last visible: seed {r["seed"]} H{r["horizon"]} #{index-1}'),(0,0))
            canvas.paste(tile(frames[index],f'ORIGINAL first black: seed {r["seed"]} H{r["horizon"]} #{index}'),(768,0))
            canvas.save(out/f'blackout-seed{r["seed"]}-H{r["horizon"]}.jpg',quality=95)
    (out/'rgb-integrity.json').write_text(json.dumps(integrity,indent=2))
    # Same simulation-time index for all six panels; terminal panels explicitly freeze.
    with av.open(str(out/'synchronized.mp4'),'w') as video:
        stream=video.add_stream('libx264',rate=20); stream.width=1536; stream.height=948; stream.pix_fmt='yuv420p'
        for i in range(601):
            canvas=Image.new('RGB',(1536,948),'white')
            for row,seed in enumerate((42,43,44)):
                for col,h in enumerate((50,20)):
                    frames=decoded[(seed,h)]; index=min(i,len(frames)-1)
                    label=f'Seed {seed} H={h} | t={index*.05:.2f}s action={index}'+(' | ENDED / FROZEN' if i>=len(frames) else '')
                    if all(black_views(frames[index])): label+=' | RGB BLACK'
                    canvas.paste(tile(frames[index],label),(col*768,row*316))
            for packet in stream.encode(av.VideoFrame.from_image(canvas)): video.mux(packet)
        for packet in stream.encode(): video.mux(packet)
    for seed in (42,43,44):
        canvas=Image.new('RGB',(1920,344),'white')
        for row,h in enumerate((50,20)):
            r=next(r for r in results if (r['seed'],r['horizon'])==(seed,h))
            for col,(event,index) in enumerate(r['events'].items()):
                if index is None:
                    im=Image.new('RGB',(384,172),'#ddd'); ImageDraw.Draw(im).text((8,25),'H50 contact timing unavailable',font=FONT,fill='black')
                else:
                    label={'reset':'reset','approach_1s':'approach','first_loaded_contact':'loaded contact',
                           'max_closure_progress':'best opening','final':'final'}[event]
                    im=tile(decoded[(seed,h)][index],f'H{h} {label} #{index} ({index*.05:.2f}s)',384)
                canvas.paste(im,(col*384,row*172))
        canvas.save(out/f'seed{seed}-events.jpg',quality=95)
    demos=ROOT/'outputs/phase1c/demo-analysis'
    canvas=Image.new('RGB',(1920,516),'white')
    for row,(episode,indices) in enumerate([(0,[0,100,200,320,428]),(437,[0,150,280,400,483]),(5,[0,80,160,240,299])]):
        frames=decode(demos/f'demo-{episode:03d}-three-camera.mp4')
        for col,index in enumerate(indices):
            canvas.paste(tile(frames[index],f'Demo {episode} frame {index} ({index*.05:.2f}s)',384),(col*384,row*172))
    canvas.save(out/'demo-phases.jpg',quality=95)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/manipulation-localization-v1')
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    files=[p for seed in (42,43,44) for h in (50,20) for p in path_for(seed,h).iterdir() if p.is_file()]
    files+=list((ROOT/'outputs/phase1c/demo-analysis').glob('*'))
    inventory={str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()}
    (out/'preservation-before.json').write_text(json.dumps(inventory,indent=2))
    results=[]; logs={}
    for seed in (42,43,44):
        for h in (50,20):
            result,steps,reset=analyze(seed,h); results.append(result); logs[(seed,h)]=(steps,reset)
    geometry=seed42_geometry(out,logs)
    tangent_analysis(out,logs)
    replay_images(out)
    (out/'measurements.json').write_text(json.dumps(plain({'episodes':results,'seed42_geometry_validation':geometry}),indent=2))
    visuals(out,results)
    assert all(sha(ROOT/name)==digest for name,digest in inventory.items())
    (out/'verification.json').write_text(json.dumps({'historical_files_unchanged':len(inventory),'no_policy_or_dynamics':True,
                                                    'source_sha256':sha(Path(__file__)),'geometry_validation':geometry},indent=2))
    print(json.dumps(plain({'geometry':geometry,'episodes':results}),indent=2))


if __name__=='__main__':
    main()
