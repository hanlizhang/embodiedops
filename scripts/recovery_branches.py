"""Registered physical feasibility branches; run only after prefix replay gates."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from recovery_feasibility import output_root, max_error, camera_error
from instruction_control_audit import ROOT, BASE, read, dump, sha


def world_delta_to_public(delta, rotation, limit=.01):
    """Base-frame OSC increments; norm-bounded world translation in metres."""
    delta=np.asarray(delta,dtype=float)
    rotation=np.asarray(rotation,dtype=float)
    assert delta.shape==(3,) and rotation.shape==(3,3)
    assert np.isfinite(delta).all() and np.isfinite(rotation).all()
    assert np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
    delta=delta*min(1.,limit/max(float(np.linalg.norm(delta)),1e-12))
    action=np.zeros(12,dtype=np.float32)
    action[4]=action[11]=-1
    action[5:8]=rotation.T@delta/.05
    return action


class GeometryExpert:
    """One fixed nearest-surface approach then tangential push; no search."""
    def __init__(self,low):
        cache=os.environ.get('NUMBA_CACHE_DIR')
        from phase1c_replay import geometry_spec
        if cache is not None: os.environ['NUMBA_CACHE_DIR']=cache
        self.low=low
        self.spec=geometry_spec(low.sim.model._model,low.robots[0],low)
        assert len(self.spec['doors'])==1
        self.door=next(iter(self.spec['doors'].values()))

    def action(self):
        import mujoco
        low=self.low; m=low.sim.model._model; d=low.sim.data._data
        arm=low.robots[0].composite_controller.part_controllers['right']
        assert arm.input_ref_frame=='base' and arm.input_type=='delta'
        best=(2.,None,None)
        for hand in self.spec['gripper_geoms']:
            for door in self.door['geoms']:
                # Post-run repair; archived executed source retains the original
                # reversed-endpoint defect. No corrected robot attempt has run.
                from recovery_geometry_audit import ordered_geometry_distance
                distance,points=ordered_geometry_distance(m,d,hand,door,2.)
                if distance<best[0]: best=(float(distance),points.copy(),[hand,door])
        distance,points,pair=best
        assert points is not None and distance < .75, 'No reachable nearby door surface'
        jid=self.door['joint_id']
        angle=float(d.qpos[self.door['qpos_address']])
        radial=points[3:]-d.xanchor[jid]
        tangent=-np.sign(angle)*np.cross(d.xaxis[jid],radial)
        assert np.linalg.norm(tangent)>1e-6
        tangent/=np.linalg.norm(tangent)
        # Approach the nearest physical surface; after contact push along closure.
        delta=(points[3:]-points[:3]) if distance>.012 else tangent*.01
        action=world_delta_to_public(delta,arm.origin_ori)
        return action,dict(stage='approach' if distance>.012 else 'tangential_push',
            nearest_distance_m=distance,geom_ids=pair,nearest_points=points.tolist(),
            hinge_world=d.xanchor[jid].tolist(),axis_world=d.xaxis[jid].tolist(),
            closing_tangent_world=tangent.tolist(),angle_rad=angle)


def worker(state_id,branch):
    import phase1d_rollout as rollout
    from phase1b_contract import CAMERAS, physical, serial, sheet
    from control_diagnostics import ContactRecorder
    from instruction_disk_guard import sample
    from lerobot.envs.robocasa import RoboCasaEnv, convert_action
    from PIL import Image
    import renderer_lifecycle
    import av
    out=output_root(); plan=read(out/'registered-plan.json')
    assert read(out/'audit-summary.json')['restoration_gate_passed']
    spec=next(s for s in plan['states'] if s['id']==state_id)
    source=ROOT/spec['source']
    folder=out/(state_id+'-'+branch); folder.mkdir(exist_ok=False)
    rollout.OUT=folder
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba-cache')
    renderer_lifecycle.install()
    start=time.monotonic(); env=None; video=None
    result=dict(state_id=state_id,branch=branch,valid=False,actions=0,success=False,
        prefix_actions=0,inference_seconds=0.,simulation_seconds=0.,peak_normal_N=0.,
        peak_self_N=0.,peak_fridge_N=0.,stop='initializing')
    try:
        assert sha(BASE/'model.safetensors')==plan['checkpoint_sha256']
        class SingleEpisode(RoboCasaEnv):
            def step(self,action):
                raw,reward,done,truncated,info=self._env.step(convert_action(action))
                return self._format_raw_obs(raw),reward,done or bool(info.get('success',False)),truncated,info
        env=SingleEpisode('CloseFridge',camera_name=CAMERAS)
        np.random.seed(spec['seed']); env._ensure_env()
        np.random.seed(spec['seed']); obs,_=env.reset(seed=spec['seed'])
        low=env._env.env; reset=physical(low); old=read(source/'reset.json')
        assert all(max_error(reset[k],old['physical'][k])==0 for k in ('qpos','qvel'))
        assert max_error(obs['agent_pos'],old['state'])==0
        assert rollout.canonical_scene(serial(low.get_ep_meta()))==rollout.canonical_scene(read(source/'scene.json')['ep_meta'])
        assert serial(low.robots[0].composite_controller_config)==read(source/'scene.json')['controller']
        for cam in CAMERAS:
            e=camera_error(obs['pixels'][cam],np.asarray(Image.open(source/f'reset-{cam}.png')))
            assert e['max']<=1 and e['mean']<=.001
        rows=[json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()]
        recorder=ContactRecorder(low.sim,folder)
        def step(action,index):
            assert time.monotonic()-start<plan['worker_seconds']
            for p in [ROOT,out]: assert sample(p)['above_floor']
            rollout.action_gate(env,action)
            with recorder.step(index): observation,trace=rollout.traced_step(env,action,index)
            if index>spec['action']: result['actions']+=1
            assert not trace['guard_failures'], str(trace['guard_failures'])
            for cam in CAMERAS: camera_error(observation['pixels'][cam],observation['pixels'][cam])
            peak=max((c['force_N_contact_frame'][0] for s in recorder.last_samples for c in s['robot_contacts']),default=0)
            assert peak<=plan['hard_force_ceiling_N'],'Hard force ceiling'
            return observation,trace
        for row in rows[:spec['action']]:
            obs,trace=step(np.asarray(row['action'],dtype=np.float32),row['step'])
            result['prefix_actions']+=1
            for k in ('qpos','qvel'):
                assert max_error(trace['after'][k],row['physical'][k])<=plan['tolerances'][k], 'Branch prefix divergence'
            assert max_error(obs['agent_pos'],row['returned_state'])<=plan['tolerances']['state']
            assert not trace['terminated'] and not trace['truncated']
        reference=np.load(out/(state_id+'-restoration-audit')/'reconstructed-observation.npz')
        errors={cam:camera_error(obs['pixels'][cam],reference[cam]) for cam in CAMERAS}
        result['branch_observation_camera_errors']=errors
        assert all(e['max']<=1 and e['mean']<=.001 for e in errors.values()),'Branch camera mismatch'
        assert max_error(obs['agent_pos'],reference['state'])<=1e-7
        np.savez_compressed(folder/'observation.npz',state=obs['agent_pos'],**obs['pixels'])
        result['initial_fridge']=rollout.fridge(low)
        sheet(obs,'BEFORE '+state_id).save(folder/'before.png')
        if branch=='expert':
            expert=GeometryExpert(low)
            dump(folder/'geometry-spec.json',expert.spec)
        else:
            import torch
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.envs.utils import preprocess_observation
            import hashlib
            policy=SmolVLAPolicy.from_pretrained(str(BASE)); policy.eval(); policy.reset()
            assert policy.config.chunk_size==policy.config.n_action_steps==50
            assert policy.config.num_steps==10
            pre,post=make_pre_post_processors(policy.config,pretrained_path=str(BASE))
            # Every candidate sees identical saved RGB/state, independently verified
            # against this branch's rendered observation (sparse raster tolerance).
            shared={'agent_pos':reference['state'].copy(),'pixels':{c:reference[c].copy() for c in CAMERAS}}
            batch=pre(dict(preprocess_observation(shared),task=['CloseFridge']))
            assert batch['observation.state'].shape==(1,16)
            assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v,torch.Tensor))
            seed=plan['candidate_noise_seeds'][int(branch[-1])]
            torch.manual_seed(seed)
            device=next(policy.parameters()).device
            noise=policy.model.sample_noise((1,50,32),device)
            torch.save(noise.cpu(),folder/'flow-noise.pt')
            result['noise_seed']=seed
            result['noise_sha256']=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            torch.cuda.synchronize(); t=time.monotonic()
            with torch.inference_mode(): chunk=post(policy.predict_action_chunk(batch,noise=noise)).detach().cpu().numpy()[0]
            torch.cuda.synchronize(); result['inference_seconds']=time.monotonic()-t
            assert chunk.shape==(50,12) and np.isfinite(chunk).all()
            np.save(folder/'candidate-actions.npy',chunk)
            result['peak_cuda_allocated']=torch.cuda.max_memory_allocated()
        video=av.open(str(folder/'rollout.mp4'),mode='w')
        stream=video.add_stream('libx264',rate=20); stream.width=768; stream.height=288; stream.pix_fmt='yuv420p'
        fridge_ids={g['id'] for g in recorder.catalog if g['name'] in set(low.fxtr.contact_geoms)}
        result['stop']='horizon'
        with (folder/'steps.jsonl').open('x') as log:
            for i in range(plan['horizon']):
                action,decision=expert.action() if branch=='expert' else (chunk[i],None)
                t=time.monotonic(); obs,trace=step(action,spec['action']+i+1)
                result['simulation_seconds']+=time.monotonic()-t
                for s in recorder.last_samples:
                    for c in s['robot_contacts']:
                        if c['constraint_address']<0: continue
                        force=max(0.,c['force_N_contact_frame'][0]); ids=set(c['geom_ids'])
                        result['peak_normal_N']=max(result['peak_normal_N'],force)
                        if ids.issubset(recorder.robot_ids): result['peak_self_N']=max(result['peak_self_N'],force)
                        if ids&fridge_ids: result['peak_fridge_N']=max(result['peak_fridge_N'],force)
                log.write(json.dumps(serial(dict(index=i+1,action=action,decision=decision,
                    physical=trace['after'],fridge=trace['fridge_after'],guards=trace['guard_failures'])))+'\n'); log.flush()
                for packet in stream.encode(av.VideoFrame.from_image(sheet(obs,f'{branch} ACTION {i+1}'))): video.mux(packet)
                if trace['fridge_after']['success'] or trace['terminated'] or trace['truncated']:
                    result['stop']='success' if trace['fridge_after']['success'] else 'simulator_termination'; break
        result['valid']=True
    except Exception as exc:
        import traceback
        result.update(stop='guard_or_exception',exception=traceback.format_exc(),reason=str(exc))
    finally:
        if video is not None:
            for packet in stream.encode(): video.mux(packet)
            video.close()
        if env is not None:
            if 'initial_fridge' in result:
                result['final_fridge']=rollout.fridge(low)
                result['success']=result['final_fridge']['success']
                sheet(obs,'AFTER '+branch).save(folder/'after.png')
            env.close()
        result['worker_seconds']=time.monotonic()-start
        dump(folder/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['valid'] else 2


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--state',required=True)
    ap.add_argument('--branch',required=True,choices=['vla0','vla1','vla2','vla3','expert'])
    args=ap.parse_args(); sys.exit(worker(args.state,args.branch))
