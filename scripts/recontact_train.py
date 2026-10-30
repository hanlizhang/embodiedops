"""One fixed 300-update mixture, using the historical native training primitives.

No simulator is imported. Independent verify mode loads step300 without updates.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import time
import json
import traceback
import numpy as np
from instruction_control_audit import ROOT, BASE, PROCESSORS, read, dump, sha
from recontact_self_imitation import OUT, STUDY, rows


def main(verify=False):
    plan=read(OUT/'registered-plan.json');index=read(OUT/'data-index.json')
    assert index['gate_passed'] and read(OUT/'collection-status.json')['status']=='gate_passed'
    assert time.time()<plan['deadline_unix']
    out=OUT/'training'
    if not verify:
        assert plan['deadline_unix']-time.time()>=plan['min_remaining_before_training_seconds']
        out.mkdir(exist_ok=False)
        dump(out/'started.json',dict(time=time.time(),data_index_sha256=sha(OUT/'data-index.json'),updates=300))
        (out/'executed-source.py.txt').write_bytes(Path(__file__).read_bytes())
    else:
        with (out/'verify-started.json').open('x') as f:json.dump(dict(time=time.time()),f)
    assert sha(OUT/'data-index.json')==read(out/'started.json')['data_index_sha256']
    assert all(sha(Path(p))==h for p,h in plan['source_hashes'].items())
    assert sha(BASE/'model.safetensors')==plan['model_hashes']['frozen']
    os.environ.update(HF_HOME=str(ROOT/'outputs/phase1a/hf-cache'),
        HF_DATASETS_CACHE=str(OUT/'ds'),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    from accelerate import Accelerator
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.common.train_utils import save_checkpoint
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.collate import lerobot_collate_fn
    from lerobot.utils.logging_utils import MetricsTracker,AverageMeter
    torch.manual_seed(42);np.random.seed(42)
    pc=SmolVLAConfig.from_pretrained(str(BASE))
    pc.freeze_vision_encoder=pc.train_expert_only=pc.train_state_proj=True
    pc.pretrained_path=str(BASE);pc.push_to_hub=False
    mapping=read(BASE/PROCESSORS[0])['steps'][0]['config']['rename_map']
    data=ROOT/'outputs/phase1a/hf-cache/lerobot/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots/b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4'
    cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='pepijn223/robocasa_CloseFridge',root=str(data),revision=data.name,
        video_backend='pyav',use_imagenet_stats=False,eval_split=.1),policy=pc,output_dir=out,seed=42,
        batch_size=1,num_workers=0,steps=300,env_eval_freq=0,eval_steps=50,max_eval_samples=8,
        log_freq=10,save_freq=150,rename_map=mapping,optimizer=pc.get_optimizer_preset(),scheduler=pc.get_scheduler_preset())
    if not verify:cfg.save_pretrained(out/'config')
    train,val=make_train_eval_datasets(cfg)
    split=read(STUDY/'C/split.json')
    assert train.episodes==split['train_episodes'] and val.episodes==split['validation_episodes']
    assert len(train)==split['train_frames'] and len(val)==split['validation_frames']
    indices=torch.randperm(len(train),generator=torch.Generator().manual_seed(42))[:300].tolist()
    assert indices==split['training_sample_indices']
    source=out/'checkpoints/000300/pretrained_model' if verify else BASE
    policy=SmolVLAPolicy.from_pretrained(str(source),config=None if verify else pc,strict=True)
    assert policy.config.n_obs_steps==1 and policy.config.chunk_size==50 and not policy.config.use_amp
    assert policy.model.vlm_with_expert.config._commit_hash=='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(source))
    intended={n:p for n,p in policy.named_parameters() if p.requires_grad}
    assert len(intended)==155 and sum(p.numel() for p in intended.values())==99880992
    old_parameters=read(STUDY/'C/parameters.json')['trainable_tensors']
    actual_parameters={n:dict(shape=list(p.shape),dtype=str(p.dtype)) for n,p in intended.items()}
    assert old_parameters==actual_parameters and not any('.vlm.' in n for n in intended)
    def prep(raw):
        batch=lerobot_collate_fn([dict(raw,task='CloseFridge')])
        for cam in train.meta.camera_keys:
            if batch[cam].dtype==torch.uint8:batch[cam]=batch[cam].float()/255.
        batch=pre(batch)
        assert batch['observation.state'].shape==(1,1,16)
        assert batch['action'].shape==(1,50,12) and batch['action_is_pad'].shape==(1,50)
        assert (~batch['action_is_pad']).any()
        assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v,torch.Tensor))
        for cam in mapping.values():assert batch[cam].shape==(1,3,256,256) and 0<=batch[cam].min()<=batch[cam].max()<=1
        return batch
    def raw_recontact(sample):
        p=Path(sample['observation']);assert sha(p)==sample['observation_sha256']
        with np.load(p) as ob:
            raw={'observation.images.'+cam:torch.from_numpy(ob[cam].copy()).permute(2,0,1) for cam in ['robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand']}
            raw['observation.state']=torch.from_numpy(ob['state'].copy())[None]
        n=sample['valid_actions'];action=np.zeros((50,12),np.float32)
        executed=np.asarray(sample['actions'],np.float32)
        assert hashlib.sha256(executed.tobytes()).hexdigest()==sample['action_bytes_sha256']
        action[:n]=executed
        pad=np.asarray(sample['action_is_pad'],bool)
        assert np.array_equal(pad,np.arange(50)>=n) and 0<n<=50
        raw.update(action=torch.from_numpy(action),action_is_pad=torch.from_numpy(pad),task='CloseFridge',episode_index=torch.tensor(-1))
        return raw
    train_probe=[train[i] for i in indices[:8]];val_raw=[val[i] for i in split['validation_sample_indices']]
    def evaluate(raws):
        policy.eval();losses=[]
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            for i,raw in enumerate(raws):
                torch.manual_seed(42000+i);loss,_=policy.forward(prep(raw));assert torch.isfinite(loss)
                losses.append(loss.item())
        return dict(mean=float(np.mean(losses)),samples=losses,noise_seeds=list(range(42000,42000+len(raws))))
    def hashes():
        return {n:hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for n,p in policy.state_dict().items()}
    if verify:
        checkpoint=out/'checkpoints/000300';target=read(checkpoint/'verification-target.json')
        assert hashes()==target['state_hashes']
        validation=evaluate(val_raw)
        assert np.allclose(validation['samples'],target['validation']['samples'],rtol=1e-5,atol=1e-6)
        assert all(sha(source/n)==plan['processors'][n] for n in PROCESSORS)
        dump(out/'reload-verification.json',dict(passed=True,step=300,state_tensors_equal=500,validation=validation,updates=0,model_sha256=sha(source/'model.safetensors')))
        dump(OUT/'frozen-models.json',dict(time=time.time(),data_index_sha256=sha(OUT/'data-index.json'),
            paths={**plan['models'],'recontact_mix_300':str(source)},
            hashes={**plan['model_hashes'],'recontact_mix_300':sha(source/'model.safetensors')}))
        print('Step300 independently reloaded and verified; model/data frozen.',flush=True);return
    rng=np.random.default_rng(731042);slots=set(rng.permutation(300)[:75].tolist())
    groups={s['id']:s['samples'] for s in index['sources'] if s['qualified']}
    cycle=[];training_plan=[]
    for i in range(300):
        if i in slots:
            if not cycle:cycle=list(rng.permutation(sorted(groups)))
            source_id=cycle.pop(0);sample=groups[source_id][int(rng.integers(len(groups[source_id])))]
            training_plan.append(dict(step=i+1,kind='recontact',sample_id=sample['id'],source=source_id,valid_actions=sample['valid_actions']))
        else:training_plan.append(dict(step=i+1,kind='ordinary',index=indices[i]))
    assert sum(x['kind']=='recontact' for x in training_plan)==75
    dump(out/'training-plan.json',training_plan);dump(out/'split.json',split)
    samples={s['id']:s for s in index['samples']}
    # Verify all custom data fields/processor alignment before the first update.
    for sample in samples.values():
        raw=raw_recontact(sample);batch=prep(raw)
        assert int((~batch['action_is_pad']).sum())==sample['valid_actions']
        unnormalized=post(batch['action']).detach().cpu().numpy()[0,:sample['valid_actions']]
        assert np.allclose(unnormalized,sample['actions'],atol=2e-6,rtol=1e-5),'Action normalization roundtrip'
    accelerator=Accelerator(mixed_precision='no',step_scheduler_with_optimizer=False)
    optimizer,scheduler=make_optimizer_and_scheduler(cfg,policy)
    meters={k:AverageMeter(k,':.6f') for k in ['loss','grad_norm','lr','update_s','gpu_mem_gb']}
    tracker=MetricsTracker(1,len(train),len(train.episodes),meters,accelerator=accelerator)
    latest_grad={}
    def before_step(opt,args,kw):
        grads={n:p.grad for n,p in intended.items() if p.grad is not None}
        assert len(grads)==155 and all(torch.isfinite(g).all() for g in grads.values())
        assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
        latest_grad.update(finite=True,tensors_with_grad=len(grads))
    hook=optimizer.register_step_pre_hook(before_step)
    initial_hashes=hashes();frozen_names=[n for n,p in policy.named_parameters() if not p.requires_grad]
    dump(out/'manifest.json',dict(recipe=plan['training'],parameters=actual_parameters,base_model_sha256=plan['model_hashes']['frozen'],
        processor_sha256=plan['processors'],data_index_sha256=sha(OUT/'data-index.json'),
        action_mask='Native forward slices to original_action_dim12 before time mask, divides by valid time*12. Internal dims12..31 excluded.',
        sampling_plan_sha256=sha(out/'training-plan.json'),script_sha256=sha(Path(__file__))))
    dump(out/'losses-000000.json',dict(step=0,validation=evaluate(val_raw),fixed_training_probe=evaluate(train_probe)))
    started=time.monotonic();step=0;exposure=dict(ordinary_samples=0,recontact_samples=0,ordinary_actions=0,recontact_actions=0,by_source={})
    historical_updates=rows(STUDY/'C/updates.jsonl')
    try:
        with (out/'updates.jsonl').open('x') as log:
            for entry in training_plan:
                assert time.time()+7200<plan['deadline_unix'],'Evaluation/delivery time reserve'
                assert shutil.disk_usage(OUT).free>2*1024**3 and shutil.disk_usage(ROOT).free>2*1024**3
                raw=raw_recontact(samples[entry['sample_id']]) if entry['kind']=='recontact' else train[entry['index']]
                batch=prep(raw);n=int((~batch['action_is_pad']).sum());lr=optimizer.param_groups[0]['lr']
                assert lr==historical_updates[entry['step']-1]['lr_used'],'Ordinary-control LR mismatch'
                torch.cuda.synchronize();tick=time.monotonic()
                tracker,_=update_policy(tracker,policy,batch,optimizer,cfg.optimizer.grad_clip_norm,accelerator,scheduler)
                torch.cuda.synchronize();step=entry['step']
                assert np.isfinite(tracker.loss.val) and np.isfinite(tracker.grad_norm.val)
                log.write(json.dumps(dict(**entry,loss=tracker.loss.val,grad_norm=tracker.grad_norm.val,lr_used=lr,
                    seconds=time.monotonic()-tick,valid_actions=n,**latest_grad))+'\n');log.flush()
                exposure[entry['kind']+'_samples']+=1;exposure[entry['kind']+'_actions']+=n
                if entry['kind']=='recontact':
                    x=exposure['by_source'].setdefault(entry['source'],dict(samples=0,actions=0));x['samples']+=1;x['actions']+=n
                if step%50==0:
                    report=dict(step=step,validation=evaluate(val_raw),fixed_training_probe=evaluate(train_probe))
                    dump(out/f'losses-{step:06d}.json',report);dump(out/'exposure.json',exposure)
                    print('UPDATE',step,'loss',tracker.loss.val,'validation',report['validation']['mean'],flush=True)
        checkpoint=out/'checkpoints/000300';state_hashes=hashes()
        assert all(state_hashes[n]==initial_hashes[n] for n in frozen_names),'Frozen parameter changed'
        assert any(state_hashes[n]!=initial_hashes[n] for n in intended),'No parameter updated'
        validation=evaluate(val_raw)
        save_checkpoint(checkpoint,300,cfg,policy,optimizer,scheduler,pre,post,num_processes=1,batch_size=1)
        for n in PROCESSORS:shutil.copyfile(BASE/n,checkpoint/'pretrained_model'/n)
        dump(checkpoint/'verification-target.json',dict(step=300,state_hashes=state_hashes,validation=validation))
        dump(out/'complete.json',dict(updates=300,seconds=time.monotonic()-started,exposure=exposure,frozen_unchanged=True,
            model_sha256=sha(checkpoint/'pretrained_model/model.safetensors')))
    except Exception:
        dump(out/'incomplete.json',dict(updates=step,seconds=time.monotonic()-started,exception=traceback.format_exc()));raise
    finally:hook.remove()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');main(ap.parse_args().verify)
