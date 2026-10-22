"""Independent C step300 model/probe check. No optimizer, backward pass or simulator."""
import hashlib
import json
import os
from pathlib import Path
import time
from instruction_control_audit import ROOT, BASE, read, dump, sha

STUDY=ROOT/'outputs/agent-workflow-v1/instruction-control-v1/attempt02'
OUT=ROOT/'outputs/instruction-storage-audit/reload'
os.environ.update(HF_HOME=str(ROOT/'outputs/phase1a/hf-cache'),
                  HF_DATASETS_CACHE=str(ROOT/'outputs/ic-cache'),
                  HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    import numpy as np
    import torch
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.collate import lerobot_collate_fn
    checkpoint=STUDY/'C/checkpoints/000300/pretrained_model'
    before={str(p):sha(p) for p in checkpoint.iterdir() if p.is_file()}
    expected=read(STUDY/'C/losses-000300.json')
    pc=SmolVLAConfig.from_pretrained(str(checkpoint))
    data=ROOT/'outputs/phase1a/hf-cache/lerobot/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots/b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4'
    cfg=TrainPipelineConfig(dataset=DatasetConfig(repo_id='pepijn223/robocasa_CloseFridge',
        root=str(data),revision=data.name,video_backend='pyav',use_imagenet_stats=False,eval_split=.1),
        policy=pc,output_dir=OUT,seed=42,batch_size=1,num_workers=0,steps=300)
    torch.manual_seed(42)
    np.random.seed(42)
    train,val=make_train_eval_datasets(cfg)
    split=read(STUDY/'C/split.json')
    assert train.episodes==split['train_episodes'] and val.episodes==split['validation_episodes']
    print('Loading saved step300 model, no optimizer.',flush=True)
    policy=SmolVLAPolicy.from_pretrained(str(checkpoint),strict=True)
    policy.eval()
    assert policy.model.vlm_with_expert.config._commit_hash=='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
    actual={n:hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            for n,p in policy.state_dict().items()}
    assert actual==read(STUDY/'serialized-C-tensor-hashes.json')
    probes={'validation':[val[i] for i in split['validation_sample_indices']],
            'fixed_training_probe':[train[i] for i in split['training_sample_indices'][:8]]}
    results={}
    for label,source in [('saved_processors',checkpoint),('original_processors',BASE)]:
        pre,post=make_pre_post_processors(policy.config,pretrained_path=str(source))
        result={}
        with torch.no_grad(),torch.random.fork_rng(devices=[0]):
            for part,raws in probes.items():
                values=[]
                for i,raw in enumerate(raws):
                    torch.manual_seed(42000+i)
                    batch=lerobot_collate_fn([dict(raw,task='CloseFridge')])
                    for cam in train.meta.camera_keys:
                        if batch[cam].dtype==torch.uint8:
                            batch[cam]=batch[cam].float()/255
                    batch=pre(batch)
                    loss,_=policy.forward(batch)
                    assert torch.isfinite(loss)
                    values.append(loss.item())
                target=expected[part]['samples']
                result[part]=dict(mean=float(np.mean(values)),samples=values,
                    exact=values==target,passed=bool(np.allclose(values,target,rtol=1e-5,atol=1e-6)),
                    max_absolute_difference=float(np.max(np.abs(np.array(values)-target))))
        results[label]=result
        print(json.dumps({label:result}),flush=True)
    assert all(sha(Path(p))==h for p,h in before.items())
    result=dict(passed=all(r['passed'] for group in results.values() for r in group.values()),
        model_sha256=sha(checkpoint/'model.safetensors'),loaded_tensors=len(actual),
        probes=results,checkpoint_unchanged=True,optimizer_created=False,updates=0,
        simulator_episodes=0,seconds=time.monotonic()-started,
        note='Inference-only model integrity; missing optimizer/scheduler remains unrecovered.')
    dump(OUT/'result.json',result)
    assert result['passed']

if __name__=='__main__':
    main()
