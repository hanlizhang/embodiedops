"""Inspect completed updates and partially serialized step300; no training or inference."""
import argparse
import json
from pathlib import Path
from instruction_control_audit import BASE, HISTORY, PROCESSORS, ROOT, read, dump, sha

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--study',type=Path,required=True)
    a=ap.parse_args()
    out=a.study.resolve()
    c=out/'C'
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    original,current=rows(HISTORY/'updates.jsonl'),rows(c/'updates.jsonl')
    assert len(current)==300
    for old,new in zip(original,current):
        assert all(old[k]==new[k] for k in ['step','sample_index','episode','lr_used'])
        assert new['finite'] and new['missing']==[]
    checkpoint=c/'checkpoints/000300'
    intended=set(read(c/'parameters.json')['trainable_tensors'])
    changed=[]
    hashes={}
    with safe_open(BASE/'model.safetensors',framework='pt',device='cpu') as b, safe_open(checkpoint/'pretrained_model/model.safetensors',framework='pt',device='cpu') as f:
        assert set(b.keys())==set(f.keys())
        for n in f.keys():
            x,y=b.get_tensor(n),f.get_tensor(n)
            assert x.shape==y.shape and x.dtype==y.dtype and torch.isfinite(y).all()
            if not torch.equal(x,y):
                changed.append(n)
            import hashlib
            hashes[n]=hashlib.sha256(y.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert set(changed)<=intended
    rng=[]
    for step in [1,150,300]:
        x=load_file(HISTORY/f'checkpoints/{step:06d}/training_state/rng_state.safetensors')
        y=load_file(c/f'checkpoints/{step:06d}/training_state/rng_state.safetensors')
        rng.append(dict(step=step,equality={k:torch.equal(x[k],y[k]) for k in x if k.startswith('torch')}))
    processors={}
    for n in PROCESSORS:
        path=checkpoint/'pretrained_model'/n
        item=dict(byte_identical=sha(BASE/n)==sha(path))
        if n.endswith('safetensors'):
            x,y=load_file(BASE/n),load_file(path)
            item['tensor_values_equal']=set(x)==set(y) and all(torch.equal(x[k],y[k]) for k in x)
        else:
            item['json_equal']=read(BASE/n)==read(path)
        processors[n]=item
    losses={str(step):read(c/('initial-losses.json' if step==0 else f'losses-{step:06d}.json')) for step in [0,150,300]}
    result=dict(completed_updates=300, anchors_episodes_lr_equal=True,finite_gradients=True,
        changed_tensors=len(changed), unchanged_trainable_tensors=len(intended-set(changed)),
        frozen_tensors_unchanged=True, serialized_model_tensors=len(hashes),
        model_sha256=sha(checkpoint/'pretrained_model/model.safetensors'),rng=rng,
        processors=processors, losses=losses,
        missing_training_state=[n for n in ['optimizer_state.safetensors','optimizer_param_groups.json','scheduler_state.json'] if not (checkpoint/'training_state'/n).exists()],
        independent_step300_model_reload=False, full_checkpoint_integrity=False)
    dump(out/'partial-training-audit.json',result)
    dump(out/'serialized-C-tensor-hashes.json',hashes)
    protected=read(out/'historical-before.json')
    changes=[p for p,h in protected.items() if sha(ROOT/p)!=h]
    dump(out/'preservation.json',dict(files=len(protected),changed=changes,passed=not changes))
    print(json.dumps({k:v for k,v in result.items() if k not in ['losses','processors']},indent=2),flush=True)

if __name__=='__main__':
    main()
