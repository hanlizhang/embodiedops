"""Reproducible demonstration sampling; motion regimes are not contact labels."""
import numpy as np

LABELS=('quiet','commanded_low_motion','other_motion')


def motion_regimes(actions, states, window=5):
    a=np.asarray(actions,dtype=float);s=np.asarray(states,dtype=float)
    if a.ndim!=2 or a.shape[1]!=12 or s.shape!=(len(a),16):
        raise ValueError('Expected episode actions [N,12] and states [N,16]')
    if not np.isfinite(a).all() or not np.isfinite(s).all():raise ValueError('Nonfinite demonstration')
    requested=np.linalg.norm(np.clip(a[:,5:8],-1,1)*.05,axis=1)
    observed=np.linalg.norm(np.diff(s[:,7:10],axis=0),axis=1)
    labels=np.full(len(a),2,dtype=int)
    # Last frame has no observed next state and cannot establish low response.
    for i in range(len(a)-1):
        end=min(i+window,len(a)-1)
        quiet=(np.linalg.norm(a[i:end,5:8],axis=1).max()<.05 and
               np.linalg.norm(a[i:end,8:11],axis=1).max()<.05 and
               np.linalg.norm(a[i:end,:4],axis=1).max()<.05 and
               np.all((a[i:end+1,11]>=.5)==(a[i,11]>=.5)))
        if quiet:labels[i]=0
        elif requested[i:end].mean()>=.005 and observed[i:end].mean()<.002:labels[i]=1
    return labels


def build_plan(episodes, count, seed, method='episode_balanced', labels=None):
    episodes=np.asarray(episodes);groups={int(e):np.flatnonzero(episodes==e) for e in np.unique(episodes)}
    if not groups:raise ValueError('No episodes')
    order_rng=np.random.default_rng(seed);frame_rng=np.random.default_rng(seed+101)
    phase_rng=np.random.default_rng(seed+202);plan=[]
    if method not in ('episode_balanced','motion_mixture'):raise ValueError(method)
    if method=='motion_mixture' and (labels is None or len(labels)!=len(episodes)):raise ValueError('Missing motion labels')
    while len(plan)<count:
        for e in order_rng.permutation(sorted(groups)):
            pool=groups[e];mode='uniform'
            if method=='motion_mixture' and phase_rng.random()<.5:
                regime=int(phase_rng.choice(np.unique(labels[pool])))
                pool=pool[labels[pool]==regime];mode=LABELS[regime]
            idx=int(frame_rng.choice(pool))
            plan.append(dict(index=idx,episode=int(e),sampling_component=mode))
            if len(plan)==count:break
    return plan


def dataset_labels(dataset):
    table=dataset.hf_dataset.with_format(None)
    episodes=np.asarray(table['episode_index']);actions=np.asarray(table['action']);states=np.asarray(table['observation.state'])
    frames=np.asarray(table['frame_index']);labels=np.empty(len(episodes),dtype=int)
    for e in np.unique(episodes):
        ix=np.flatnonzero(episodes==e)
        if not np.all(np.diff(frames[ix])==1):raise ValueError('Nonconsecutive episode frames')
        labels[ix]=motion_regimes(actions[ix],states[ix])
    return episodes,labels
