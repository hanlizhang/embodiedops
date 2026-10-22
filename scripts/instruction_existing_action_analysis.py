"""Compare existing first chunks only; no policy execution, simulator or new sampling."""
import itertools
import argparse
from pathlib import Path
import numpy as np
from instruction_control_audit import ROOT,read,dump,sha
from instruction_storage_contract import evaluation_root

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-directory',type=Path)
    args=ap.parse_args()
    out=evaluation_root();run=args.results_directory or out/'resume-20261022'
    comparisons=[]
    for seed in [43,44]:
        for condition in ['NL','ID']:
            for a,b in itertools.combinations('ABC',2):
                pa,pb=[out/f'{m}-{condition}-{seed}' for m in [a,b]]
                oa,ob=[np.load(p/'observation-0001.npz') for p in [pa,pb]]
                equality={k:bool(np.array_equal(oa[k],ob[k])) for k in oa.files}
                pixel_delta={k:int(np.abs(oa[k].astype(np.int16)-ob[k].astype(np.int16)).max()) for k in oa.files if k!='state'}
                aa,ab=[np.array(read(p/'chunk-01.json')['all_actions']) for p in [pa,pb]]
                assert aa.shape==ab.shape==(50,12)
                comparisons.append(dict(seed=seed,condition=condition,pair=a+'-'+b,exact_same_observation=all(equality.values()),
                    observation_equality=equality,max_pixel_difference=pixel_delta,
                    action_rmse_per_dimension=np.sqrt(np.mean((aa-ab)**2,axis=0)).tolist(),
                    action_max_abs_difference=float(np.max(np.abs(aa-ab))),
                    sources=[dict(path=str(p/'chunk-01.json'),sha256=sha(p/'chunk-01.json')) for p in [pa,pb]]))
    dump(run/'existing-first-chunk-comparison.json',dict(comparisons=comparisons,
        note='Existing reset/approach chunks only. Identical raw inputs are explicitly marked; other comparisons are not fixed-observation evidence. Same registered refill seed and sampler; noise bytes were not separately fingerprinted. No aligned demonstration target, no new inference. Prediction differences are not manipulation improvement.'))
    print([r for r in comparisons if r['seed']==44 and r['condition']=='NL'])

if __name__=='__main__':main()
