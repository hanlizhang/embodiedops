"""Read frozen results; descriptive comparisons, including an incomplete matrix."""
import statistics
import argparse
from pathlib import Path
from instruction_control_audit import ROOT,read,dump,sha
from instruction_storage_contract import evaluation_root,STUDY

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-directory',type=Path)
    args=ap.parse_args()
    out=evaluation_root();run=args.results_directory or out/'resume-20261022'
    complete=(run/'complete.json').exists()
    rows=read(run/'results.json')
    assert len(rows)<=60 and all(r['valid'] for r in rows)
    historical=read(ROOT/'outputs/phase1d/evaluation/results.json')['episodes']
    grouped=[]
    for c in ['ID','NL']:
        for m in 'ABC':
            rr=[r for r in rows if r['condition']==c and r['model']==m]
            success=[r for r in rr if r['success']]
            grouped.append(dict(model=m,condition=c,successes=len(success),episodes=len(rr),
                mean_total_door_progress_deg=statistics.mean(r['closing_deg'] for r in rr),
                mean_per_door_progress_deg=statistics.mean(v for r in rr for v in r['per_door_progress_deg']),
                mean_fridge_loaded_actions=statistics.mean(r['contacts']['fridge']['loaded_actions'] for r in rr),
                median_fridge_peak_N=statistics.median(r['contacts']['fridge']['peak_N'] for r in rr),
                maximum_fridge_peak_N=max(r['contacts']['fridge']['peak_N'] for r in rr),
                maximum_self_peak_N=max(r['contacts']['self']['peak_N'] for r in rr),
                mean_chunk_latency_s=statistics.mean(r['mean_inference_seconds'] for r in rr),
                success_action_counts=[r['actions'] for r in success],
                success_simulated_seconds=[r['actions']/20 for r in success],
                success_worker_seconds=[r['worker_seconds'] for r in success]))
    reproduction=[]
    for old in historical:
        m={'pretrained':'A','trained':'B'}[old['checkpoint']]
        new=next((r for r in rows if r['seed']==old['seed'] and r['model']==m and r['condition']=='NL'),None)
        if new is None:continue
        reproduction.append(dict(seed=old['seed'],model=m,historical_success=old['success'],current_success=new['success'],
            historical_actions=old['actions'],current_actions=new['actions'],
            historical_final_degrees=old['door_degrees'],current_final_degrees=new['final_angles_deg']))
    oldfiles=read(STUDY/'historical-before.json')
    assert all(sha(ROOT/p)==h for p,h in oldfiles.items())
    dump(run/'physical-comparison.json',dict(complete=complete,groups=grouped,historical_NL_reproduction=reproduction,
        historical_files_preserved=len(oldfiles),episode_worker_seconds=sum(r['worker_seconds'] for r in rows),
        inference_seconds=sum(r['total_inference_seconds'] for r in rows),
        rollout_seconds=sum(r['rollout_seconds'] for r in rows),
        max_cuda_allocated_bytes=max(r['peak_cuda_allocated'] for r in rows),
        interpretation='Descriptive matched diagnostic scenes; no population/generalization claim. Commands are not achieved motion.'))
    print(grouped)

if __name__=='__main__':main()
