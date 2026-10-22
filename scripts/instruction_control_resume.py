"""Resume only unrun registered cells; preserve prior attempts and frozen sources."""
import json
from pathlib import Path
import subprocess
import sys
import time
from instruction_control_audit import ROOT, BASE, PROCESSORS, read, dump, sha
from instruction_control_evaluate import detailed_metrics
from instruction_storage_contract import evaluation_root
from instruction_disk_guard import snapshot, require_safe

def main():
    out=evaluation_root()
    plan=read(out/'registered-plan.json')
    results=read(out/'results.json')
    assert len(results)==18 and all(r['valid'] for r in results)
    assert {(r['seed'],r['model'],r['condition']) for r in results}=={(s,m,c) for s in range(42,45) for m in 'ABC' for c in ['NL','ID']}
    models={k:Path(v) for k,v in plan['checkpoints'].items()}
    def frozen():
        assert evaluation_root()==out
        for n,h in plan['frozen_sources'].items():assert sha(ROOT/'scripts'/n)==h,n
        for m,p in models.items():
            assert sha(p/'model.safetensors')==plan['hashes'][m],m
            for n in PROCESSORS:assert sha(p/n)==sha(BASE/n),n
    frozen()
    require_safe(snapshot())
    failed=out/'A-NL-45'
    assert (failed/'run-started.json').exists()
    assert not any((failed/n).exists() for n in ['reset.json','steps.jsonl','summary.json'])
    run=out/'resume-20261022'
    assert not run.exists(),'Resume invocation already exists; no automatic retries'
    # Inventory before creating new artifacts. Shared compiled cache may legitimately grow.
    preserved={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'numba-cache' not in p.parts}
    run.mkdir()
    dump(run/'preserved-files.json',preserved)
    dump(run/'prior-results.json',results)
    dump(run/'runner-source.json',{'resume':sha(Path(__file__)),'disk_guard':sha(ROOT/'scripts/instruction_disk_guard.py')})
    by={(r['seed'],r['model'],r['condition']):r for r in results}
    ledger=[]
    for s in range(42,52):
        for m in 'ABC':
            for c in ['NL','ID']:
                key=(s,m,c)
                folder=out/(f'{m}-{c}-{s}'+('-resume1' if key==(45,'A','NL') else ''))
                ledger.append(dict(seed=s,model=m,condition=c,status='completed' if key in by else 'unrun',path=str(folder),
                    historical_invalid_attempt=str(failed) if key==(45,'A','NL') else None))
    def save():
        dump(run/'ledger.json',ledger)
        dump(run/'results.json',results)
    def disk(label):
        data=snapshot()
        dump(run/(label+'-disk.json'),data)
        require_safe(data)
    def validate(folder,m,c,s):
        manifest=read(folder/'manifest.json')
        assert manifest['model_sha256']==plan['hashes'][m]
        assert read(folder/'paired-reset-validation.json')['passed']
        assert manifest['rng_rule']=='manual_seed(seed * 100000 + action_index) before each new chunk'
        assert read(folder/'reset.json')['hashes']==plan['historical_scene_reset_hashes'][str(s)]
        for n in PROCESSORS:assert manifest['checkpoint_file_hashes'][n]==sha(BASE/n)
        if m!='A':
            reference=read(ROOT/by[(s,'A',c)]['path']/'manifest.json')
            for k in ['versions','simulator_sources','execution_horizon','rng_rule','renderer_owner_cleanup']:
                assert manifest[k]==reference[k],k
        from PIL import Image
        import numpy as np
        cameras=list(folder.glob('final-robot0_*.png'))
        assert len(cameras)==3
        for camera in cameras:assert np.asarray(Image.open(camera)).max()>0,'Invalid final camera'
    save()
    for r in results:validate(ROOT/r['path'],r['model'],r['condition'],r['seed'])
    # Count prior completed worker time plus the full allowance for the interrupted attempt.
    prior_seconds=sum(r['worker_seconds'] for r in results)+600
    started=time.monotonic()
    current=None
    try:
        for cell in ledger:
            if cell['status']=='completed':continue
            current=cell
            s,m,c=cell['seed'],cell['model'],cell['condition']
            folder=Path(cell['path'])
            assert not folder.exists()
            frozen()
            disk(folder.name+'-before')
            remaining=plan['total_seconds']-prior_seconds-(time.monotonic()-started)
            assert remaining>=600,'Original compute budget exhausted'
            cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/phase1d_rollout.py'),
                '--output',str(folder),'--seed',str(s),'--instruction',
                plan['instructions'][str(s)] if c=='NL' else 'CloseFridge',
                '--checkpoint','lerobot/smolvla_robocasa' if m=='A' else str(models[m]),
                '--workflow-diagnostics','--renderer-owner-cleanup','--study-observations']
            if m=='A' and c=='NL':cmd+=['--reference',str(ROOT/f'outputs/phase1d/evaluation/trained-seed{s}')]
            elif m=='A':cmd+=['--reference',str(ROOT/by[(s,'A','NL')]['path']),'--instruction-contrast']
            else:cmd+=['--reference',str(ROOT/by[(s,'A',c)]['path'])]
            dump(run/(folder.name+'-command.json'),cmd)
            cell['status']='running';save()
            print(f'Starting {folder.name} ({len(results)+1}/60)',flush=True)
            t=time.monotonic()
            with (run/(folder.name+'.log')).open('x') as log:
                p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            elapsed=time.monotonic()-t
            disk(folder.name+'-after')
            assert elapsed<=600 and p.returncode in (0,2),f'Worker error {p.returncode}'
            validate(folder,m,c,s)
            r=detailed_metrics(folder)
            assert r['valid'] and not r['guard_failure'],'Invalid episode or physical guard failure; stop'
            r.update(model=m,condition=c,seed=s,worker_seconds=elapsed)
            by[(s,m,c)]=r;results.append(r)
            cell['status']='completed';save();current=None
            print(f'Finished {folder.name}: {r["termination"]}; final degrees {r["final_angles_deg"]}',flush=True)
            if m=='C' and c=='ID':
                assert sum(r['seed']==s for r in results)==6
                disk(f'batch-{s}-complete')
                dump(run/f'batch-{s}-complete.json',dict(contracts_passed=True,episodes=6))
        assert len(results)==60
        frozen()
        for n,h in preserved.items():assert sha(out/n)==h,f'Preservation mismatch {n}'
        dump(run/'complete.json',dict(episodes=60,new_episodes=42,resume_seconds=time.monotonic()-started,
            total_worker_seconds=sum(r['worker_seconds'] for r in results),historical_files_preserved=len(preserved),
            optimizer_scheduler_complete=False,training_replayed=False))
    except Exception as e:
        if current is not None:current['status']='invalid';current['error']=repr(e)
        save()
        dump(run/'stopped.json',dict(reason=repr(e),completed=len(results),resume_seconds=time.monotonic()-started))
        raise

if __name__=='__main__':main()
