"""Authorized 39-cell continuation; no retries beyond B-ID-45 fresh recovery."""
import json
from pathlib import Path
import subprocess
import sys
import time
from instruction_control_audit import ROOT, BASE, PROCESSORS, read, dump, sha
from instruction_control_evaluate import detailed_metrics
from instruction_storage_contract import evaluation_root
from instruction_runtime_storage import runtime_snapshot as snapshot, require_safe

def main():
    out=evaluation_root()
    plan=read(out/'registered-plan.json')
    previous=out/'resume-20261022'
    results=read(previous/'results.json')
    old_ledger=read(previous/'ledger.json')
    assert len(old_ledger)==60 and sum(c['status']=='completed' for c in old_ledger)==21
    assert sum(c['status']=='unrun' for c in old_ledger)==38 and sum(c['status']=='invalid' for c in old_ledger)==1
    assert len(results)==21 and all(r['valid'] for r in results)
    assert {(r['seed'],r['model'],r['condition']) for r in results}=={(s,m,c) for s in range(42,45) for m in 'ABC' for c in ['NL','ID']} | {(45,'A','NL'),(45,'A','ID'),(45,'B','NL')}
    assert plan['seeds']==list(range(42,52)) and plan['episode_cap']==60
    assert plan['execution_horizon']==50 and plan['flow_steps']==10
    assert plan['actions_per_episode']==600 and plan['episode_seconds']==600
    models={k:Path(v) for k,v in plan['checkpoints'].items()}
    def frozen():
        assert evaluation_root()==out
        for n,h in plan['frozen_sources'].items():assert sha(ROOT/'scripts'/n)==h,n
        for m,p in models.items():
            assert sha(p/'model.safetensors')==plan['hashes'][m],m
            for n in PROCESSORS:assert sha(p/n)==sha(BASE/n),n
    frozen()
    require_safe(snapshot())
    failed=out/'B-ID-45'
    assert (failed/'run-started.json').exists()
    assert not (failed/'summary.json').exists()
    assert len((failed/'steps.jsonl').read_text().splitlines())==12
    run=out/'resume-after-archive-20261022'
    assert not run.exists(),'Resume invocation already exists; no automatic retries'
    # Inventory before creating new artifacts. Shared compiled cache may legitimately grow.
    preserved={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and 'numba-cache' not in p.parts}
    run.mkdir()
    dump(run/'preserved-files.json',preserved)
    dump(run/'prior-results.json',results)
    dump(run/'runner-source.json',{'resume':sha(Path(__file__)),'disk_guard':sha(ROOT/'scripts/instruction_disk_guard.py'),'runtime_storage':sha(ROOT/'scripts/instruction_runtime_storage.py')})
    by={(r['seed'],r['model'],r['condition']):r for r in results}
    ledger=[]
    for s in range(42,52):
        for m in 'ABC':
            for c in ['NL','ID']:
                key=(s,m,c)
                folder=ROOT/by[key]['path'] if key in by else out/(f'{m}-{c}-{s}'+('-resume1' if key==(45,'B','ID') else ''))
                ledger.append(dict(seed=s,model=m,condition=c,status='completed' if key in by else 'unrun',path=str(folder),
                    historical_invalid_attempt=str(out/'A-NL-45') if key==(45,'A','NL') else (str(failed) if key==(45,'B','ID') else None)))
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
    prior_seconds=sum(r['worker_seconds'] for r in results)+1200
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
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while True:
                        telemetry=snapshot()
                        telemetry.update(worker_pid=p.pid,cell=folder.name)
                        with (run/'runtime-storage.jsonl').open('a',encoding='utf-8') as f:
                            f.write(json.dumps(telemetry)+'\n')
                        require_safe(telemetry)
                        if p.poll() is not None:break
                        if time.monotonic()-t>=600:raise TimeoutError('Original 600-second worker cap')
                        try:
                            p.wait(timeout=min(5,max(.01,600-(time.monotonic()-t))))
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if p.poll() is None:
                        p.kill()
                        p.wait(timeout=15)
                    dump(run/(folder.name+'-after.json'),snapshot())
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
        dump(run/'complete.json',dict(episodes=60,new_episodes=39,resume_seconds=time.monotonic()-started,
            total_worker_seconds=sum(r['worker_seconds'] for r in results),historical_files_preserved=len(preserved),
            optimizer_scheduler_complete=False,training_replayed=False))
    except Exception as e:
        if current is not None:
            current['status']='invalid';current['error']=repr(e)
            steps=Path(current['path'])/'steps.jsonl'
            current['recorded_action_rows']=len(steps.read_text().splitlines()) if steps.exists() else 0
        save()
        dump(run/'stopped.json',dict(reason=repr(e),completed=len(results),resume_seconds=time.monotonic()-started))
        raise

if __name__=='__main__':main()
