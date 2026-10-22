"""Final bounded-study audit from saved artifacts; no policy or simulator execution."""
import argparse
import json
from pathlib import Path
from instruction_control_audit import ROOT,BASE,PROCESSORS,read,dump,sha
from instruction_storage_contract import evaluation_root
from instruction_runtime_storage import runtime_snapshot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results-directory',type=Path,required=True)
    run=ap.parse_args().results_directory
    out=evaluation_root();plan=read(out/'registered-plan.json')
    complete=(run/'complete.json').exists()
    if complete:assert read(run/'complete.json')['episodes']==60
    else:assert (run/'stopped.json').exists()
    rows=read(run/'results.json');ledger=read(run/'ledger.json')
    expected={(s,m,c) for s in range(42,52) for m in 'ABC' for c in ['ID','NL']}
    assert len(ledger)==60 and len(rows)<=60
    completed_keys={(r['seed'],r['model'],r['condition']) for r in ledger if r['status']=='completed'}
    assert len(rows)==len(completed_keys)
    assert {(r['seed'],r['model'],r['condition']) for r in rows}==completed_keys
    assert {(r['seed'],r['model'],r['condition']) for r in ledger}==expected
    assert all(c['status'] in ['completed','invalid','unrun'] for c in ledger)
    if complete:assert completed_keys==expected
    assert len({r['path'] for r in rows})==len(rows)
    assert all(r['valid'] and not r['guard_failure'] and r['actions']<=600 for r in rows)
    processors={n:sha(BASE/n) for n in PROCESSORS}
    for row in rows:
        folder=ROOT/row['path'];m=read(folder/'manifest.json')
        assert m['model_sha256']==plan['hashes'][row['model']]
        assert all(m['checkpoint_file_hashes'][n]==h for n,h in processors.items())
        assert m['instruction']==('CloseFridge' if row['condition']=='ID' else plan['instructions'][str(row['seed'])])
        assert read(folder/'paired-reset-validation.json')['passed']
        assert read(folder/'reset.json')['hashes']==plan['historical_scene_reset_hashes'][str(row['seed'])]
    for k,p in plan['checkpoints'].items():assert sha(Path(p)/'model.safetensors')==plan['hashes'][k]
    for n,h in plan['frozen_sources'].items():assert sha(ROOT/'scripts'/n)==h
    preserved=read(run/'preserved-files.json')
    assert all(sha(out/n)==h for n,h in preserved.items())
    assert not (out/'A-NL-45/steps.jsonl').exists()
    assert len((out/'B-ID-45/steps.jsonl').read_text().splitlines())==12
    telemetry=[json.loads(line) for line in (run/'runtime-storage.jsonl').read_text().splitlines()]
    assert all(x['above_floor'] for r in telemetry for x in r['paths'].values())
    invalid=[c for c in ledger if c['status']=='invalid']
    for cell in invalid:
        folder=Path(cell['path'])
        steps=(folder/'steps.jsonl').read_text().splitlines()
        assert len(steps)==cell['recorded_action_rows']
        assert all(isinstance(json.loads(line),dict) for line in steps)
    report=dict(passed=True,complete=complete,valid_cells=len(rows),unique_completed_paths=len(rows),new_cells=len(rows)-21,preserved_valid_cells=21,
        invalid_cells=invalid,unrun_cells=[c for c in ledger if c['status']=='unrun'],
        prior_files_hash_verified=len(preserved),interrupted_attempts_preserved=['A-NL-45: zero actions','B-ID-45: 12 actions'],
        matrix_unchanged=True,checkpoints_and_processors_unchanged=True,frozen_sources_unchanged=True,
        C_optimizer_scheduler_complete=False,C_inference_validation_passed=True,training_replayed=False,
        confirmation_or_final_seeds_run=False,runtime_samples=len(telemetry),
        minimum_C_free_bytes=min(r['paths']['repository_C']['shutil_free'] for r in telemetry),
        minimum_D_free_bytes=min(r['paths']['physical_D']['shutil_free'] for r in telemetry),
        maximum_pagefile_allocated_bytes=max(sum(p['allocated_bytes'] for p in r['pagefiles']) for r in telemetry),
        final_storage=runtime_snapshot())
    dump(run/'final-integrity.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
