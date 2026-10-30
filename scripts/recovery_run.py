"""Exclusive serial branch driver with whole-process timeout and preservation."""
import json
from pathlib import Path
import subprocess
import sys
import time
from recovery_feasibility import output_root
from instruction_control_audit import ROOT,read,dump,sha


def main():
    out=output_root(); plan=read(out/'registered-plan.json')
    assert read(out/'audit-summary.json')['restoration_gate_passed']
    assert all(sha(ROOT/p)==h for p,h in read(out/'preserved-before.json').items())
    sources=['recovery_run.py','recovery_branches.py','recovery_feasibility.py',
             'phase1d_rollout.py','phase1b_contract.py','phase1c_replay.py',
             'control_diagnostics.py','renderer_lifecycle.py','instruction_disk_guard.py']
    frozen={n:sha(ROOT/'scripts'/n) for n in sources}
    branch_plan=dict(frozen_sources=frozen,created=time.time(),branches=['vla0','vla1','vla2','vla3','expert'],
        states=[s['id'] for s in plan['states']],recovery_actions_per_branch=50,
        audit_replay_cap=1200,additional_prefix_actions_cap=5250,
        total_branch_attempts=10,total_episode_attempts_including_audits=12,
        expert='Nearest collision-surface approach at <=1cm target increment; when distance <=12mm push 1cm along negative-angle hinge tangent; open gripper, achieved mode, fixed base and orientation; no retries or search',
        identical_policy_inputs='Use first audited boundary RGB/16D for every candidate after verifying branch render within registered tolerance',
        limitations='50 actions is 2.5 seconds; no claim of general recoverability from unsuccessful pilot',
        runtime_cap_seconds=5400)
    with (out/'branch-plan.json').open('x') as f:json.dump(branch_plan,f,indent=2)
    for n in sources: (out/(n+'.txt')).write_bytes((ROOT/'scripts'/n).read_bytes())
    results=[]; status={}
    start=time.monotonic()
    try:
        for state in plan['states']:
            for branch in branch_plan['branches']:
                assert all(sha(ROOT/'scripts'/n)==h for n,h in frozen.items())
                remaining=5400-(time.monotonic()-start)
                assert remaining>0
                cmd=[sys.executable,'-B','-u',str(ROOT/'scripts/recovery_branches.py'),
                     '--state',state['id'],'--branch',branch]
                name=state['id']+'-'+branch
                print('Starting '+name,flush=True)
                with (out/(name+'.log')).open('x') as log:
                    p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(600,remaining))
                result=read(out/name/'result.json'); results.append(result)
                dump(out/'branch-results.json',results)
                print(json.dumps(result),flush=True)
                assert p.returncode==0 and result['valid'],'Invalid branch; study stopped without retry'
        status['complete']=True
    except Exception as exc: status.update(complete=False,reason=str(exc))
    finally:
        protected=read(out/'preserved-before.json')
        status.update(seconds=time.monotonic()-start,completed=len(results),
            preservation=dict(checked=len(protected),changed=[p for p,h in protected.items() if sha(ROOT/p)!=h]),
            frozen_sources_unchanged=all(sha(ROOT/'scripts'/n)==h for n,h in frozen.items()))
        dump(out/'branch-status.json',status)
    print(json.dumps(status,indent=2))


if __name__=='__main__':main()
