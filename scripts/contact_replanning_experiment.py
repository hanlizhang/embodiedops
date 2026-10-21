"""Frozen four-pair execution experiment. No training, retries or final seeds."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from agent_workflow import ROOT, read, dump, sha, episode_metrics, acceptance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT / 'outputs/agent-workflow-v1')
    out.mkdir(parents=True, exist_ok=False)
    config = ROOT / 'environment/contact-replanning-v1.json'
    plan = read(config)
    source = ROOT / 'outputs/agent-workflow-v1/learning-5000-v1'
    resume = ROOT / 'outputs/agent-workflow-v1/learning-5000-resume-v1'
    selected = read(source / 'selected-checkpoints.json')['episode_balanced']
    checkpoint = Path(selected['checkpoint'])
    assert sha(checkpoint / 'model.safetensors') == selected['sha256']
    sources = [ROOT/'scripts'/n for n in ('phase1d_rollout.py', 'control_diagnostics.py',
               'contact_replanning.py', 'contact_replanning_experiment.py', 'renderer_lifecycle.py',
               'phase1b_contract.py', 'agent_workflow.py')]+[config, ROOT/'tests/test_contact_replanning.py']
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    for p in sources:
        shutil.copyfile(p, out/(p.name+'.txt'))
    shutil.copyfile(config, out/'frozen-config.json')
    historical = {str(p.relative_to(ROOT)): sha(p) for root in (source, resume)
                  for p in root.rglob('*') if p.is_file()}
    dump(out/'historical-before.json', historical)
    dump(out/'registered-plan.json', dict(config=plan, selected=selected, sources=frozen,
         registered_unix=time.time(), episodes=8, no_retry=True,
         rng='seed * 100000 + next physical action index at each refill; diverges after interruption',
         baseline_reuse=False, reserved_final_seeds_not_run=[120, 121, 122, 123]))
    shutil.copytree(ROOT/'outputs/phase1b/numba-cache', out/'numba-cache')
    started = time.monotonic()
    pairs = []
    try:
        for seed in plan['seeds']:
            metrics = {}
            for arm in ('baseline', 'candidate'):
                assert all(sha(ROOT/p) == h for p,h in frozen.items()), 'Frozen source changed'
                folder = out/f'{arm}-{seed}'
                reference = ((resume if seed == 46 else source)/f'diagnostic-{seed}-episode_balanced'
                             if arm == 'baseline' else out/f'baseline-{seed}')
                command = [sys.executable, '-B', '-u', str(ROOT/'scripts/phase1d_rollout.py'),
                    '--output', str(folder), '--seed', str(seed), '--checkpoint', str(checkpoint),
                    '--reference', str(reference), '--instruction', 'CloseFridge',
                    '--workflow-diagnostics', '--renderer-owner-cleanup',
                    '--execution-feedback', arm, '--feedback-config', str(out/'frozen-config.json')]
                dump(out/f'command-{arm}-{seed}.json', command)
                print(f'Starting {arm} seed {seed}', flush=True)
                tick = time.monotonic()
                with (out/f'{arm}-{seed}.log').open('x') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=plan['episode_timeout_seconds'])
                elapsed = time.monotonic()-tick
                assert elapsed <= plan['episode_timeout_seconds'], 'Worker exceeded deadline'
                assert result.returncode in (0, 2)
                m = episode_metrics(folder)
                m['worker_seconds'] = elapsed
                assert m['valid'] or m['guard_failure'], f'Invalid episode; no retry: {m}'
                manifest = read(folder/'manifest.json')
                assert manifest['model_sha256'] == selected['sha256']
                ref_manifest = read(reference/'manifest.json')
                for k in ('model_sha256', 'checkpoint_file_hashes', 'versions', 'rng_rule', 'execution_horizon'):
                    assert manifest[k] == ref_manifest[k], f'Contract mismatch: {k}'
                assert read(folder/'policy.json') == read(reference/'policy.json')
                assert read(folder/'scene.json')['controller'] == read(reference/'scene.json')['controller']
                assert read(folder/'paired-reset-validation.json')['passed']
                metrics[arm] = m
                dump(out/f'metrics-{arm}-{seed}.json', m)
                print(f'Finished {arm} seed {seed}: {m["termination"]}, {m["actions"]} actions', flush=True)
            pairs.append(dict(seed=seed, **metrics, matched_reset=True))
            dump(out/'partial-pairs.json', pairs)
        dump(out/'result.json', dict(gate=acceptance(pairs, plan['acceptance']),
                                     total_seconds=time.monotonic()-started))
    finally:
        changed = [p for p,h in historical.items() if sha(ROOT/p) != h]
        dump(out/'preservation.json', dict(checked=len(historical), changed=changed, passed=not changed))
        assert not changed


if __name__ == '__main__':
    main()
