"""User-authorized single seed46 baseline retry and remaining candidate."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agent_workflow import ROOT, read, dump, sha, episode_metrics, acceptance


def main():
    source = ROOT/'outputs/agent-workflow-v1/contact-replanning-v1'
    out = ROOT/'outputs/agent-workflow-v1/contact-replanning-v1-seed46-continuation'
    plan = read(source/'registered-plan.json')
    frozen = plan['sources']
    assert all(sha(ROOT/p) == h for p,h in frozen.items())
    failed = read(source/'baseline-46/summary.json')
    assert failed['simulator_steps'] == failed['inference_calls'] == 0
    assert 'out of memory' in failed['exception']
    checkpoint = Path(plan['selected']['checkpoint'])
    assert sha(checkpoint/'model.safetensors') == plan['selected']['sha256']
    out.mkdir(parents=True, exist_ok=False)
    protected = read(source/'historical-before.json')
    protected.update({str(p.relative_to(ROOT)): sha(p) for p in source.rglob('*') if p.is_file()})
    dump(out/'preservation-before.json', protected)
    for name in ('contact-replanning-v1.md', 'contact-replanning-v1-results.json'):
        shutil.copyfile(ROOT/'environment'/name, out/('prior-'+name))
    shutil.copyfile(source/'frozen-config.json', out/'frozen-config.json')
    shutil.copyfile(Path(__file__), out/'executed-resume.py.txt')
    shutil.copytree(ROOT/'outputs/phase1b/numba-cache', out/'numba-cache')
    dump(out/'continuation-plan.json', dict(
        authorization='User explicitly approved one seed46 baseline retry and remaining candidate; all settings unchanged.',
        frozen_sources=frozen, config_sha256=sha(out/'frozen-config.json'),
        max_baseline_attempts=1, max_candidate_attempts=1, max_actions_per_episode=600,
        episode_timeout_seconds=plan['config']['episode_timeout_seconds'],
        excluded_attempt=dict(path=str(source/'baseline-46'), **failed), no_further_retry=True))
    pairs = read(source/'partial-pairs.json')
    assert [p['seed'] for p in pairs] == [43,44,45]
    metrics = {}
    start = time.monotonic()
    try:
        for arm in ('baseline','candidate'):
            assert all(sha(ROOT/p) == h for p,h in frozen.items())
            folder = out/f'{arm}-46'
            reference = (ROOT/'outputs/agent-workflow-v1/learning-5000-resume-v1/diagnostic-46-episode_balanced'
                         if arm == 'baseline' else out/'baseline-46')
            command = [sys.executable, '-B', '-u', str(ROOT/'scripts/phase1d_rollout.py'),
                '--output', str(folder), '--seed', '46', '--checkpoint', str(checkpoint),
                '--reference', str(reference), '--instruction', 'CloseFridge',
                '--workflow-diagnostics', '--renderer-owner-cleanup',
                '--execution-feedback', arm, '--feedback-config', str(out/'frozen-config.json')]
            dump(out/f'command-{arm}-46.json', command)
            gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free',
                                  '--format=csv'], capture_output=True, text=True)
            (out/f'gpu-before-{arm}.txt').write_text(gpu.stdout+gpu.stderr)
            print(f'Starting seed46 {arm}', flush=True)
            tick = time.monotonic()
            with (out/f'{arm}-46.log').open('x') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=plan['config']['episode_timeout_seconds'])
            elapsed = time.monotonic()-tick
            assert elapsed <= plan['config']['episode_timeout_seconds']
            assert result.returncode in (0,2)
            m = episode_metrics(folder)
            m['worker_seconds'] = elapsed
            dump(out/f'metrics-{arm}-46.json', m)
            assert m['valid'] or m['guard_failure'], 'Invalid episode: no further retry'
            manifest, ref = read(folder/'manifest.json'), read(reference/'manifest.json')
            assert manifest['model_sha256'] == plan['selected']['sha256']
            for key in ('model_sha256','checkpoint_file_hashes','versions','rng_rule','execution_horizon'):
                assert manifest[key] == ref[key], key
            assert read(folder/'policy.json') == read(reference/'policy.json')
            assert read(folder/'scene.json')['controller'] == read(reference/'scene.json')['controller']
            assert read(folder/'paired-reset-validation.json')['passed']
            metrics[arm] = m
            print(f'Finished {arm}: {m["termination"]}, {m["actions"]} actions', flush=True)
        pairs.append(dict(seed=46, **metrics, matched_reset=True))
        dump(out/'result.json', dict(gate=acceptance(pairs,plan['config']['acceptance']),
                                    total_seconds=time.monotonic()-start,
                                    excluded_attempts=[dict(episode='original-baseline-46', path=str(source/'baseline-46'),
                                        termination=failed['termination_reason'], actions=0, inference_calls=0,
                                        exception=failed['exception'])]))
    finally:
        changed = [p for p,h in protected.items() if sha(ROOT/p) != h]
        dump(out/'preservation.json', dict(checked=len(protected), changed=changed, passed=not changed))
        assert not changed


if __name__ == '__main__':
    main()
