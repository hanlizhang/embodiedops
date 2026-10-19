"""Fixed H=20 seeds 42--44, historical H=50 audit, no training or retries."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/control-horizon-v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def reference(seed):
    return (ROOT / 'outputs/phase1d/language-control-native' if seed == 42 else
            ROOT / f'outputs/probes/instruction-distribution-v1/approved-retry-1/task-id-seed{seed}')


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding='utf-8')


def preflight():
    import numpy as np
    import torch
    import mujoco
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from control_diagnostics import contact_snapshot, action_change
    assert Path(sys.prefix).name == 'embodiedops-robocasa'
    assert not (OUT / 'preflight.json').exists()
    # Exercise the installed select_action/reset with synthetic chunks, no checkpoint load.
    for h in (20, 50):
        calls = []
        def chunk(batch, noise):
            calls.append(batch['observation.state'].item())
            return torch.arange(50).reshape(1, 50, 1)
        p = SimpleNamespace(config=SimpleNamespace(n_action_steps=h), _rtc_enabled=lambda: False,
                            eval=lambda: None, _prepare_batch=lambda b: b, _get_action_chunk=chunk)
        SmolVLAPolicy.reset(p)
        p._check_get_actions_condition = lambda: len(p._queues['action']) == 0
        for i in range(2*h+1):
            value = SmolVLAPolicy.select_action(p, {'observation.state': torch.tensor(i)})
            assert value.item() == i % h
        assert calls == [0, h, 2*h]
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .2"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    sim = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data))
    assert not contact_snapshot(sim)
    for _ in range(500):
        mujoco.mj_step(model, data)
    before = data.qpos.copy()
    contacts = contact_snapshot(sim)
    assert contacts and contacts[0]['force_N_contact_frame'][0] > 0
    assert np.array_equal(before, data.qpos)
    assert action_change(np.zeros(12), np.ones(12))['mode_switch']
    history = {}
    for seed in (42, 43, 44):
        folder = reference(seed)
        policy = read(folder/'policy.json')['config']
        assert (policy['chunk_size'], policy['n_action_steps'], policy['num_steps']) == (50, 50, 10)
        assert read(folder/'reset.json')['task'] == 'CloseFridge'
        rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in rows if r['inference']] == list(range(1, len(rows)+1, 50))
        assert all(r['finite'] and not r['guard_failures'] for r in rows)
        history[str(seed)] = {'path': str(folder), 'actions': len(rows), 'success': read(folder/'summary.json')['final_fridge']['success']}
    # Preserve pre-existing source/report and all historical experiment files by hashes.
    paths = []
    for sub in ('outputs/phase1d', 'outputs/phase2', 'outputs/probes', 'environment'):
        paths.extend(p for p in (ROOT/sub).rglob('*') if p.is_file())
    paths.extend(ROOT/'scripts'/n for n in ('instruction_distribution_probe.py', 'phase2_train_v1.py', 'phase2_audit_v1.py'))
    save('preservation-before.json', {str(p.relative_to(ROOT)): sha(p) for p in paths})
    shutil.copytree(ROOT/'outputs/phase1b/numba-cache', OUT/'numba-cache')
    source = Path(inspect.getfile(SmolVLAPolicy))
    save('preflight.json', {'queue_test': 'passed H20 and H50, fresh batch at each replan; 50 predicted, first H executed',
        'contact_test': 'passed airborne vs loaded sphere, positive solver force, qpos unchanged',
        'installed_policy_source': str(source), 'installed_policy_sha256': sha(source),
        'historical_contracts': history, 'seeds': [42,43,44], 'new_horizon': 20,
        'runner_sha256': sha(ROOT/'scripts/phase1d_rollout.py'),
        'diagnostics_sha256': sha(ROOT/'scripts/control_diagnostics.py'),
        'prediction_chunk': 50, 'flow_steps': 10, 'max_actions': 600,
        'instruction': 'CloseFridge', 'historical_contacts': 'not recorded; unavailable, never inferred from proximity'})
    print('Preflight passed; no policy rollouts or training executed.', flush=True)


def run(remaining=False):
    plan = read(OUT/'preflight.json')
    assert sha(ROOT/'scripts/phase1d_rollout.py') == plan['runner_sha256']
    assert sha(ROOT/'scripts/control_diagnostics.py') == plan['diagnostics_sha256']
    seeds = (43,44) if remaining else (42,43,44)
    if remaining:
        assert read(OUT/'h20-seed42/summary.json')['termination_reason'].startswith('guard_failure:')
        assert all(not (OUT/f'h20-seed{s}').exists() for s in seeds)
    with (OUT/('remaining-started.json' if remaining else 'execution-started.json')).open('x') as f:
        json.dump({'seeds': seeds, 'horizon': 20, 'no_retry': True}, f)
    for seed in seeds:
        command = [sys.executable, '-B', '-u', str(ROOT/'scripts/phase1d_rollout.py'),
                   '--output', str(OUT/f'h20-seed{seed}'), '--seed', str(seed),
                   '--reference', str(reference(seed)), '--instruction', 'CloseFridge',
                   '--execution-horizon', '20', '--control-diagnostics']
        save(f'command-seed{seed}.json', command)
        print(f'Running H=20 seed {seed}', flush=True)
        with (OUT/f'seed{seed}.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        print(f'Seed {seed}: exit {result.returncode}', flush=True)
        summary = read(OUT/f'h20-seed{seed}/summary.json')
        assert result.returncode == 0 or (result.returncode == 2 and summary['termination_reason'].startswith('guard_failure:')), 'Stopped on runtime exception; no automatic retry'
    old = read(OUT/'preservation-before.json')
    assert all(sha(ROOT/p) == digest for p, digest in old.items())
    save('preservation-after.json', {'unchanged_files': len(old), 'passed': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'run', 'remaining'])
    args = parser.parse_args()
    {'preflight': preflight, 'run': run, 'remaining': lambda: run(remaining=True)}[args.mode]()
