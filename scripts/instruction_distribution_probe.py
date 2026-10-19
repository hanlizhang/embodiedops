"""Bounded seed-42--44 instruction contrast; exactly two new episodes, no retries.

--prepare audits existing evidence and registers the plan without simulation.
--run requires that plan and launches only task-ID seeds 43 and 44.
All generated files remain under outputs/probes/instruction-distribution-v1.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/probes/instruction-distribution-v1'
EVAL = ROOT / 'outputs/phase1d/evaluation'
TRAIN = ROOT / 'outputs/phase1d/training-v2'
REV = 'f7af588d11e30b0069f200792fb285904b74c80b'
MODEL = ROOT / 'outputs/phase1a/hf-cache/hub/models--lerobot--smolvla_robocasa/snapshots' / REV
DATA = ROOT / 'outputs/phase1a/hf-cache/lerobot/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots/b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def natural(seed):
    return ROOT / 'outputs/phase1c/retry-20261018' if seed == 42 else EVAL / f'pretrained-seed{seed}'


def task_id(seed):
    return ROOT / 'outputs/phase1d/language-control-native' if seed == 42 else OUT / f'task-id-seed{seed}'


def inventory():
    files = [p for folder in ['outputs/phase1c', 'outputs/phase1d'] for p in (ROOT / folder).rglob('*') if p.is_file()]
    files += list(MODEL.glob('*')) + [ROOT / 'RESUME_EVIDENCE.md']
    return {p.relative_to(ROOT).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns, sha(p)]
            for p in sorted(files) if p.is_file()}


def prepare():
    assert not OUT.exists(), 'Preserve the existing probe; do not prepare twice'
    OUT.mkdir(parents=True)
    save('preservation-before.json', inventory())
    assert sha(MODEL / 'model.safetensors') == '96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b'
    rows = []
    for seed in [42, 43, 44]:
        for arm, path in [('natural', natural(seed)), ('task-id', task_id(seed))]:
            reused = arm == 'natural' or seed == 42
            row = {'seed': seed, 'arm': arm, 'reused': reused, 'artifacts': path.relative_to(ROOT).as_posix()}
            if reused:
                m, s, reset = read(path / 'manifest.json'), read(path / 'summary.json'), read(path / 'reset.json')
                assert m['model'] == 'lerobot/smolvla_robocasa' and m['model_revision'] == REV and m['seed'] == seed
                assert s['termination_reason'] == 'registered_horizon_600' and s['simulator_steps'] == 600
                assert not s.get('exception') and not s['final_fridge']['success']
                assert reset['task'] == ('CloseFridge' if arm == 'task-id' else ('Close the fridge doors.' if seed == 42 else 'Close the fridge door.'))
                for name, h in m['checkpoint_file_hashes'].items():
                    assert sha(MODEL / name) == h
                row['summary_sha256'] = sha(path / 'summary.json')
            rows.append(row)
    # Test the new guard in isolation, without importing a policy or simulator.
    source = (ROOT / 'scripts/phase1d_rollout.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'validate_reference_instruction')
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<guard>', 'exec'), ns)
    guard = ns['validate_reference_instruction']
    guard('Close the fridge door.', 'Close the fridge door.', 'Close the fridge door.')
    guard('Close the fridge door.', 'CloseFridge', 'Close the fridge door.', True)
    for args in [('Close the fridge door.', 'CloseFridge', 'Close the fridge door.'),
                 ('Close the fridge door.', 'OpenFridge', 'Close the fridge door.', True),
                 ('Close the fridge doors.', 'CloseFridge', 'Close the fridge door.', True)]:
        try:
            guard(*args)
        except AssertionError:
            continue
        raise AssertionError('Invalid contrast accepted')
    # Read cached training tables only; no dataset downloads, model or training invocation.
    import pandas as pd
    import pyarrow.parquet as pq
    tasks = pq.read_table(DATA / 'meta/tasks.parquet').to_pandas()
    frames = pq.read_table(DATA / 'data/chunk-000/file-000.parquet', columns=['episode_index', 'task_index', 'index', 'frame_index']).to_pandas()
    split = read(TRAIN / 'split.json')
    selected = frames[frames.episode_index.isin(split['train_episodes'])].reset_index(drop=True)
    updates = [json.loads(x) for x in (TRAIN / 'updates.jsonl').read_text().splitlines()]
    labels = []
    for update in updates:
        row = selected.iloc[update['sample_index']]
        assert int(row.episode_index) == update['episode']
        labels.append(str(tasks.iloc[int(row.task_index)].name))
    assert sha(ROOT / 'scripts/phase1d_train.py') == read(TRAIN / 'manifest.json')['script_sha256']
    save('finetuning-language-audit.json', {
        'dataset_revision': DATA.name, 'tasks': tasks.reset_index().to_dict(orient='records'),
        'all_frame_task_counts': {str(tasks.iloc[int(k)].name): int(v) for k, v in frames.task_index.value_counts().items()},
        'actual_300_update_task_counts': pd.Series(labels).value_counts().to_dict(),
        'update_tasks_in_order': labels, 'training_script_matches_manifest': True,
        'method': 'Reconstruct selected rows from saved split/sample indices; verify each logged episode. Installed reader resolves task_index through task table; trainer passes task unchanged to saved newline/tokenizer processor.',
        'limitation': 'Task strings reconstructed from immutable dataset/code, not independently logged training token IDs.'})
    plan = {'registered_unix': time.time(), 'model': 'lerobot/smolvla_robocasa', 'revision': REV,
            'seeds': [42, 43, 44], 'new_episodes': [43, 44], 'matrix': rows,
            'primary_outcome': 'Actual RoboCasa _check_success()', 'max_actions_per_episode': 600,
            'no_retries_or_expansion': True, 'runner_sha256': sha(ROOT / 'scripts/phase1d_rollout.py'),
            'randomness': 'Existing CPU/CUDA seed per scene immediately before policy execution; no per-chunk reseeding. New noise/RNG fingerprints only; reused arms lack fingerprints.',
            'known_outcomes': 'All four reused episodes failed; fixed consecutive seeds, not wholly unseen preregistration.'}
    save('preregistration.json', plan)
    # Keep generated Numba cache writes out of old Phase 1B artifacts.
    shutil.copytree(ROOT / 'outputs/phase1b/numba-cache', OUT / 'numba-cache')
    save('guard-checks.json', {'default_same_language_preserved': True, 'only_declared_contrast_accepted': True})
    print('Prepared fixed matrix and preservation inventory; no rollout executed.', flush=True)


def run():
    plan = read(OUT / 'preregistration.json')
    assert plan['new_episodes'] == [43, 44]
    assert sha(ROOT / 'scripts/phase1d_rollout.py') == plan['runner_sha256']
    assert inventory() == read(OUT / 'preservation-before.json'), 'Old artifacts changed since preparation'
    with (OUT / 'execution-started.json').open('x', encoding='utf-8') as f:
        json.dump({'started_unix': time.time(), 'new_episodes': [43, 44]}, f)
    for seed in [43, 44]:
        command = [sys.executable, '-B', '-u', str(ROOT / 'scripts/phase1d_rollout.py'),
                   '--output', str(task_id(seed)), '--seed', str(seed), '--instruction', 'CloseFridge',
                   '--checkpoint', 'lerobot/smolvla_robocasa', '--reference', str(natural(seed)), '--instruction-contrast']
        save(f'command-seed{seed}.json', command)
        print(f'Launching the one authorized task-ID episode for seed {seed}', flush=True)
        with (OUT / f'task-id-seed{seed}.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert result.returncode == 0, f'Seed {seed} stopped with error; preserve artifacts, do not retry'
        print(f'Seed {seed}: {read(task_id(seed) / "summary.json")["termination_reason"]}', flush=True)
    assert inventory() == read(OUT / 'preservation-before.json'), 'Old artifacts changed'
    save('preservation-after.json', {'all_old_files_unchanged': True, 'checked_files': len(read(OUT / 'preservation-before.json'))})


def approved_retry():
    """Explicit user-approved replacement 43 and still-unexecuted 44; no overwrites."""
    runner = ROOT / 'scripts/phase1d_rollout.py'
    source = runner.read_text(encoding='utf-8')
    tree = ast.parse(source)
    chunk = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'chunk')
    assert chunk.args.vararg.arg == 'chunk_args' and chunk.args.kwarg.arg == 'chunk_kwargs'
    assert not any(isinstance(n, ast.Name) and n.id == 'args' for n in ast.walk(chunk))
    assert 'policy.model.sample_noise =' not in source and 'sampling_fingerprint' not in source
    assert inventory() == read(OUT / 'preservation-before.json')
    failed = read(OUT / 'task-id-seed43/summary.json')
    assert failed['simulator_steps'] == 0 and failed['termination_reason'] == 'exception'
    assert not (OUT / 'task-id-seed44').exists()
    # Include all preceding probe artifacts, including the failed attempt, in preservation.
    old = {p.relative_to(ROOT).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns, sha(p)]
           for p in OUT.rglob('*') if p.is_file()}
    retry = OUT / 'approved-retry-1'
    retry.mkdir()  # Exclusive: rerunning this mode cannot launch another episode.
    (retry / 'preservation-before.json').write_text(json.dumps(old, indent=2), encoding='utf-8')
    amendment = {'authorized_new_attempts': [43, 44], 'runner_sha256': sha(runner),
                 'registered_unix': time.time(), 'original_preregistration_sha256': sha(OUT / 'preregistration.json'),
                 'optional_fingerprints': 'removed; sampler and seeds unchanged',
                 'wrapper_shadowing_check': 'passed: no args namespace access in chunk wrapper',
                 'failed_seed43_preserved': 'task-id-seed43', 'no_automatic_retries': True}
    (retry / 'amendment.json').write_text(json.dumps(amendment, indent=2), encoding='utf-8')
    shutil.copytree(OUT / 'numba-cache', retry / 'numba-cache')
    for seed in [43, 44]:
        target = retry / f'task-id-seed{seed}'
        assert not target.exists() and sha(runner) == amendment['runner_sha256']
        command = [sys.executable, '-B', '-u', str(runner), '--output', str(target),
                   '--seed', str(seed), '--instruction', 'CloseFridge', '--checkpoint', 'lerobot/smolvla_robocasa',
                   '--reference', str(natural(seed)), '--instruction-contrast']
        (retry / f'command-seed{seed}.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        print(f'Launching authorized seed {seed} in {target}', flush=True)
        with (retry / f'task-id-seed{seed}.log').open('x', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert completed.returncode == 0, f'Seed {seed} incomplete; no automatic retry'
        print(f'Seed {seed}: {read(target / "summary.json")["termination_reason"]}', flush=True)
    assert inventory() == read(OUT / 'preservation-before.json')
    for name, expected in old.items():
        p = ROOT / name
        assert [p.stat().st_size, p.stat().st_mtime_ns, sha(p)] == expected, name
    (retry / 'preservation-after.json').write_text(json.dumps({
        'historical_files_unchanged': True, 'prior_probe_files_unchanged': len(old)}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'run', 'approved-retry'], required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'approved-retry': approved_retry}[args.mode]()
