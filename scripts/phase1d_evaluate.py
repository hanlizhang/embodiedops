"""Serial paired evaluation; reuses the recorded natural-language seed-42 baseline.

Only runs a trained checkpoint after its separate-process reload verification.
Each child permits one episode, preserving the Phase 1C action queue and guards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'outputs/phase1c/retry-20261018'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def weight_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Checkpoint step directory')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/phase1d/evaluation')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    verified = read(checkpoint / 'reload-verification.json')
    assert verified['passed'] and verified['step'] == 300
    assert len(set(args.seeds)) == len(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)
    model = checkpoint / 'pretrained_model'
    digest = weight_hash(model / 'model.safetensors')
    config = {'seeds': args.seeds, 'trained_checkpoint': str(model), 'model_sha256': digest,
              'training_steps': verified['step'], 'instruction': 'environment native task_description for both checkpoints',
              'original_seed42_reused': str(ORIGINAL), 'success': 'actual RoboCasa _check_success()',
              'horizon': 600, 'guarded_early_stop': True, 'queue_actions': 50, 'flow_steps': 10}
    (args.output / 'evaluation-config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    def run(seed, label, reference=None):
        out = ORIGINAL if label == 'pretrained' and seed == 42 else args.output / f'{label}-seed{seed}'
        if not (out / 'summary.json').exists():
            command = [sys.executable, '-u', str(ROOT / 'scripts/phase1d_rollout.py'),
                       '--output', str(out), '--instruction', 'environment', '--seed', str(seed)]
            if reference:
                command += ['--checkpoint', str(model), '--reference', str(reference)]
            print(json.dumps({'launch': label, 'seed': seed, 'command': command}), flush=True)
            with (args.output / f'{label}-seed{seed}.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            assert completed.returncode == 0, f'{label} seed {seed} failed; inspect log and summary before continuing'
        summary = read(out / 'summary.json')
        assert summary['termination_reason'] in ['success', 'registered_horizon_600']
        assert not summary.get('exception')
        instruction = read(out / 'reset.json')['task']
        manifest = read(out / 'manifest.json')
        assert instruction == manifest.get('environment_instruction', 'Close the fridge doors.')
        if reference:
            assert read(out / 'paired-reset-validation.json')['passed']
            assert read(out / 'manifest.json')['model'] == str(model)
        final = summary['final_fridge']
        result = {'seed': seed, 'checkpoint': label, 'success': final['success'], 'instruction': instruction,
                  'door_degrees': {k: float(np.degrees(v)) for k, v in final['qpos'].items()},
                  'door_normalized': final['normalized'], 'actions': summary['simulator_steps'],
                  'inference_calls': summary['inference_calls'], 'physics_steps': summary['physics_steps'],
                  'simulated_seconds': summary['simulated_seconds'],
                  'wall_clock_rollout_seconds': summary['wall_clock_rollout_seconds'],
                  'mean_chunk_inference_seconds': float(np.mean(summary['inference_seconds'])),
                  'warm_chunk_inference_seconds': float(np.mean(summary['inference_seconds'][1:])),
                  'stop': summary['termination_reason'], 'exception': summary.get('exception'),
                  'video': str(out / 'rollout.mp4'), 'artifacts': str(out)}
        print(json.dumps(result), flush=True)
        return out, result

    rows = []
    for seed in args.seeds:
        baseline, before = run(seed, 'pretrained')
        _, after = run(seed, 'trained', baseline)
        rows += [before, after]
        report = {'config': config, 'episodes': rows,
                  'success_rates': {label: float(np.mean([r['success'] for r in rows if r['checkpoint'] == label]))
                                    for label in ['pretrained', 'trained']}}
        (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert weight_hash(model / 'model.safetensors') == digest


if __name__ == '__main__':
    main()
