"""One instruction-only 300-update candidate, gated by historical/reload integrity."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from instruction_control_audit import ROOT, HISTORY, BASE, PROCESSORS, read, dump, sha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    assert read(out/'audit.json')['passed']
    with (out/'training-started.json').open('x') as f:
        json.dump(dict(started=time.time(), updates=300, candidate_count=1), f)
    source = ROOT/'scripts/phase1d_train.py'
    shutil.copyfile(source, out/'training-source.py.txt')
    source_hash = sha(source)
    started = time.monotonic()

    def run(name, params, budget=600):
        assert sha(source) == source_hash
        assert shutil.disk_usage(out).free > 2*1024**3, 'Insufficient disk reserve'
        cmd = [sys.executable, '-B', '-u', str(source), '--dataset-cache', str(ROOT/'outputs/ic-cache')] + params
        dump(out/f'{name}-command.json', cmd)
        print(f'Starting {name} (cap {budget}s)', flush=True)
        t = time.monotonic()
        with (out/f'{name}.log').open('x') as log:
            p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=budget)
        elapsed = time.monotonic()-t
        dump(out/f'{name}-worker.json', dict(returncode=p.returncode, seconds=elapsed, budget=budget))
        assert p.returncode == 0 and elapsed <= budget, f'{name} failed; no automatic retry'
        assert time.monotonic()-started <= 1800

    try:
        run('verify-B', ['--mode', 'verify', '--checkpoint', str(HISTORY/'checkpoints/000300'),
             '--output', str(out/'B-reload'), '--verify-output', str(out/'B-reload-verification.json')])
        candidate = out/'C'
        common = ['--output', str(candidate), '--task-id']
        run('C-smoke', ['--mode', 'smoke']+common)
        run('C-verify-1', ['--mode', 'verify', '--checkpoint', str(candidate/'checkpoints/000001')]+common)
        # Verify actual Torch RNG equality across the original and new smoke step.
        import torch
        from safetensors.torch import load_file
        a = load_file(HISTORY/'checkpoints/000001/training_state/rng_state.safetensors')
        b = load_file(candidate/'checkpoints/000001/training_state/rng_state.safetensors')
        rng = {k: torch.equal(a[k], b[k]) for k in a if k.startswith('torch')}
        dump(out/'smoke-rng-comparison.json', rng)
        assert rng and all(rng.values()), 'Training RNG differs; controlled continuation stopped'
        run('C-continue', ['--mode', 'continue', '--checkpoint', str(candidate/'checkpoints/000001')]+common)
        run('C-verify-300', ['--mode', 'verify', '--checkpoint', str(candidate/'checkpoints/000300')]+common)
        original = [json.loads(s) for s in (HISTORY/'updates.jsonl').read_text().splitlines()]
        current = [json.loads(s) for s in (candidate/'updates.jsonl').read_text().splitlines()]
        assert len(current) == 300
        for a, b in zip(original, current):
            assert all(a[k] == b[k] for k in ['step', 'sample_index', 'episode', 'lr_used'])
        dump(out/'training-complete.json', dict(passed=True, seconds=time.monotonic()-started,
             C_sha256=sha(candidate/'checkpoints/000300/pretrained_model/model.safetensors')))
    except Exception as e:
        dump(out/'training-stopped.json', dict(reason=repr(e), seconds=time.monotonic()-started))
        raise
    finally:
        old = read(out/'historical-before.json')
        changed = [p for p,h in old.items() if sha(ROOT/p) != h]
        dump(out/'preservation.json', dict(passed=not changed, changed=changed, files=len(old)))

if __name__ == '__main__':
    main()
