"""Read-only historical integrity audit; writes only to a new study directory."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT/'outputs/phase1d/training-v2'
BASE = ROOT/'outputs/phase1a/hf-cache/hub/models--lerobot--smolvla_robocasa/snapshots/f7af588d11e30b0069f200792fb285904b74c80b'
PROCESSORS = ['policy_preprocessor.json', 'policy_postprocessor.json',
              'policy_preprocessor_step_5_normalizer_processor.safetensors',
              'policy_postprocessor_step_0_unnormalizer_processor.safetensors']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(p, x):
    p.write_text(json.dumps(x, indent=2), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    b = HISTORY/'checkpoints/000300'
    manifest = read(HISTORY/'manifest.json')
    assert sha(BASE/'model.safetensors') == manifest['base_model_sha256']
    assert sha(b/'pretrained_model/model.safetensors') == '0e011aa494e0d4e370b11feb5eec6e0c49af2fadd1f3f97ee03056909de0fe6d'
    for n in PROCESSORS:
        assert sha(BASE/n) == sha(b/'pretrained_model'/n) == manifest['processor_sha256'][n]
    target = read(b/'verification-target.json')
    with safe_open(b/'pretrained_model/model.safetensors', framework='pt', device='cpu') as f:
        actual = {n: hashlib.sha256(f.get_tensor(n).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n in f.keys()}
    assert actual == target['state_hashes'] and len(actual) == 500
    split = read(HISTORY/'split.json')
    updates = [json.loads(s) for s in (HISTORY/'updates.jsonl').read_text().splitlines()]
    assert len(updates) == 300
    assert [r['sample_index'] for r in updates] == split['training_sample_indices']
    assert not set(split['train_episodes']) & set(split['validation_episodes'])
    assert all(r['episode'] in split['train_episodes'] and r['finite'] for r in updates)
    protected = {str(p.relative_to(ROOT)): sha(p) for root in [ROOT/'outputs/phase1d', BASE]
                 for p in root.rglob('*') if p.is_file()}
    dump(args.output/'historical-before.json', protected)
    dump(args.output/'audit.json', dict(passed=True, time=time.time(), base_sha256=sha(BASE/'model.safetensors'),
         B_sha256=sha(b/'pretrained_model/model.safetensors'), verified_tensors=len(actual),
         processors=manifest['processor_sha256'], updates=300, train_episodes=len(split['train_episodes']),
         validation_episodes=len(split['validation_episodes']), optimizer=manifest['optimizer'],
         scheduler=manifest['scheduler'], protected_files=len(protected)))
    print('Historical checkpoint, 500 tensor hashes, saved processors and 300 anchors verified.', flush=True)

if __name__ == '__main__':
    main()
