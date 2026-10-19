"""Read-only historical/current training audit; writes only the Phase 2 audit directory."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    from safetensors import safe_open
    out = ROOT/'outputs/phase2/audit-v1'
    old = ROOT/'outputs/phase1d/training-v2'
    new = ROOT/'outputs/phase2/sanity-v1'

    def read(path):
        return json.loads(path.read_text())

    def sha(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()

    manifest = read(old/'manifest.json')
    source = ROOT/'scripts/phase1d_train.py'
    assert sha(source) == manifest['script_sha256']
    committed = subprocess.check_output(['git', 'show', 'HEAD:scripts/phase1d_train.py'], cwd=ROOT)
    assert committed.replace(b'\r\n', b'\n') == source.read_bytes().replace(b'\r\n', b'\n')
    split = read(old/'split.json')
    updates = [json.loads(s) for s in (old/'updates.jsonl').read_text().splitlines()]
    assert len(updates) == 300 and all(r['finite'] and not r['missing'] for r in updates)
    assert not set(split['train_episodes']) & set(split['validation_episodes'])
    cp = old/'checkpoints/000300/pretrained_model/model.safetensors'
    assert sha(cp) == '0e011aa494e0d4e370b11feb5eec6e0c49af2fadd1f3f97ee03056909de0fe6d'
    assert read(old/'checkpoints/000300/reload-verification.json')['passed']
    # Inspect stored Adam moments without loading a policy or executing a forward pass.
    # SmolVLA.get_optim_params returns ALL parameters, including frozen ones.
    # This architecture's saved state consists entirely of named parameters.
    names = list(read(old/'checkpoints/000300/verification-target.json')['state_hashes'])
    assert len(names) == 500
    norm = 'model.vlm_with_expert.lm_expert.layers.0.input_layernorm.weight'
    idx = names.index(norm)
    with safe_open(old/'checkpoints/000300/training_state/optimizer_state.safetensors', framework='pt', device='cpu') as f:
        keys = list(f.keys())
        chosen = [k for k in keys if k.startswith(f'state/{idx}/')]
        assert len(chosen) == 3
        moments = {k: {'shape': list(f.get_tensor(k).shape), 'dtype': str(f.get_tensor(k).dtype),
                       'max_abs': float(f.get_tensor(k).float().abs().max())} for k in chosen}
        m = f.get_tensor(f'state/{idx}/exp_avg').float()
        v = f.get_tensor(f'state/{idx}/exp_avg_sq').float()
        step_value = float(f.get_tensor(f'state/{idx}/step'))
    import torch
    with safe_open(cp, framework='pt', device='cpu') as f:
        weight = f.get_tensor(norm)
    assert weight.shape == m.shape and weight.dtype == torch.bfloat16
    b1, b2 = manifest['optimizer']['betas']
    nominal = updates[-1]['lr_used']*(m/(1-b1**step_value))/((v/(1-b2**step_value)).sqrt()+manifest['optimizer']['eps'])
    rounded = (weight.float()-nominal).to(weight.dtype)
    precision = {'weight_min': float(weight.min()), 'weight_max': float(weight.max()),
        'final_moment_based_fp32_update_max_abs': float(nominal.abs().max()),
        'elements_changed_if_that_update_is_rounded_to_bf16': int(torch.count_nonzero(rounded != weight)),
        'interpretation': 'final-moment FP32 estimate rounded to BF16; not an exact reconstruction of every historical fused arithmetic operation'}
    audit = read(new/'parameter-audit.json')
    new_updates = [json.loads(s) for s in (new/'updates.jsonl').read_text().splitlines()]
    new_split = read(new/'split.json')
    assert len(new_updates) == 512 and all(r['all_trainable_gradients_finite'] for r in new_updates)
    assert len(set(r['index'] for r in new_split['training_order'])) == 32
    assert not set(new_split['train_episodes']) & set(new_split['validation_episodes'])
    result = {'historical': {'committed_source_matches': True, 'source_sha256': sha(source),
        'checkpoint_sha256': sha(cp), 'reload_verified': True, 'updates': len(updates),
        'sampled_episodes': len(set(r['episode'] for r in updates)),
        'train_episodes': len(split['train_episodes']), 'val_episodes': len(split['validation_episodes']),
        'normalization_tensor_optimizer_index': idx, 'normalization_moments': moments,
        'normalization_precision_estimate': precision,
        'first_lr': updates[0]['lr_used'], 'final_lr': updates[-1]['lr_used'],
        'all_trainable_gradients_finite_and_present': True},
        'sanity': {'unique_training_frames': 32, 'unique_training_episodes': len(set(r['episode'] for r in new_split['training_order'])),
        'unique_validation_frames': len(set(r['index'] for r in new_split['validation_samples'])),
        'unique_validation_episodes': len(set(r['episode'] for r in new_split['validation_samples'])),
        'changed_tensors': len(audit['changed']), 'unchanged_trainable': len(audit['unchanged_trainable']),
        'all_frozen_unchanged': audit['all_frozen_unchanged'],
        'norm_probe_min_grad_norm': min(r['norm_probe_grad_norm'] for r in new_updates),
        'norm_probe_max_grad_norm': max(r['norm_probe_grad_norm'] for r in new_updates),
        'norm_probe_total_changed_elements_across_steps': sum(r['updates'][norm]['changed_elements'] for r in new_updates),
        'mean_update_seconds': float(np.mean([r['seconds'] for r in new_updates])),
        'first_64_raw_loss_mean': float(np.mean([r['loss'] for r in new_updates[:64]])),
        'last_64_raw_loss_mean': float(np.mean([r['loss'] for r in new_updates[-64:]]))}}
    (out/'training-audit.json').write_text(json.dumps(result, indent=2))
    # Verify every recorded historical file's bytes, size and modification time.
    inventory = read(out/'historical-before.json')
    differences = []
    for name, expected in inventory.items():
        p = ROOT/name
        actual = {'sha256': sha(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} if p.exists() else None
        if actual != expected:
            differences.append(name)
    (out/'preservation-after.json').write_text(json.dumps({'files_checked': len(inventory), 'differences': differences}, indent=2))
    assert not differences, differences
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
