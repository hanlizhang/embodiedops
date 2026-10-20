"""Versioned, offline SmolVLA conditioning experiment; never creates a simulator.

Use --config and an exclusive --output for training. --verify reloads a saved
checkpoint in a separate process. Only the disposable --preflight-steps mode
also checks one resumed optimizer update; sustained-checkpoint verification
does not update weights.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODEL_REV = 'f7af588d11e30b0069f200792fb285904b74c80b'
DATA_REV = 'b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4'
MODEL = ROOT / 'outputs/phase1a/hf-cache/hub/models--lerobot--smolvla_robocasa/snapshots' / MODEL_REV
DATA = ROOT / 'outputs/phase1a/hf-cache/lerobot/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots' / DATA_REV
PROCESSORS = ['policy_preprocessor.json', 'policy_postprocessor.json',
              'policy_preprocessor_step_5_normalizer_processor.safetensors',
              'policy_postprocessor_step_0_unnormalizer_processor.safetensors']


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def plot(out):
    """Dependency-free standalone SVG of actual measured losses."""
    rows = [json.loads(s) for s in (out/'updates.jsonl').read_text().splitlines()]
    probes = [json.loads(s) for s in (out/'probes.jsonl').read_text().splitlines()]
    train_key = 'task_id' if json.loads((out/'settings.json').read_text())['condition']=='task_id' else 'natural'
    series = [('train EMA .05', '#777', [(r['step'], r['loss_ema']) for r in rows]),
              ('fixed train '+train_key, '#15803d', [(r['step'], r['train'][train_key]['mean']) for r in probes]),
              ('validation NL', '#2563eb', [(r['step'], r['validation']['natural']['mean']) for r in probes]),
              ('validation task ID', '#c2410c', [(r['step'], r['validation']['task_id']['mean']) for r in probes])]
    xmax = max(r['step'] for r in rows)
    ymax = max(y for _, _, points in series for _, y in points)*1.1
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">',
           '<rect width="960" height="540" fill="white"/>',
           '<text x="70" y="30" font-size="20">Measured flow-matching loss (not task success)</text>']
    for i in range(6):
        y = 440-i*72
        svg.append(f'<path d="M70 {y}H920" stroke="#ddd"/><text x="10" y="{y+5}">{ymax*i/5:.3f}</text>')
        svg.append(f'<text x="{70+i*170}" y="465">{xmax*i/5:.0f}</text>')
    for i, (label, color, points) in enumerate(series):
        coords = ' '.join(f'{70+x/xmax*850:.2f},{440-y/ymax*360:.2f}' for x, y in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{70+(i%2)*420}" y="{495+(i//2)*24}" fill="{color}">{label}</text>')
    svg.append('</svg>')
    (out/'losses.svg').write_text('\n'.join(svg), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--preflight-steps', type=int, default=0,
                        help='Master-precision only: eight disposable updates, same full schedule')
    parser.add_argument('--preflight-evidence', type=Path)
    parser.add_argument('--dataset-cache', type=Path,
                        help='Optional short generated-cache path to avoid Windows lock filename limits')
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    master_mode = settings.get('optimizer_precision') == 'fp32_master'
    assert args.preflight_steps in (0, 8)
    if args.preflight_steps:
        assert master_mode
    if master_mode and not args.preflight_steps and not args.verify:
        assert args.preflight_evidence and json.loads(args.preflight_evidence.read_text())['passed']
    out = args.output.resolve()
    # Training refuses an existing directory; verification uses an exclusive marker.
    if args.verify:
        assert json.loads((out/'settings.json').read_text()) == settings
        verify_marker = f'verify-{args.verify.name}-started.json' if settings.get('multiple_reload_checks') else 'verify-started.json'
        with (out/verify_marker).open('x') as f:
            json.dump({'checkpoint': str(args.verify)}, f)
    else:
        out.mkdir(parents=True, exist_ok=False)
        dump(out/'settings.json', settings)
    cache = args.dataset_cache.resolve() if args.dataset_cache else out/'datasets-cache'
    assert cache.is_relative_to(ROOT/'outputs'), 'Generated cache must stay in project outputs'
    os.environ.update(HF_HOME=str(ROOT/'outputs/phase1a/hf-cache'),
                      HF_DATASETS_CACHE=str(cache), HF_HUB_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import numpy as np
    import torch
    from accelerate import Accelerator
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_train_eval_datasets
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    from lerobot.common.train_utils import save_checkpoint, load_training_state
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.collate import lerobot_collate_fn
    from lerobot.utils.logging_utils import MetricsTracker, AverageMeter

    assert md.version('lerobot') == '0.6.1' and torch.cuda.is_available()
    assert settings['condition'] in ('natural', 'mixed', 'task_id')
    assert 64 <= settings['validation_samples'] <= 128
    assert 0 < settings['warmup_steps'] < settings['steps']
    torch.manual_seed(settings['seed'])
    np.random.seed(settings['seed'])
    torch.backends.cudnn.benchmark = False
    # Warn rather than silently substitute kernels: cross-device bit identity is not promised.
    torch.use_deterministic_algorithms(True, warn_only=True)
    pc = SmolVLAConfig.from_pretrained(str(MODEL))
    pc.freeze_vision_encoder = pc.train_expert_only = pc.train_state_proj = True
    pc.pretrained_path = str(MODEL)
    pc.push_to_hub = False
    assert not pc.use_amp and pc.chunk_size == pc.n_action_steps == 50
    opt_cfg = pc.get_optimizer_preset()
    opt_cfg.lr = settings['learning_rate']
    mapping = json.loads((MODEL/PROCESSORS[0]).read_text())['steps'][0]['config']['rename_map']
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id='pepijn223/robocasa_CloseFridge',
        root=str(DATA), revision=DATA_REV, video_backend='pyav', use_imagenet_stats=False, eval_split=.1),
        policy=pc, output_dir=out, seed=settings['seed'], batch_size=1, num_workers=0,
        steps=settings['steps'], env_eval_freq=0, max_eval_samples=settings['validation_samples'],
        eval_steps=settings['probe_interval'], save_freq=settings['checkpoint_interval'],
        rename_map=mapping, optimizer=opt_cfg, scheduler=CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=settings['warmup_steps'], num_decay_steps=settings['steps'],
            peak_lr=settings['learning_rate'], decay_lr=settings['end_learning_rate']))
    if not args.verify:
        cfg.save_pretrained(out/'config')
    train, val = make_train_eval_datasets(cfg)
    assert val is not None and not set(train.episodes) & set(val.episodes)
    assert set(train.episodes) | set(val.episodes) == set(range(513))

    def sample_plan(dataset, count, seed):
        episodes = np.asarray(dataset.hf_dataset.with_format(None)['episode_index'])
        groups = {int(e): np.flatnonzero(episodes == e) for e in np.unique(episodes)}
        rng = np.random.default_rng(seed)
        plan = []
        while len(plan) < count:
            for e in rng.permutation(sorted(groups)):
                idx = int(rng.choice(groups[e]))
                plan.append({'index': idx, 'episode': int(e)})
                if len(plan) == count:
                    break
        return plan

    train_plan = sample_plan(train, max(settings['steps'], 32), settings['seed'])
    train_probe_plan = train_plan[:32]
    if 'sampling' in settings:
        from learning_sampling import build_plan, dataset_labels, LABELS
        episodes, labels = dataset_labels(train)
        train_plan = build_plan(episodes, settings['steps'], settings['seed'], settings['sampling'], labels)
        # Shared independent probe anchors, regardless of the training sampling arm.
        train_probe_plan = build_plan(episodes, 32, 31415)
        sampling_audit = {'regime_counts': {name: int((labels==i).sum()) for i,name in enumerate(LABELS)},
                          'labels_are_contacts': False,
                          'selected_regime_counts': {name: int(sum(labels[r['index']]==i for r in train_plan)) for i,name in enumerate(LABELS)},
                          'sampler_sha256': sha(ROOT/'scripts/learning_sampling.py')}
        if not args.verify: dump(out/'sampling-audit.json', sampling_audit)
    if settings['sanity']:
        train_plan = [train_plan[i % 32] for i in range(settings['steps'])]
    val_plan = sample_plan(val, settings['validation_samples'], settings['seed']+1)
    split = {'train_episodes': train.episodes, 'validation_episodes': val.episodes,
             'train_frames': len(train), 'validation_frames': len(val),
             'training_order': train_plan, 'validation_samples': val_plan,
             'training_probe': train_probe_plan, 'pretraining_overlap_unknown': True,
             'rule': 'last ceil(10%) episodes per natural instruction; sample one frame per shuffled episode per cycle'}
    baseline_folder = ROOT/'outputs/agent-workflow-v1/learning-5000-v1/episode_balanced'
    if master_mode:
        original_settings = json.loads((baseline_folder/'settings.json').read_text())
        assert all(settings[k] == v for k, v in original_settings.items())
        assert split == json.loads((baseline_folder/'split.json').read_text())
        baseline_updates = [json.loads(r) for r in (baseline_folder/'updates.jsonl').read_text().splitlines()]
    if args.verify:
        assert json.loads((out/'split.json').read_text()) == split
    else:
        dump(out/'split.json', split)
    source = args.verify/'pretrained_model' if args.verify else MODEL
    policy = SmolVLAPolicy.from_pretrained(str(source), config=None if args.verify else pc, strict=True)
    assert policy.model.vlm_with_expert.config._commit_hash == '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(source))
    assert pre.steps[0].get_config()['rename_map'] == mapping
    intended = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    assert len(intended) == 155 and not any('.vlm.' in n for n in intended)
    assert sum(p.numel() for p in intended.values()) == 99880992
    accelerator = Accelerator(mixed_precision='no', step_scheduler_with_optimizer=False)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)

    def digest(value):
        if isinstance(value, torch.Tensor):
            data = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': hashlib.sha256(data).hexdigest()}
        if isinstance(value, dict):
            return {str(k): digest(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [digest(v) for v in value]
        return value

    master = None
    if master_mode:
        from fp32_master import FP32Masters
        from torch.nn.utils import parametrize
        native_before = digest(policy.state_dict())
        master = FP32Masters(policy)
        assert len(master.masters) == 129
        assert all(p.dtype == torch.float32 for p in intended.values())
        assert digest(policy.state_dict()) == native_before
        original_forward = policy.forward
        def native_forward(*a, **kw):
            with parametrize.cached():
                return original_forward(*a, **kw)
        policy.forward = native_forward
        if args.verify:
            master.load(args.verify/'training_state/fp32_masters.safetensors')
        else:
            assert native_before == json.loads((baseline_folder/'parameters-before.json').read_text())

    def prep(raw, task_id=False):
        batch = lerobot_collate_fn([raw])
        assert raw['task'] in ('Close the fridge door.', 'Close the fridge doors.')
        if task_id:
            batch['task'] = ['CloseFridge']
        for cam in train.meta.camera_keys:
            if batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].float()/255
        batch = pre(batch)
        assert batch['observation.state'].shape == (1, 1, 16)
        assert batch['action'].shape == (1, 50, 12) and batch['action_is_pad'].shape == (1, 50)
        assert (~batch['action_is_pad']).any()
        assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v, torch.Tensor))
        for cam in mapping.values():
            assert batch[cam].shape == (1, 3, 256, 256) and batch[cam].min() >= 0 and batch[cam].max() <= 1
        return batch

    train_raw = [train[r['index']] for r in train_probe_plan]
    val_raw = [val[r['index']] for r in val_plan]

    def evaluate(raws, seed):
        result = {}
        policy.eval()
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            for name in ('natural', 'task_id'):
                losses, draws = [], []
                for i, raw in enumerate(raws):
                    torch.manual_seed(seed+i)
                    noise = policy.model.sample_noise((1, 50, policy.config.max_action_dim), policy.config.device)
                    t = policy.model.sample_time(1, policy.config.device)
                    loss, _ = policy.forward(prep(raw, name == 'task_id'), noise=noise, time=t)
                    assert torch.isfinite(loss)
                    losses.append(loss.item())
                    draws.append({'noise': digest(noise), 'time': digest(t), 'seed': seed+i})
                result[name] = {'mean': float(np.mean(losses)), 'samples': losses, 'draws': draws}
        assert result['natural']['draws'] == result['task_id']['draws']
        result['selection_mean'] = (result['task_id']['mean'] if settings.get('selection_metric')=='task_id' else
                                    (result['natural']['mean']+result['task_id']['mean'])/2)
        return result

    if args.verify:
        target = json.loads((args.verify/'verification-target.json').read_text())
        assert digest(policy.state_dict()) == target['model']
        step, optimizer, scheduler = load_training_state(args.verify, optimizer, scheduler)
        assert step == target['step']
        assert digest(optimizer.state_dict()) == target['optimizer']
        assert digest(scheduler.state_dict()) == target['scheduler']
        if master:
            assert digest(master.state()) == target['masters']
        actual = evaluate(val_raw, 82000)
        for arm in ('natural', 'task_id'):
            assert actual[arm]['draws'] == target['validation'][arm]['draws']
            assert np.allclose(actual[arm]['samples'], target['validation'][arm]['samples'], rtol=1e-5, atol=1e-6)
        assert all(sha(source/n) == sha(MODEL/n) for n in PROCESSORS)
        continuation_verified = None
        if args.preflight_steps:
            # Verify a real resumed update against a separately saved next-step
            # target. This checkpoint is disposable and never initializes training.
            expected = json.loads((out/'preflight-next-update.json').read_text())
            batch = prep(train[train_plan[step]['index']], True)
            meters = {k: AverageMeter(k, ':.6f') for k in ('loss','grad_norm','lr','update_s','gpu_mem_gb')}
            tracker = MetricsTracker(1,len(train),len(train.episodes),meters,accelerator=accelerator)
            torch.manual_seed(100000+step)
            tracker, _ = update_policy(tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, accelerator, scheduler)
            assert digest(policy.state_dict()) == expected['model']
            assert digest(master.state()) == expected['masters']
            assert digest(optimizer.state_dict()) == expected['optimizer']
            assert digest(scheduler.state_dict()) == expected['scheduler']
            continuation_verified = True
        reload_name = f'reload-{step:06d}.json' if settings.get('multiple_reload_checks') else 'reload-verification.json'
        dump(out/reload_name, {'passed': True, 'step': step,
            'model_tensors_equal': len(target['model']), 'optimizer_state_equal': True,
            'scheduler_state_equal': True, 'validation_equal': True, 'processors_byte_identical': True,
            'master_parameters_equal': bool(master), 'resumed_update_equal': continuation_verified,
            'checkpoint_sha256': sha(source/'model.safetensors')})
        print('Model, optimizer, scheduler, fixed validation and processors reload verified.', flush=True)
        return

    before = digest(policy.state_dict())
    dump(out/'manifest.json', {'model': 'lerobot/smolvla_robocasa', 'model_revision': MODEL_REV,
        'dataset': 'pepijn223/robocasa_CloseFridge', 'dataset_revision': DATA_REV,
        'base_sha256': sha(MODEL/'model.safetensors'), 'script_sha256': sha(Path(__file__)),
        'settings_sha256': sha(args.config), 'processor_hashes': {n: sha(MODEL/n) for n in PROCESSORS},
        'packages': {n: md.version(n) for n in ('torch', 'lerobot', 'accelerate', 'numpy', 'av')},
        'device': torch.cuda.get_device_name(), 'trainable_parameters': 99880992,
        'precision': ('native BF16/FP32 forward; differentiable FP32 masters, gradients and Adam moments'
                      if master else 'unchanged native BF16/FP32; no AMP, PEFT or gradient checkpointing'),
        'master_adapter_sha256': sha(ROOT/'scripts/fp32_master.py') if master else None,
        'train_noise': 'fresh per update; torch.manual_seed(100000 + update index), identical across A/B',
        'validation_noise': 'explicit noise/time with seeds 82000+i; both language arms share draws',
        'determinism': 'deterministic algorithms warn_only; cross-hardware bit identity not guaranteed'})
    dump(out/'parameters-before.json', before)
    target_name = 'model.action_out_proj.weight'
    norm_name = 'model.vlm_with_expert.lm_expert.layers.0.input_layernorm.weight'
    meters = {k: AverageMeter(k, ':.6f') for k in ('loss', 'grad_norm', 'lr', 'update_s', 'gpu_mem_gb')}
    tracker = MetricsTracker(1, len(train), len(train.episodes), meters, accelerator=accelerator)
    grad = {}

    def check_grad(opt, hook_args, hook_kwargs):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in intended.values())
        assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
        grad.update(all_trainable_gradients_finite=True, tensors_with_grad=len(intended),
                    norm_probe_grad_norm=float(intended[norm_name].grad.float().norm()))
        if master:
            assert all(p.grad.dtype == torch.float32 for p in intended.values())

    handle = optimizer.register_step_pre_hook(check_grad)
    started = time.perf_counter()
    step, ema, best = 0, None, None
    peak, reserved = 0, 0

    def probe():
        record = {'step': step, 'train': evaluate(train_raw, 72000), 'validation': evaluate(val_raw, 82000)}
        with (out/'probes.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps({'probe_step': step, 'train_nl': record['train']['natural']['mean'],
            'val_nl': record['validation']['natural']['mean'], 'val_id': record['validation']['task_id']['mean']}), flush=True)
        return record

    try:
        initial = probe()
        if master:
            original_probe = json.loads((baseline_folder/'probes.jsonl').read_text().splitlines()[0])
            for part in ('train','validation'):
                for language in ('natural','task_id'):
                    assert initial[part][language]['draws'] == original_probe[part][language]['draws']
                    assert np.allclose(initial[part][language]['samples'], original_probe[part][language]['samples'], rtol=1e-5, atol=1e-6)
            dump(out/'initial-contract.json', {'passed':True,'sample_order_identical':True,
                 'original_forward_weights_and_dtypes_equal':True,'fixed_initial_losses_equal':True,
                 'gradient_path':'loss -> native forward cast -> FP32 master leaf -> AdamW',
                 'clipping_note':'Same global threshold10; gradient accumulation and norm arithmetic now FP32'})
        for index, row in enumerate(train_plan):
            raw = train_raw[index % 32] if settings['sanity'] else train[row['index']]
            task_id = settings['condition'] == 'task_id' or (settings['condition'] == 'mixed' and index % 2 == 1)
            batch = prep(raw, task_id)
            assert int(batch['episode_index'].item()) == row['episode']
            old = {n: intended[n].detach().clone() for n in (target_name, norm_name)}
            old_native = {n: master.forward_weight(n).detach().clone() for n in old} if master else {}
            lr = optimizer.param_groups[0]['lr']
            if master:
                assert row == {k:baseline_updates[index][k] for k in row}
                assert lr == baseline_updates[index]['lr_used']
            torch.manual_seed(100000+index)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            tracker, _ = update_policy(tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, accelerator, scheduler)
            torch.cuda.synchronize()
            step = index+1
            assert np.isfinite(tracker.loss.val) and np.isfinite(tracker.grad_norm.val)
            changes = {n: {'changed_elements': int(torch.count_nonzero(intended[n].detach()-v)),
                           'max_abs_delta': float((intended[n].detach().float()-v.float()).abs().max())} for n, v in old.items()}
            assert changes[target_name]['changed_elements'] > 0
            native_changes = {}
            if master:
                for n, v in old_native.items():
                    d = master.forward_weight(n).detach().float()-v.float()
                    native_changes[n] = {'changed_elements':int(torch.count_nonzero(d)),
                                         'max_abs_delta':float(d.abs().max())}
                assert all(optimizer.state[p]['exp_avg'].dtype == torch.float32 and
                           optimizer.state[p]['exp_avg_sq'].dtype == torch.float32 for p in intended.values())
                if step == 1:
                    assert np.isclose(tracker.loss.val, baseline_updates[0]['loss'], rtol=1e-5, atol=1e-6)
            ema = tracker.loss.val if ema is None else .05*tracker.loss.val+.95*ema
            peak = max(peak, torch.cuda.max_memory_allocated())
            reserved = max(reserved, torch.cuda.max_memory_reserved())
            record = {'step': step, **row, 'task': 'CloseFridge' if task_id else raw['task'],
                'loss': tracker.loss.val, 'loss_ema': ema, 'grad_norm': tracker.grad_norm.val,
                'lr_used': lr, 'seconds': time.perf_counter()-tick,
                'peak_allocated': peak, 'peak_reserved': reserved, 'updates': changes,
                'native_forward_updates':native_changes, 'gradient_clipped':tracker.grad_norm.val>cfg.optimizer.grad_clip_norm,
                **grad}
            with (out/'updates.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if step % 32 == 0:
                print(json.dumps({'step': step, 'loss': record['loss'], 'ema': ema, 'lr': lr}), flush=True)
            save = step % settings['checkpoint_interval'] == 0 or step == settings['steps'] or step == args.preflight_steps
            if step % settings['probe_interval'] == 0 or step in settings.get('extra_probe_steps',[]) or save:
                measured = probe()
                plot(out)
            if save:
                path = out/'checkpoints'/f'{step:06d}'
                assert not path.exists()
                save_checkpoint(path, step, cfg, policy, optimizer, scheduler, pre, post, num_processes=1, batch_size=1)
                for n in PROCESSORS:
                    shutil.copyfile(MODEL/n, path/'pretrained_model'/n)
                if master:
                    master.save(path/'training_state/fp32_masters.safetensors')
                dump(path/'verification-target.json', {'step': step, 'model': digest(policy.state_dict()),
                    'optimizer': digest(optimizer.state_dict()), 'scheduler': digest(scheduler.state_dict()),
                    'validation': measured['validation'], 'masters':digest(master.state()) if master else None})
                score = measured['validation']['selection_mean']
                if best is None or score < best['score']:
                    best = {'step': step, 'score': score, 'checkpoint': str(path.relative_to(out))}
                dump(out/'selected-checkpoint.json', {**best, 'rule': 'lowest '+settings.get('selection_metric','mean NL/ID')+' validation loss among saved checkpoints; earliest tie',
                    'pretrained_score': initial['validation']['selection_mean']})
            if step == args.preflight_steps:
                assert master and any(json.loads(r)['updates'][norm_name]['changed_elements'] > 0
                                      for r in (out/'updates.jsonl').read_text().splitlines())
                batch = prep(train[train_plan[step]['index']], True)
                torch.manual_seed(100000+step)
                tracker, _ = update_policy(tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, accelerator, scheduler)
                dump(out/'preflight-next-update.json', {'model':digest(policy.state_dict()),
                     'masters':digest(master.state()), 'optimizer':digest(optimizer.state_dict()),
                     'scheduler':digest(scheduler.state_dict())})
                dump(out/'preflight-local.json', {'passed':True,'saved_updates':step,'extra_continuation_update':1,
                     'finite_fp32_master_gradients':True,'norm_master_accumulation_nonzero':True,
                     'quantized_forward_updates_recorded_separately':True,'peak_allocated':peak})
                return
        after = digest(policy.state_dict())
        changed = [n for n in before if before[n] != after[n]]
        assert all(n in intended for n in changed)
        unchanged = [n for n in intended if before[n] == after[n]]
        dump(out/'parameter-audit.json', {'changed': changed, 'unchanged_trainable': unchanged,
            'all_frozen_unchanged': True, 'optimizer_state_dtypes': {
                n: {k: str(v.dtype) for k, v in optimizer.state[p].items() if isinstance(v, torch.Tensor)}
                for n, p in intended.items()}})
        dump(out/'summary.json', {'completed_updates': step, 'seconds_including_probes_and_checkpoints': time.perf_counter()-started,
            'peak_allocated': peak, 'peak_reserved': reserved, 'initial_train_nl': initial['train']['natural']['mean'],
            'final_train_nl': measured['train']['natural']['mean'], 'initial_validation': initial['validation']['selection_mean'],
            'final_validation': measured['validation']['selection_mean'], 'selected': best,
            'sanity_loss_reduced': measured['train']['natural']['mean'] < initial['train']['natural']['mean'],
            'exception': None})
    except Exception:
        dump(out/'failure.json', {'completed_updates': step, 'exception': traceback.format_exc()})
        raise
    finally:
        handle.remove()


if __name__ == '__main__':
    main()
