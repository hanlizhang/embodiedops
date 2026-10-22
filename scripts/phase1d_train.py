"""Bounded SmolVLA training using LeRobot 0.6.1 training primitives.

Preserves the pinned pretrained processors instead of the CLI's dataset-stat
override. No environment is created. Smoke, reload verification and continuation
are separate processes so checkpoint validation gates sustained training.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase1d/training'
os.environ['HF_HOME'] = str(ROOT / 'outputs/phase1a/hf-cache')
os.environ['HF_DATASETS_CACHE'] = str(ROOT / 'outputs/phase1d/datasets-cache')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
MODEL = 'lerobot/smolvla_robocasa'
MODEL_REV = 'f7af588d11e30b0069f200792fb285904b74c80b'
DATASET = 'pepijn223/robocasa_CloseFridge'
DATA_REV = 'b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4'
BACKBONE_REV = '7b375e1b73b11138ff12fe22c8f2822d8fe03467'
DATA_ROOT = ROOT / 'outputs/phase1a/hf-cache/lerobot/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots' / DATA_REV
MODEL_ROOT = ROOT / 'outputs/phase1a/hf-cache/hub/models--lerobot--smolvla_robocasa/snapshots' / MODEL_REV
PROCESSORS = ['policy_preprocessor.json', 'policy_postprocessor.json',
              'policy_preprocessor_step_5_normalizer_processor.safetensors',
              'policy_postprocessor_step_0_unnormalizer_processor.safetensors']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def filehash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'verify', 'continue'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--task-id', action='store_true', help='Controlled instruction-only arm: CloseFridge before saved preprocessing')
    parser.add_argument('--verify-output', type=Path, help='Write verification outside a historical checkpoint')
    parser.add_argument('--dataset-cache', type=Path)
    args = parser.parse_args()
    if args.dataset_cache:
        os.environ['HF_DATASETS_CACHE'] = str(args.dataset_cache.resolve())
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / (args.mode + ('-' + args.checkpoint.name if args.checkpoint else '') + '-started.json')
    with marker.open('x', encoding='utf-8') as f:
        json.dump({'started_unix': time.time(), 'mode': args.mode}, f)

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
    from lerobot.common.train_utils import save_checkpoint, load_training_state
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.collate import lerobot_collate_fn
    from lerobot.utils.logging_utils import MetricsTracker, AverageMeter

    assert md.version('lerobot') == '0.6.1'
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    np.random.seed(42)
    cfg_policy = SmolVLAConfig.from_pretrained(str(MODEL_ROOT))
    # Supported configuration switches only; architecture and inference settings retained.
    cfg_policy.freeze_vision_encoder = True
    cfg_policy.train_expert_only = True
    cfg_policy.train_state_proj = True
    cfg_policy.pretrained_path = str(MODEL_ROOT)
    cfg_policy.push_to_hub = False
    assert cfg_policy.use_amp is False and cfg_policy.chunk_size == cfg_policy.n_action_steps == 50
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET, root=str(DATA_ROOT), revision=DATA_REV,
                              video_backend='pyav', use_imagenet_stats=False, eval_split=.1),
        policy=cfg_policy, output_dir=OUT, seed=42, batch_size=1, num_workers=0,
        steps=300, env_eval_freq=0, eval_steps=50, max_eval_samples=8,
        log_freq=10, save_freq=150, rename_map=json.loads((MODEL_ROOT/PROCESSORS[0]).read_text())['steps'][0]['config']['rename_map'],
        optimizer=cfg_policy.get_optimizer_preset(), scheduler=cfg_policy.get_scheduler_preset())
    cfg.save_pretrained(OUT/'config')
    train, val = make_train_eval_datasets(cfg)
    assert val is not None and not set(train.episodes) & set(val.episodes)
    assert set(train.episodes) | set(val.episodes) == set(range(513))
    train_indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(42))[:cfg.steps].tolist()
    val_indices = np.linspace(0, len(val)-1, 8, dtype=int).tolist()
    split = {'train_episodes': train.episodes, 'validation_episodes': val.episodes,
             'train_frames': len(train), 'validation_frames': len(val),
             'training_sample_indices': train_indices, 'validation_sample_indices': val_indices,
             'split_rule': 'LeRobot last ceil(10%) episodes per instruction; no episode overlap',
             'pretraining_overlap_unknown': True}
    dump(OUT/'split.json', split)
    if args.task_id:
        historical_split = json.loads((ROOT/'outputs/phase1d/training-v2/split.json').read_text())
        assert split == historical_split, 'Instruction study must retain original samples and split'

    source = str(args.checkpoint/'pretrained_model') if args.checkpoint else str(MODEL_ROOT)
    policy = SmolVLAPolicy.from_pretrained(source, config=None if args.checkpoint else cfg_policy, strict=True)
    assert policy.config.freeze_vision_encoder and policy.config.train_expert_only and policy.config.train_state_proj
    assert policy.config.chunk_size == 50 and policy.config.n_action_steps == 50 and not policy.config.use_amp
    assert policy.model.vlm_with_expert.config._commit_hash == BACKBONE_REV
    pre, post = make_pre_post_processors(policy.config, pretrained_path=source)
    expected_mapping = cfg.rename_map
    assert pre.steps[0].get_config()['rename_map'] == expected_mapping
    intended = {n:p for n,p in policy.named_parameters() if p.requires_grad}
    assert intended and not any('.vlm.' in n for n in intended)
    assert any('lm_expert' in n for n in intended) and 'model.state_proj.weight' in intended
    target_name = 'model.action_out_proj.weight'
    assert target_name in intended
    accelerator = Accelerator(mixed_precision='no', step_scheduler_with_optimizer=False)

    def prep(raw):
        if args.task_id:
            raw = dict(raw, task='CloseFridge')
        batch = lerobot_collate_fn([raw])
        for cam in train.meta.camera_keys:
            if batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].float()/255.0
        batch = pre(batch)
        # LeRobot retains the state time axis, but decodes a single video frame
        # without it. SmolVLA.prepare_images explicitly supports this BCHW form.
        assert batch['observation.state'].shape == (1,1,16) and batch['action'].shape == (1,50,12)
        assert batch['action_is_pad'].shape == (1,50)
        assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v,torch.Tensor))
        for cam in expected_mapping.values():
            assert batch[cam].shape == (1,3,256,256) and batch[cam].min()>=0 and batch[cam].max()<=1
        return batch

    # Fixed observations and RNG draws make before/after validation losses comparable.
    val_raw = [val[i] for i in val_indices]
    train_probe = [train[i] for i in train_indices[:8]]
    def evaluate(raws):
        losses=[]
        policy.eval()
        with torch.random.fork_rng(devices=[0]), torch.no_grad():
            for i, raw in enumerate(raws):
                torch.manual_seed(42000+i)
                loss,_=policy.forward(prep(raw))
                assert torch.isfinite(loss)
                losses.append(loss.item())
        return {'mean':float(np.mean(losses)), 'samples':losses, 'noise_seeds':list(range(42000,42000+len(raws)))}

    def hashes():
        return {n:hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                for n,p in policy.state_dict().items()}

    if args.mode == 'verify':
        assert args.checkpoint
        saved = json.loads((args.checkpoint/'verification-target.json').read_text())
        actual_hashes=hashes()
        assert actual_hashes == saved['state_hashes'], 'Checkpoint parameters changed on reload'
        loss=evaluate(val_raw)
        assert np.allclose(loss['samples'],saved['validation']['samples'],rtol=1e-5,atol=1e-6), 'Reload loss mismatch'
        for name in PROCESSORS:
            assert filehash(args.checkpoint/'pretrained_model'/name)==filehash(MODEL_ROOT/name)
        result={'passed':True,'checkpoint':str(args.checkpoint),'step':saved['step'],
                'state_tensors_equal':len(actual_hashes),'validation':loss,'processors_byte_identical':True}
        dump(args.verify_output or args.checkpoint/'reload-verification.json',result)
        print(json.dumps(result),flush=True)
        return

    optimizer,scheduler=make_optimizer_and_scheduler(cfg,policy)
    step=0
    if args.mode=='continue':
        assert args.checkpoint and json.loads((args.checkpoint/'reload-verification.json').read_text())['passed']
        step,optimizer,scheduler=load_training_state(args.checkpoint,optimizer,scheduler)
        assert step==1, 'Only continuation from verified smoke checkpoint supported'
    parameter_info={'trainable_parameters':sum(p.numel() for p in intended.values()),
                    'total_parameters':sum(p.numel() for p in policy.parameters()),
                    'trainable_tensors':{n:{'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in intended.items()},
                    'frozen_parameters':sum(p.numel() for p in policy.parameters() if not p.requires_grad)}
    dump(OUT/'parameters.json',parameter_info)
    manifest={'model':MODEL,'model_revision':MODEL_REV,'dataset':DATASET,'dataset_revision':DATA_REV,
              'device':torch.cuda.get_device_name(),'cuda_total_bytes':torch.cuda.get_device_properties(0).total_memory,
              'python_packages':{n:md.version(n) for n in ['torch','lerobot','accelerate','numpy','av','safetensors']},
              'script_sha256':filehash(Path(__file__)),'base_model_sha256':filehash(MODEL_ROOT/'model.safetensors'),
              'processor_sha256':{n:filehash(MODEL_ROOT/n) for n in PROCESSORS},
              'trainable_count':parameter_info['trainable_parameters'],'steps_budget':cfg.steps,
              'supported_freezing':True,'gradient_checkpointing':False,'peft':False,'precision':'native BF16/FP32, use_amp=false',
              'optimizer':cfg.optimizer.to_dict() if hasattr(cfg.optimizer,'to_dict') else vars(cfg.optimizer),
              'scheduler':vars(cfg.scheduler),'effective_warmup_steps':int(cfg.scheduler.num_warmup_steps*cfg.steps/cfg.scheduler.num_decay_steps)}
    manifest['instruction_intervention'] = 'CloseFridge' if args.task_id else 'original dataset task'
    dump(OUT/'manifest.json',manifest)
    if step==0:
        initial={'step':0,'validation':evaluate(val_raw),'fixed_training_probe':evaluate(train_probe)}
        dump(OUT/'initial-losses.json',initial)
        print(json.dumps({'initial':initial,'trainable':parameter_info['trainable_parameters']}),flush=True)

    latest_grad={}
    def before_step(opt,hook_args,hook_kwargs):
        grads={n:p.grad for n,p in intended.items() if p.grad is not None}
        assert grads and all(torch.isfinite(g).all() for g in grads.values()), 'Nonfinite/missing gradients'
        assert target_name in grads and 'model.state_proj.weight' in grads
        assert any('lm_expert' in n for n in grads)
        assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
        latest_grad.update(finite=True,tensors_with_grad=len(grads),
                           missing=[n for n in intended if n not in grads],
                           target_grad_norm=float(grads[target_name].float().norm()))
    handle=optimizer.register_step_pre_hook(before_step)
    meters={k:AverageMeter(k,':.6f') for k in ['loss','grad_norm','lr','update_s','gpu_mem_gb']}
    tracker=MetricsTracker(1,len(train),len(train.episodes),meters,initial_step=step,accelerator=accelerator)
    before=intended[target_name].detach().cpu().clone()
    frozen_name,frozen_p=next((n,p) for n,p in policy.named_parameters() if not p.requires_grad)
    frozen_hash=hashlib.sha256(frozen_p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    started=time.perf_counter();max_memory=0;max_reserved=0
    limit=1 if args.mode=='smoke' else cfg.steps
    def checkpoint():
        path=OUT/'checkpoints'/f'{step:06d}'
        assert not path.exists(), 'Checkpoint output already exists'
        validation=evaluate(val_raw)
        save_checkpoint(path,step,cfg,policy,optimizer,scheduler,pre,post,num_processes=1,batch_size=1)
        # Preserve original processor bytes as well as their semantics.
        for name in PROCESSORS:
            shutil.copyfile(MODEL_ROOT/name,path/'pretrained_model'/name)
        dump(path/'verification-target.json',{'step':step,'state_hashes':hashes(),'validation':validation})
        return str(path)
    try:
        with (OUT/'updates.jsonl').open('a',encoding='utf-8') as log:
            for index in range(step,limit):
                batch=prep(train[train_indices[index]])
                lr_used=optimizer.param_groups[0]['lr']
                torch.cuda.synchronize();start=time.perf_counter()
                tracker,_=update_policy(tracker,policy,batch,optimizer,cfg.optimizer.grad_clip_norm,accelerator,scheduler)
                torch.cuda.synchronize();step=index+1
                assert np.isfinite(tracker.loss.val) and np.isfinite(tracker.grad_norm.val)
                max_memory=max(max_memory,torch.cuda.max_memory_allocated())
                max_reserved=max(max_reserved,torch.cuda.max_memory_reserved())
                record={'step':step,'loss':tracker.loss.val,'grad_norm':tracker.grad_norm.val,'lr_used':lr_used,
                        'seconds':time.perf_counter()-start,'peak_allocated':torch.cuda.max_memory_allocated(),
                        'peak_reserved':torch.cuda.max_memory_reserved(), 'episode':int(batch['episode_index'].item()),
                        'sample_index':train_indices[index],**latest_grad}
                log.write(json.dumps(record)+'\n');log.flush()
                if step==1:
                    after=intended[target_name].detach().cpu()
                    difference=(after-before).abs()
                    assert difference.max()>0,'Optimizer did not change intended parameter'
                    assert hashlib.sha256(frozen_p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()==frozen_hash
                    dump(OUT/'smoke-update.json',{**record,'parameter':target_name,
                          'changed_elements':int(torch.count_nonzero(difference)), 'max_abs_change':float(difference.max()),
                          'trainable_parameters':parameter_info['trainable_parameters'],'frozen_probe_unchanged':frozen_name})
                if step%10==0 or step==1:print(json.dumps(record),flush=True)
                if step%50==0:
                    report={'step':step,'validation':evaluate(val_raw),'fixed_training_probe':evaluate(train_probe)}
                    dump(OUT/f'losses-{step:06d}.json',report);print(json.dumps(report),flush=True)
                if step in [1,150,300]:
                    saved=checkpoint();print(json.dumps({'checkpoint':saved,'step':step}),flush=True)
        dump(OUT/f'{args.mode}-summary.json',{'completed_optimizer_steps':step,'process_seconds':time.perf_counter()-started,
              'peak_allocated':max_memory,'peak_reserved':max_reserved,'last_checkpoint':saved,'exception':None})
    except Exception:
        dump(OUT/f'{args.mode}-failure.json',{'completed_optimizer_steps':step,'exception':traceback.format_exc(),
              'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        raise
    finally:
        handle.remove()


if __name__=='__main__':
    main()
