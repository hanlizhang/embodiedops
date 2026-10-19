"""Build the Phase 1D report and local comparison page from measured artifacts."""
import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase1d'
TRAIN = OUT / 'training-v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def doors(row):
    angles = row['door_degrees']
    left = next((v for k, v in angles.items() if k.endswith('left_door_joint')), None)
    right = next((v for k, v in angles.items() if k.endswith('right_door_joint')), None)
    others = [v for k, v in angles.items() if not k.endswith(('left_door_joint', 'right_door_joint'))]
    return [f'{v:.3f}' if v is not None else 'N/A' for v in [left, right]] + [', '.join(f'{v:.3f}' for v in others) or 'N/A']


def main():
    updates = [json.loads(s) for s in (TRAIN / 'updates.jsonl').read_text().splitlines()]
    assert [u['step'] for u in updates] == list(range(1, 301))
    assert all(u['finite'] and not u['missing'] for u in updates)
    initial = read(TRAIN / 'initial-losses.json')
    losses = [initial] + [read(p) for p in sorted(TRAIN.glob('losses-*.json'))]
    smoke = read(TRAIN / 'smoke-update.json')
    summary = read(TRAIN / 'continue-summary.json')
    manifest = read(TRAIN / 'manifest.json')
    split = read(TRAIN / 'split.json')
    assert {u['episode'] for u in updates} <= set(split['train_episodes'])
    assert not {u['episode'] for u in updates} & set(split['validation_episodes'])
    reload = read(TRAIN / 'checkpoints/000300/reload-verification.json')
    weights = read(TRAIN / 'final-weight-audit.json')
    assert reload['passed']
    evaluation = read(OUT / 'evaluation/results.json')
    rows = evaluation['episodes']
    final_audit = read(OUT / 'final-audit.json')
    assert final_audit['passed'] and final_audit['episodes'] == 20
    assert hashlib.sha256((OUT / 'evaluation/results.json').read_bytes()).hexdigest() == final_audit['results_sha256']
    assert len(rows) == 20 and {(r['seed'], r['checkpoint']) for r in rows} == {
        (s, c) for s in range(42, 52) for c in ['pretrained', 'trained']}
    strata = {str(k): {c: {'successes': sum(r['success'] for r in rows if r['checkpoint'] == c and len(r['door_degrees']) == k),
                           'episodes': sum(r['checkpoint'] == c and len(r['door_degrees']) == k for r in rows)}
                      for c in ['pretrained', 'trained']} for k in [1, 2]}
    n = len(rows) // 2
    counts = {label: sum(r['success'] for r in rows if r['checkpoint'] == label) for label in ['pretrained', 'trained']}
    peak = max(u['peak_allocated'] for u in updates)
    reserved = max(u['peak_reserved'] for u in updates)
    elapsed = read(TRAIN / 'smoke-summary.json')['process_seconds'] + summary['process_seconds']
    timings = {}
    for mode, marker in [('smoke', 'smoke-started.json'), ('continue', 'continue-000001-started.json')]:
        start = read(TRAIN / marker)['started_unix']
        end = (TRAIN / f'{mode}-summary.json').stat().st_mtime
        timings[mode] = {'started_utc': datetime.fromtimestamp(start, timezone.utc).isoformat(),
                         'completion_artifact_written_utc': datetime.fromtimestamp(end, timezone.utc).isoformat(),
                         'start_to_completion_artifact_seconds': end-start,
                         'end_time_source': 'completion summary filesystem mtime; not process-exit time'}
    (OUT / 'training-timing.json').write_text(json.dumps(timings, indent=2), encoding='utf-8')
    loss_table = '\n'.join(f"| {x['step']} | {x['fixed_training_probe']['mean']:.6f} | {x['validation']['mean']:.6f} |" for x in losses)
    eval_table = '\n'.join(f"| {r['seed']} | {r['checkpoint']} | {r['success']} | {' | '.join(doors(r))} | {r['actions']} | {r['mean_chunk_inference_seconds']:.3f} | {r['wall_clock_rollout_seconds']:.2f} |" for r in rows)
    audit = {'completed_optimizer_steps': len(updates), 'all_gradient_checks_passed': True,
             'trainable_parameters': smoke['trainable_parameters'], 'peak_allocated': peak,
             'peak_reserved': reserved, 'training_loop_seconds_including_checkpoints_and_probes': elapsed,
             'mean_update_seconds': statistics.mean(u['seconds'] for u in updates),
             'first_50_update_loss_mean': statistics.mean(u['loss'] for u in updates[:50]),
             'last_50_update_loss_mean': statistics.mean(u['loss'] for u in updates[-50:]),
             'timing': timings,
             'fixed_losses': losses, 'success_counts': counts, 'seeds_per_checkpoint': n,
             'final_checkpoint_reload_passed': reload['passed'], 'door_strata': strata}
    (OUT / 'training-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    # Small, portable evidence summaries can accompany code; raw videos/weights stay ignored.
    portable_rows = [{**r, 'video': Path(r['video']).relative_to(ROOT).as_posix(),
                       'artifacts': Path(r['artifacts']).relative_to(ROOT).as_posix()} for r in rows]
    evidence = {'results_sha256': final_audit['results_sha256'], 'episodes': portable_rows,
                'success_counts': counts, 'door_strata': strata,
                'checkpoint_sha256': final_audit['checkpoint_sha256'],
                'checkpoint': 'outputs/phase1d/training-v2/checkpoints/000300/pretrained_model',
                'completed_optimizer_steps': 300, 'fixed_losses': losses,
                'peak_allocated_bytes': peak, 'peak_reserved_bytes': reserved,
                'training_loop_seconds': elapsed}
    (ROOT / 'environment/phase1d-results.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    portable_audit = {**final_audit, 'source_hashes': {
        Path(p).relative_to(ROOT).as_posix(): h for p, h in final_audit['source_hashes'].items()}}
    (ROOT / 'environment/phase1d-audit.json').write_text(json.dumps(portable_audit, indent=2), encoding='utf-8')
    shutil.copyfile(TRAIN / 'config/train_config.json', ROOT / 'environment/phase1d-training-config.json')
    report = f'''# Phase 1D — real SmolVLA training and paired CloseFridge evaluation

2026-10-19. **300 optimizer updates completed on the local RTX 3060 Laptop.** Paired task success:
pretrained **{counts['pretrained']}/{n}**, trained **{counts['trained']}/{n}** using the actual RoboCasa predicate.
These results distinguish successful training execution from successful robot control.
Evaluation status: **{n}/{len(evaluation['config']['seeds'])} planned seed pairs completed**.
One task-relevant fridge door: pretrained **{strata['1']['pretrained']['successes']}/7**, trained **{strata['1']['trained']['successes']}/7**.
Two task-relevant fridge doors: pretrained **{strata['2']['pretrained']['successes']}/3**, trained **{strata['2']['trained']['successes']}/3**.
Trained successes: seed 44 at 267 actions, seed 47 at 422, seed 49 at 413.
Seed 48 ended at {next(iter(next(r['door_degrees'] for r in rows if r['seed']==48 and r['checkpoint']=='trained').values())):.3f} degrees and **failed** the actual predicate.
The observed sample improvement is 30 percentage points. Ten paired seeds and one
training run do not establish statistically reliable generalization. The three
discordant pairs favor training; an exact two-sided paired sign test is p=0.25.
Overlap between the checkpoint's original pretraining data and our CloseFridge
training data or evaluation scenes is unknown. Episode-disjoint validation only
establishes separation within this local dataset split, not pretraining independence.
The fixed held-out loss worsened from 0.531797 to 0.594187; the fixed training
probe worsened from 0.301324 to 0.352550. This run does not demonstrate lower loss.
No dependencies changed, no original checkpoint or Phase 1A/1B evidence changed,
no RESUME_EVIDENCE update, and no commit.

## Language control (one additional seed-42 episode)

`CloseFridge` closed the right door, but not the left: final angles **87.867° / ~0°**;
actual task success remained false after 600 actions. The prior natural-language
run ended at **87.265° / 80.183°**. Both reset qpos/qvel hashes match exactly.
Base net displacement changed from 0.1405 m to 0.0040 m; EEF world path length
changed from 1.658 m to 3.158 m. The language control used open-gripper commands
and achieved-mode commands throughout (the original used 55 desired-mode commands).
This is one observed language sensitivity result, not a success-rate estimate.
Video: `outputs/phase1d/language-control-native/rollout.mp4`.

An earlier sandbox initialization was interrupted before reset or any action;
its artifacts remain in `outputs/phase1d/language-control`. Native execution
completed normally. No additional prompt sweep was performed.

## Exact training contract

- LeRobot 0.6.1; Torch {manifest['python_packages']['torch']}; existing `embodiedops-robocasa` environment.
- Policy: `{manifest['model']}` at `{manifest['model_revision']}`.
- Dataset: `{manifest['dataset']}` at `{manifest['dataset_revision']}`.
- RoboCasa 1.0.0 source: `56e355ccc64389dfc1b8a61a33b9127b975ba681`.
- robosuite 1.5.2 source: `aaa8b9b214ce8e77e82926d677b4d61d55e577ab`; MuJoCo 3.3.1.
- Backbone: `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.
- Native supported switches: `freeze_vision_encoder=true`, `train_expert_only=true`,
  `train_state_proj=true`. **{smoke['trainable_parameters']:,} / 450,046,176** parameters trainable,
  in 155 tensors. No custom architecture, PEFT installation, or gradient checkpointing.
  The entire VLM is frozen; the action expert, state projection and action/time
  projections are trainable.
  SmolVLA's installed configuration exposes no gradient-checkpointing switch;
  PEFT is optional and absent. Freezing fit comfortably without either.
- Batch 1, workers 0, seed 42, native BF16/FP32, `use_amp=false`, no augmentation.
- Checkpoint's AdamW preset: peak LR 1e-4, betas (0.9, 0.95), eps 1e-8,
  weight decay 1e-10, gradient clipping 10. The native scheduler scales its
  1000/20000 warmup/decay prescription to 15/300 steps; initial LR 6.25e-6,
  final decay target 2.5e-6. No loss-driven hyperparameter changes.
- Episode split: {len(split['train_episodes'])} training episodes ({split['train_frames']} frames),
  {len(split['validation_episodes'])} validation episodes ({split['validation_frames']} frames),
  disjoint. Native split takes the last ceil(10%) episodes per instruction.
  Exact episode lists and 300 deterministic shuffled training sample indices are saved.
  The updates touched {len({u['episode'] for u in updates})} training episodes and no validation episodes.
  This is only 300 sampled transitions, not a full epoch. Pretraining overlap is unknown.
- Left camera → camera1; right → camera2; wrist → camera3. Original processor JSONs
  and state/action normalization tensors are byte-identical in saved checkpoints.
  The stock CLI replaces normalizer statistics from the selected dataset;
  this runner instead reuses the saved processors while calling LeRobot's actual
  dataset, policy, optimizer, update and checkpoint functions.

Each training sample supplies three RGB images: collated `[1,3,256,256]`, uint8
converted to float [0,1]; SmolVLA internally resizes to `[1,3,512,512]` and rescales
to [-1,1]. State is `[1,1,16]`, internally last-time-selected and padded to 32.
Language tokens/mask are length 48. Actions are `[1,50,12]` with `[1,50]` padding
mask; the model pads actions to 32. Native flow matching samples noise and time,
forms `x_t = t*noise + (1-t)*actions`, predicts velocity, and minimizes squared
error against `noise-actions`, over the 12 real action dimensions and valid
episode positions. `update_policy` runs forward, backward, gradient clipping,
AdamW, zero_grad and scheduler.step. No forward-only substitute was used.

The initial smoke attempt stopped before forward because this runner incorrectly
expected a camera time axis. Actual decoded BCHW images match the installed
`prepare_images` contract; only that assertion was corrected. The zero-update
attempt remains in `outputs/phase1d/training`; the successful run is `training-v2`.

## Measured training and checkpoint evidence

Smoke loss **{smoke['loss']:.6f}**, gradient norm **{smoke['grad_norm']:.6f}**;
all 155 intended tensors had finite gradients, with no missing gradients.
The action output projection changed in **{smoke['changed_elements']:,}** elements,
max absolute change {smoke['max_abs_change']:.9g}; the sampled frozen parameter stayed equal.
All 300 updates passed gradient checks. Peak allocated/reserved CUDA memory:
**{peak / 2**30:.3f} / {reserved / 2**30:.3f} GiB** (Torch allocator, not whole-device use).
Training loops took **{elapsed:.2f} s**, including probes/checkpoint writes but excluding
imports, dataset setup, initial loss probes and separate reload processes.
`outputs/phase1d/training-timing.json` records UTC starts from the original run markers and
completion-summary write timestamps from filesystem metadata, including setup time.
Mean synchronized optimizer update: **{audit['mean_update_seconds']:.3f} s**.
First/last 50 stochastic update loss means: **{audit['first_50_update_loss_mean']:.6f} / {audit['last_50_update_loss_mean']:.6f}**;
these differ in samples/noise and are not a paired loss comparison.

An exact final-vs-pretrained tensor audit found {weights['changed_tensor_count']} changed tensors,
all in the intended trainable set; every frozen tensor remained equal.
The 33 trainable expert normalization-weight tensors remained byte-identical
despite receiving gradients under the retained native precision. No precision
or architecture change was made to force additional updates.

The following probes use eight fixed training frames and eight fixed held-out
frames, with fixed flow-noise seeds. They are small diagnostic probes, not a
full validation-set score or evidence of task success.

| Step | Fixed training probe loss | Fixed validation probe loss |
|---:|---:|---:|
{loss_table}

Full checkpoints: `outputs/phase1d/training-v2/checkpoints/000001`, `000150`, `000300`.
Steps 1 and 300 were reloaded in separate processes: all **500 state tensors**
matched SHA256 hashes exactly, processors matched byte-for-byte, and fixed
validation losses matched within rtol 1e-5 / atol 1e-6. Optimizer, scheduler and RNG
state are saved; continuation loaded the verified step-1 training state.
Final model SHA256: `{evaluation['config']['model_sha256']}`.
Local sustained training succeeded, so a Nebius fallback was not required.

## Paired robot evaluation

Both checkpoints use each scene's native instruction (singular `Close the fridge door.`
or plural `Close the fridge doors.`); only weights differ within each pair.
The separate `CloseFridge` language control is excluded from this comparison.
The original seed-42 natural-language baseline was reused. Other episodes use
the same pinned scene generator, 600-action horizon, 20 Hz control, 25 physics
substeps/action, 50-action queue and 10 flow steps. Before each trained run,
reset qpos/qvel and state must match its paired baseline exactly. All three camera
views must match within Phase 1C's existing sparse one-level rasterization tolerance
(maximum 1/255, mean at most 0.001/255). New scene metadata also matches when present;
seed 42 additionally matches the original Phase 1B scene metadata. An initial
zero-action evaluation attempt rejected four one-level RGB differences in the
right view under an overly strict exact-pixel check; it is preserved in
`evaluation/trained-seed42-reset-rounding`. The first learned action
and all subsequent actions retain the Phase 1C guards. Success is RoboCasa's
`_check_success()`, requiring all relevant fridge doors closed. Some seeded scenes
have one task-relevant fridge joint; its angle is in the Other/single column,
with left/right marked N/A (a separate freezer door is not relabeled as a fridge door).
No success surrogate from loss.

Another zero-action attempt, `evaluation/trained-seed45-metadata-order`, differed
only in the order of `receptacle` / `stackable` category labels. RoboCasa constructs
`groups_containing_sampled_obj` by iterating a group dictionary populated from a
Python set (`kitchen_objects.py:2962`, `kitchen_object_utils.py:488`). The comparator
now sorts only this membership-label list; it preserves all poses, asset paths,
controller values and other list ordering. Raw scene metadata remains recorded.
The evaluation aggregator also accepts each fixture's actual singular/plural
instruction instead of assuming every fixture has two fridge doors; completed
episodes were reused after this reporting assertion was corrected.

| Seed | Checkpoint | Success | Left ° | Right ° | Other/single ° | Actions | Chunk inference s | Episode wall s |
|---:|---|---|---:|---:|---:|---:|---:|---:|
{eval_table}

Episode wall times exclude initialization; chunk latency includes instrumentation
and CPU copies, so these are diagnostic timings, not throughput benchmarks.
Videos include reset plus every action endpoint. Exact paths, inference calls,
physics steps, simulated duration and exceptions are in `evaluation/results.json`.
See `outputs/phase1d/comparison.html` for paired before/after videos.
This small seeded evaluation does not establish generalization or causality.

## Final offline audit

`outputs/phase1d/final-audit.json` verifies exactly 20 unique complete episodes,
with portable summaries in `environment/phase1d-audit.json` and `phase1d-results.json`.
seeds 42–51 paired by checkpoint. Summaries, sequential action logs, stop reasons,
model identities, native language, reset guards and final door values agree.
The saved actual `_check_success()` flags also agree with the pinned source:
`ManipulateDoor._check_success` calls `Fridge.is_closed`, which requires every
task-relevant normalized fridge joint to be at most 0.005. Visual near-closure
is not success. No policy or simulator was executed during this review.

All 20 MP4s decoded completely with strictly increasing timestamps: reset plus
every executed action, 768x288 three-camera frames. All 11 previously completed
episode directories (487 files) match pre-resume inventories including file bytes,
sizes and modification times. The interrupted seed-47 directory/log and prior
aggregate results/config remain preserved with matching inventories:
`trained-seed47-interrupted-before-reset`, its `.log`,
`results-before-seed47-resume.json`, `evaluation-config-before-seed47-resume.json`.
The interruption occurred before reset/inference and is excluded from success rates.
Step-300 weight SHA256 and all 500 tensor hashes were rechecked offline against
the saved reload target; 122 intended tensors changed, all frozen tensors agree
with the original. The 461/52 episode split is disjoint and covers all 513 episodes;
all 300 logged updates use training episodes and have finite loss/gradients.

## Reproduction and review

Exact code: `scripts/phase1d_train.py`, `scripts/phase1d_rollout.py`,
`scripts/phase1d_evaluate.py`, `scripts/phase1d_report.py`.
Exact resolved training config: `environment/phase1d-training-config.json`.
This provenance copy retains the original machine's absolute paths; it is not
a portable CLI configuration. Scripts resolve caches relative to the checkout.
Raw manifests, step logs, split, losses, gradient/weight checks and reload evidence
are under `outputs/phase1d/training-v2`.

```powershell
conda activate embodiedops-robocasa
# Run from the repository root. Historical commands below are not executed by reporting.
python scripts/phase1d_train.py --mode smoke --output outputs/phase1d/training-v2
python scripts/phase1d_train.py --mode verify --output outputs/phase1d/training-v2 --checkpoint outputs/phase1d/training-v2/checkpoints/000001
python scripts/phase1d_train.py --mode continue --output outputs/phase1d/training-v2 --checkpoint outputs/phase1d/training-v2/checkpoints/000001
python scripts/phase1d_train.py --mode verify --output outputs/phase1d/training-v2 --checkpoint outputs/phase1d/training-v2/checkpoints/000300
python scripts/phase1d_evaluate.py --checkpoint outputs/phase1d/training-v2/checkpoints/000300 --seeds {' '.join(map(str,evaluation['config']['seeds']))}
python scripts/phase1d_report.py
```

These record the executed workflow. Exclusive run markers prevent overwriting
existing experiments; choose a new output directory for any separately authorized
repeat. Native simulator execution was needed because sandbox initialization stalled.
'''
    (ROOT / 'environment/phase1d.md').write_text(report, encoding='utf-8')
    chart = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 260" role="img" aria-label="Fixed training and validation losses">',
             '<rect width="800" height="260" fill="#101b2e"/>']
    for y in [0, .2, .4, .6, .8]:
        py = 215-y*210
        chart += [f'<path d="M50 {py} H760" stroke="#334155"/><text x="10" y="{py+4}" fill="#cbd5e1" font-size="12">{y:.1f}</text>']
    for key,color,label,x in [('fixed_training_probe','#38bdf8','Training probe',60),('validation','#fbbf24','Held-out probe',260)]:
        points = ' '.join(f"{50+r['step']/300*710},{215-r[key]['mean']*210}" for r in losses)
        chart += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{x}" y="20" fill="{color}" font-size="14">{label}</text>']
    for step in [0,50,100,150,200,250,300]:
        chart.append(f'<text x="{45+step/300*710}" y="240" fill="#cbd5e1" font-size="12">{step}</text>')
    chart.append('</svg>')
    (OUT / 'losses.svg').write_text(''.join(chart), encoding='utf-8')
    pairs = []
    for seed in sorted({r['seed'] for r in rows}):
        pair = {r['checkpoint']: r for r in rows if r['seed'] == seed}
        for r in pair.values():
            r['video_relative'] = os.path.relpath(r['video'], OUT).replace('\\','/')
            r['angles'] = doors(r)
            assert (OUT / r['video_relative']).is_file()
            r.pop('video', None)
            r.pop('artifacts', None)
        pairs.append({'seed': seed, **pair})
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>EmbodiedOps · CloseFridge training evidence</title>
<style>body{background:#0b1220;color:#e2e8f0;font:16px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 24px}h1{font-size:38px;line-height:1.15}h2{font-size:22px}.muted{color:#94a3b8}.cards,.videos{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:#152238;border:1px solid #334155;border-radius:12px;padding:20px}.metric{font-size:30px;color:#7dd3fc}video{width:100%;background:#000}button,select{padding:10px 16px;margin:8px;background:#213650;color:white;border:1px solid #64748b;border-radius:6px}img{width:100%;max-width:800px}a{color:#7dd3fc}code{overflow-wrap:anywhere}@media(max-width:800px){.cards,.videos{grid-template-columns:1fr}}</style>
<p class="muted">EMBODIEDOPS / MEASURED EXPERIMENT</p><h1>CloseFridge: 0/10 to 3/10<br>after 300 local updates.</h1>
<p>SmolVLA · RoboCasa · CloseFridge · RTX 3060 Laptop 6 GB</p>
<div class="cards"><div class="card"><div class="metric">300 optimizer updates</div>Finite gradients, changed weights, saved and reloaded checkpoints.<br>Peak allocated CUDA: PEAK GiB.</div><div class="card"><div class="metric">BEFORE → AFTER task successes</div>Identical seeded environments and language; actual RoboCasa success predicate.<br>Observed sample improvement; robust generalization is not established.</div></div>
<h2>Paired continuous rollouts</h2><p class="muted">Each pane shows left, right and wrist cameras. Videos run at simulation speed. Start both videos together to compare physical behavior; browser playback may drift. Success is the recorded RoboCasa predicate, not visual appearance. Seed 48 looks nearly closed but officially failed.</p>
<label>Scene seed <select id="seed"></select></label><button id="play">Replay both from reset</button><button id="pause">Pause both</button>
<div class="videos"><div><h3>Pretrained</h3><video id="before" controls preload="metadata"></video><p id="bm"></p></div><div><h3>After 300 updates</h3><video id="after" controls preload="metadata"></video><p id="am"></p></div></div>
<h2>Fixed loss probes</h2><img src="losses.svg" alt="Training and held-out loss by optimizer step"><p class="muted">Eight fixed frames per split, fixed noise draws. Validation probe worsened: 0.5318 to 0.5942. Episode-disjoint validation; pretraining overlap unknown. These are diagnostic losses, not robot success measurements.</p>
<h2>Separate language control</h2><p>Changing only the instruction to <code>CloseFridge</code> closed the right door on seed 42. The left stayed open; the task still failed. This run is excluded from the paired checkpoint comparison.</p>
<p><a href="evaluation/results.json">Raw evaluation results</a> · <a href="training-audit.json">Training audit</a> · <a href="../../environment/phase1d.md">Technical report</a></p>
<script>const pairs=PAIRS;const sel=document.querySelector('#seed'),b=document.querySelector('#before'),a=document.querySelector('#after');for(const p of pairs){const o=document.createElement('option');o.value=p.seed;o.textContent=p.seed+({44:' - featured success',47:' - success',49:' - success',48:' - near closure, official failure'}[p.seed]||' - failed');sel.append(o)}function load(){const p=pairs.find(p=>p.seed==sel.value);a.pause();b.pause();for(const [v,r,id] of [[b,p.pretrained,'bm'],[a,p.trained,'am']]){v.src=r.video_relative;document.getElementById(id).textContent=`Official simulated task success: ${r.success} · Door angles: ${r.angles.map((v,i)=>v==='N/A'?'':['Left','Right','Other/single'][i]+': '+v+'°').filter(Boolean).join(' / ')} · ${r.actions} actions · Chunk inference: ${r.mean_chunk_inference_seconds.toFixed(3)} s`}}sel.value='44';sel.onchange=load;document.querySelector('#play').onclick=()=>{for(const v of [a,b]){v.currentTime=0;v.play()}};document.querySelector('#pause').onclick=()=>{a.pause();b.pause()};load();</script></html>'''
    page = page.replace('PEAK', f'{peak/2**30:.3f}').replace('BEFORE', f"{counts['pretrained']}/{n}").replace('AFTER', f"{counts['trained']}/{n}").replace('PAIRS', json.dumps(pairs))
    (OUT / 'comparison.html').write_text(page, encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='fixed_losses'}, indent=2))


if __name__ == '__main__':
    main()
