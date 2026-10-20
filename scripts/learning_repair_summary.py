"""Audit completed sustained-learning artifacts and write compact result tables.

Read-only with respect to checkpoints and rollout evidence; no model or simulator.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from statistics import mean, median

from agent_workflow import ROOT, read, rows, dump, sha, acceptance


def loss_chart(out, training):
    """Standalone scientific plot from fixed probes, using installed Pillow."""
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 20)
    title_font = ImageFont.truetype(str(font_path), 25)
    im = Image.new('RGB', (1400, 620), 'white'); draw = ImageDraw.Draw(im)
    draw.text((40,20), 'Measured task-ID flow-matching loss — fixed samples and noise', fill='#111827', font=title_font)
    colors = {'episode_balanced':'#2563eb', 'motion_mixture':'#c2410c'}
    for panel, (key, title) in enumerate([('fixed_train_ID','Independent training probe (32 anchors)'),
                                        ('validation_ID','Held-out episodes (104 anchors)')]):
        left, top, width, height = 90+panel*680, 115, 550, 370
        values = [v for a in training.values() for v in a[key].values()]
        lo, hi = min(values)-.015, max(values)+.015
        xy = lambda step, value: (left+int(step)/5000*width, top+height-(value-lo)/(hi-lo)*height)
        draw.text((left,70), title, fill='#111827', font=font)
        for i in range(6):
            v = lo+(hi-lo)*i/5; y=xy(0,v)[1]
            draw.line([(left,y),(left+width,y)], fill='#e5e7eb', width=1)
            draw.text((left-68,y-10), f'{v:.3f}', fill='#374151', font=font)
            x=left+i*width/5
            draw.text((x-20,top+height+12), str(i*1000), fill='#374151', font=font)
        base = next(iter(training.values()))[key]['0']; y=xy(0,base)[1]
        for x in range(left,left+width,16):draw.line([(x,y),(min(x+8,left+width),y)],fill='#6b7280',width=2)
        for arm, data in training.items():
            points = [xy(s,v) for s,v in data[key].items()]
            draw.line(points, fill=colors[arm], width=3)
            for x,y in points:draw.ellipse((x-4,y-4,x+4,y+4),fill=colors[arm])
            selected = data['selected_step']; x,y=xy(selected,data[key][str(selected)])
            draw.rectangle((x-7,y-7,x+7,y+7),outline=colors[arm],width=2)
        draw.text((left+180,top+height+45),'Optimizer updates',fill='#374151',font=font)
    draw.text((90,565),'Blue: episode-balanced    Orange: motion mixture    Dashed: pretrained    Square: selected checkpoint',fill='#374151',font=font)
    im.save(out/'loss-comparison.png')


def summarize(out, allow_incomplete=False):
    plan = read(out/'registered-plan.json')
    trace = rows(out/'trace.jsonl')
    complete = (out/'result.json').exists()
    if complete:
        result = read(out/'result.json')
    else:
        assert allow_incomplete and (out/'blocked.json').exists(), 'Use --allow-incomplete to explicitly audit a blocked run'
        blocked = read(out/'blocked.json')
        result = dict(selected=read(out/'selected-checkpoints.json'), decisions={},
                      episodes=blocked['episodes'], gpu_worker_process_hours=blocked['gpu_worker_process_hours'],
                      wall_seconds=next(e['elapsed_seconds'] for e in trace if e['event']=='blocked'))
    training = {}
    for arm in plan['arms']:
        folder = out/arm
        updates, probes = rows(folder/'updates.jsonl'), rows(folder/'probes.jsonl')
        split, manifest = read(folder/'split.json'), read(folder/'manifest.json')
        selected = result['selected'][arm]
        assert len(updates) == plan['training']['steps']
        assert [r['step'] for r in updates] == list(range(1, len(updates)+1))
        assert all(all(r[k] == p[k] for k in p) for r, p in zip(updates, split['training_order']))
        assert set(r['episode'] for r in updates) == set(split['train_episodes'])
        assert not set(split['train_episodes']) & set(split['validation_episodes'])
        assert all(r['all_trainable_gradients_finite'] and r['tensors_with_grad'] == 155 for r in updates)
        assert manifest['script_sha256'] == sha(out/'phase2_train_v1.py.txt')
        assert read(folder/'sampling-audit.json')['sampler_sha256'] == sha(out/'learning_sampling.py.txt')
        checkpoint_probes = [p for p in probes if p['step'] in plan['checkpoint_steps']]
        best = min(checkpoint_probes, key=lambda p: (p['validation']['task_id']['mean'], p['step']))
        assert best['step'] == selected['selection']['step']
        hashes = {}
        for step in plan['checkpoint_steps']:
            ckpt = folder/'checkpoints'/f'{step:06d}'
            reload = read(folder/f'reload-{step:06d}.json')
            assert reload['passed'] and reload['model_tensors_equal'] == 500
            digest = sha(ckpt/'pretrained_model/model.safetensors')
            assert digest == reload['checkpoint_sha256'] != manifest['base_sha256']
            hashes[str(step)] = digest
        audit = read(folder/'parameter-audit.json')
        assert audit['all_frozen_unchanged']
        visits = Counter(r['episode'] for r in updates)
        worker = next(e for e in trace if e['event']=='worker_end' and e['name']==arm+'-train')
        summary = read(folder/'summary.json')
        training[arm] = dict(
            updates=len(updates), episodes=len(visits), distinct_anchors=len({r['index'] for r in updates}),
            visits_per_episode=[min(visits.values()), max(visits.values())],
            validation_episodes=len(split['validation_episodes']), validation_anchors=len(split['validation_samples']),
            sampling=read(folder/'sampling-audit.json'), selected_step=best['step'],
            fixed_train_ID={str(p['step']): p['train']['task_id']['mean'] for p in probes},
            validation_ID={str(p['step']): p['validation']['task_id']['mean'] for p in probes},
            validation_NL={str(p['step']): p['validation']['natural']['mean'] for p in probes},
            grad_norm_median=median(r['grad_norm'] for r in updates), grad_norm_max=max(r['grad_norm'] for r in updates),
            updates_above_clip=sum(r['grad_norm']>10 for r in updates),
            lr_min=min(r['lr_used'] for r in updates), lr_max=max(r['lr_used'] for r in updates),
            changed_tensors_final=len(audit['changed']), unchanged_trainable_final=len(audit['unchanged_trainable']),
            peak_allocated_GiB=summary['peak_allocated']/2**30, peak_reserved_GiB=summary['peak_reserved']/2**30,
            training_seconds=summary['seconds_including_probes_and_checkpoints'], worker_seconds=worker['seconds'],
            synchronized_update_seconds=sum(r['seconds'] for r in updates), checkpoint_sha256=hashes)
    a, b = [read(out/arm/'split.json') for arm in plan['arms']]
    assert [r['episode'] for r in a['training_order']] == [r['episode'] for r in b['training_order']]
    assert a['training_probe'] == b['training_probe'] and a['validation_samples'] == b['validation_samples']
    episodes = []
    for event in trace:
        if event['event'] != 'robot_result':
            continue
        m = dict(event['metrics'])
        folder = ROOT/m['path']
        # Actual rollout source artifacts remain linked, not replaced by table values.
        assert read(folder/'summary.json')['final_fridge']['success'] == m['success']
        manifest = read(folder/'manifest.json')
        expected = (read(out/plan['arms'][0]/'manifest.json')['base_sha256'] if event['arm']=='pretrained'
                    else result['selected'][event['arm']]['sha256'])
        assert manifest['model_sha256'] == expected
        assert manifest['seed'] == event['seed'] and manifest['execution_horizon'] == 50
        assert manifest['resolved_instruction'] == plan['instruction']
        assert manifest['script_sha256'] == sha(out/'phase1d_rollout.py.txt')
        catalog = {g['id']: g for g in read(folder/'geometry-catalog.json')}
        peak = None
        with (folder/'contacts.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                for substep, sample in enumerate(row['samples']):
                    for contact in sample['robot_contacts']:
                        force = contact['force_N_contact_frame'][0]
                        if contact['constraint_address'] >= 0 and force > 1e-8 and (peak is None or force > peak['normal_N']):
                            geoms = [catalog[i] for i in contact['geom_ids']]
                            peak = dict(normal_N=force, step=row['step'], substep_zero_based=substep,
                                        geoms=[g['name'] for g in geoms], robot_self_contact=all(g['robot'] for g in geoms))
        assert (peak['normal_N'] if peak else 0.) == m['peak_contact_normal_N']
        m['peak_contact_detail'] = peak
        m.update(stage=event['stage'], seed=event['seed'], arm=event['arm'])
        episodes.append(m)
    assert len(episodes) == result['episodes']
    if complete: assert len(episodes) == plan['max_simulator_episodes']
    # Compare only the intersection of valid scenes across all three policies.
    common = {}
    for stage in ('diagnostic', 'final'):
        seed_sets = [{r['seed'] for r in episodes if r['stage']==stage and r['arm']==arm and r['valid']}
                     for arm in ['pretrained']+plan['arms']]
        common[stage] = sorted(set.intersection(*seed_sets))
    if not complete:
        policy = read(ROOT/plan['acceptance_policy_source'])['acceptance']
        partial = read(out/'incomplete-evaluation.json')
        gates = {}
        for arm in plan['arms']:
            observed_pairs = [p for p in partial['pairs'][arm] if p['seed'] in common['diagnostic']]
            assert observed_pairs, 'No complete matched pairs to assess'
            gates[arm] = acceptance(observed_pairs, policy)
            reasons = ['incomplete_registered_matrix', 'invalid_observation_black_camera']+gates[arm]['reasons']
            if not result['selected'][arm]['validation_improved']:
                reasons.append('validation_regression_blocks_promotion')
            result['decisions'][arm] = dict(accepted=False, reasons=sorted(set(reasons)),
                                            technically_completed=False, final_evaluation='not_run')
        dump(out/'partial-acceptance.json', dict(complete=False, scope='valid common diagnostic seeds only',
             common_seeds=common, existing_acceptance_gate=gates, decisions=result['decisions']))
        dump(out/'review-trace.json', dict(event='offline_partial_acceptance_review',
             source_trace_sha256=sha(out/'trace.jsonl'), new_training_updates=0, new_simulator_episodes=0,
             tool='agent_workflow.acceptance', decisions=result['decisions'], complete=False))
    groups = {}
    for stage in ('diagnostic', 'final'):
        for arm in ['pretrained']+plan['arms']:
            rs = [r for r in episodes if r['stage']==stage and r['arm']==arm and r['seed'] in common[stage]]
            if complete: assert sorted(r['seed'] for r in rs) == sorted(plan[stage+'_seeds'])
            if not rs: continue
            groups[stage+'-'+arm] = dict(successes=sum(r['success'] for r in rs), episodes=len(rs),
                mean_closing_deg=mean(r['closing_deg'] for r in rs),
                max_contact_normal_N=max(r['peak_contact_normal_N'] for r in rs),
                mean_inference_seconds=mean(r['mean_inference_seconds'] for r in rs),
                guard_failures=sum(r['guard_failure'] for r in rs), rollout_seconds=sum(r['rollout_seconds'] for r in rs))
    before = read(out/'preservation-before.json')
    assert read(out/'preservation-after.json') == dict(passed=True, weights=before)
    assert all(sha(ROOT/p) == digest for p, digest in before.items())
    dump(out/'audited-summary.json', dict(passed=True, experiment_complete=complete, common_valid_seeds=common,
         attempted_episodes=len(episodes), valid_episodes=sum(r['valid'] for r in episodes), training=training, robot_groups=groups,
         decisions=result['decisions'], wall_seconds=result['wall_seconds'],
         gpu_worker_process_hours=result['gpu_worker_process_hours'], active_gpu_hours=None, energy_kWh=None))
    loss_chart(out, training)
    dump(out/'audited-episodes.json', episodes)
    keys = ['stage','seed','arm','scene_type','valid','success','actions','termination','closing_deg','best_closing_deg',
            'initial_angles_deg','final_angles_deg','loaded_actions','peak_contact_normal_N','eef_path_m',
            'mean_inference_seconds','rollout_seconds','peak_cuda_allocated','peak_contact_detail','path']
    with (out/'rollout-table.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(episodes)
    print(json.dumps(dict(passed=True, experiment_complete=complete, training={a: dict(selected_step=v['selected_step'],
          validation=v['validation_ID'], distinct_anchors=v['distinct_anchors']) for a,v in training.items()},
          robot_groups=groups, decisions=result['decisions']), indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true', help='Report a stopped run explicitly as incomplete')
    args=parser.parse_args()
    summarize(args.output.resolve(), args.allow_incomplete)
