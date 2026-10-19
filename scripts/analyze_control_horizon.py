"""Summarize actual traces; historical missing contacts remain null."""
import json
from pathlib import Path
import numpy as np
from control_horizon_experiment import ROOT, OUT, read, reference, save
from control_diagnostics import action_change


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def stats(values):
    return {'mean': float(np.mean(values)), 'max': float(np.max(values)),
            'p95': float(np.percentile(values, 95))} if values else None


def analyze(folder, seed, horizon):
    rows = lines(folder/'steps.jsonl')
    reset, summary = read(folder/'reset.json'), read(folder/'summary.json')
    assert len(rows) == summary['simulator_steps']
    assert [r['step'] for r in rows if r['inference']] == list(range(1, len(rows)+1, horizon))
    assert all(r['finite'] for r in rows)
    result = {'seed': seed, 'horizon': horizon, 'artifacts': str(folder.relative_to(ROOT)),
              'success': summary['final_fridge']['success'], 'actions': len(rows),
              'stop': summary['termination_reason'], 'inferences': summary['inference_calls'],
              'inference_seconds_total': sum(summary['inference_seconds']),
              'inference_seconds': stats(summary['inference_seconds']),
              'warm_inference_seconds': stats(summary['inference_seconds'][1:]),
              'wall_seconds': summary['wall_clock_rollout_seconds'],
              'simulated_seconds': summary['simulated_seconds'], 'doors': {}, 'trajectories': {}}
    result['guard_failures'] = [{'step': r['step'], 'failures': r['guard_failures']} for r in rows if r['guard_failures']]
    for name, initial in reset['fridge']['qpos'].items():
        angles = np.rad2deg([initial]+[r['fridge']['qpos'][name] for r in rows])
        result['doors'][name] = {'initial_deg': angles[0], 'final_deg': angles[-1],
                               'min_deg': float(min(angles)), 'max_deg': float(max(angles)),
                               'net_closing_deg': angles[0]-angles[-1]}
    for key in ('robot0_eef_pos', 'robot0_base_pos', 'robot0_base_to_eef_pos'):
        points = np.array([reset['physical'][key]]+[r['physical'][key] for r in rows])
        result['trajectories'][key] = {'path_m': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                                       'net_m': float(np.linalg.norm(points[-1]-points[0])),
                                       'min_xyz_m': points.min(axis=0).tolist(), 'max_xyz_m': points.max(axis=0).tolist()}
    changes = [(r['inference'], action_change(p['action'], r['action'])) for p,r in zip(rows, rows[1:])]
    result['continuity'] = {label: stats([d['continuous_l2'] for boundary,d in changes if boundary == flag])
                            for label,flag in [('replan', True), ('within_chunk', False)]}
    result['continuity']['mode_switches'] = sum(d['mode_switch'] for _,d in changes)
    result['continuity']['gripper_switches'] = sum(d['gripper_switch'] for _,d in changes)
    result['contacts'] = None
    if (folder/'contacts.jsonl').exists():
        catalog = read(folder/'geometry-catalog.json')
        contacts = lines(folder/'contacts.jsonl')
        assert len(contacts) == len(rows) and all(len(c['samples']) == 25 for c in contacts)
        pairs, contact_steps = {}, set()
        for record in contacts:
            for sample in record['samples']:
                for c in sample['robot_contacts']:
                    names = [catalog[g]['name'] for g in c['geom_ids']]
                    # Fixture provenance remains inspectable in the full geometry catalog.
                    fridge = any('fridge' in ' '.join([catalog[g]['name'], *catalog[g]['body_ancestors']]).lower() for g in c['geom_ids'])
                    key = ' <-> '.join(sorted(names))
                    p = pairs.setdefault(key, {'fridge': fridge, 'samples': 0, 'loaded_samples': 0,
                                              'max_normal_N': 0., 'normal_impulse_Ns': 0.,
                                              'first_step': record['step'], 'last_step': record['step']})
                    normal = c['force_N_contact_frame'][0]
                    p['samples'] += 1
                    p['loaded_samples'] += int(normal > 1e-8 and c['constraint_address'] >= 0)
                    p['max_normal_N'] = max(p['max_normal_N'], normal)
                    p['normal_impulse_Ns'] += max(0., normal)*.002
                    p['last_step'] = record['step']
                    if fridge and normal > 1e-8 and c['constraint_address'] >= 0:
                        contact_steps.add(record['step'])
        result['contacts'] = {'sampling': 'every 0.002s after solver; robot-involving contacts only',
                              'robot_fridge_loaded_steps': sorted(contact_steps), 'pairs': pairs}
        result['simulation_step_seconds_including_diagnostics'] = sum(r['simulation_step_seconds_including_diagnostics'] for r in rows)
    return result


def main():
    results = [analyze(reference(seed) if h == 50 else OUT/f'h20-seed{seed}', seed, h)
               for seed in (42,43,44) for h in (50,20)]
    import torch
    comparisons = []
    for seed in (42,43,44):
        old, new = reference(seed), OUT/f'h20-seed{seed}'
        a = torch.load(old/'first-chunk.pt', map_location='cpu', weights_only=True)
        b = torch.load(new/'first-chunk.pt', map_location='cpu', weights_only=True)
        ar, br = lines(old/'steps.jsonl'), lines(new/'steps.jsonl')
        comparisons.append({'seed': seed, 'first_chunk_max_abs': {k: float((a[k]-b[k]).abs().max()) for k in a},
                            'first20_action_max_abs': float(np.max(np.abs(np.array([r['action'] for r in ar[:20]])-np.array([r['action'] for r in br[:20]])))),
                            'first20_qpos_max_abs': float(np.max(np.abs(np.array([r['physical']['qpos'] for r in ar[:20]])-np.array([r['physical']['qpos'] for r in br[:20]]))))})
    save('results.json', {'episodes': results, 'prefix_comparison': comparisons,
                         'limitations': ['H50 contact forces unavailable', 'Three known design seeds; no generalization claim',
                                         'Wall times include unequal instrumentation; inference is synchronous, simulator pauses while planning',
                                         'RNG seeded once per episode; after step20, replanning changes noise-to-simulation-time alignment',
                                         'Mixed-unit continuous command L2 is descriptive, not a physical jerk metric']})
    print(json.dumps({'episodes': [{k:r[k] for k in ('seed','horizon','success','actions','inferences','wall_seconds','doors')} for r in results], 'prefix': comparisons}, indent=2))


if __name__ == '__main__':
    main()
