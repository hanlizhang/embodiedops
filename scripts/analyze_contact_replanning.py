"""Read-only paired analysis; door quantities are evaluation only."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from agent_workflow import ROOT, read, rows, dump, episode_metrics, acceptance
from contact_replanning import Supervisor, Rule


def window(records, angles, start, end):
    part = records[start:end]
    if not part:
        return None
    per_door = np.diff(angles[start:end+1], axis=0)
    movements = per_door.sum(axis=1)
    return dict(actions=len(part), closing_deg=float(-movements.sum()),
                per_door_closing_deg=(-per_door.sum(axis=0)).tolist(),
                per_door_opening_steps_over_0_1deg=(per_door > .1).sum(axis=0).tolist(),
                opening_steps_over_0_1deg=int((movements > .1).sum()),
                door_stall_steps_under_0_05deg=int((np.abs(movements) < .05).sum()),
                mean_eef_motion_m=float(np.mean([r['feedback']['tracking']['achieved_motion_m'] for r in part])),
                mean_target_error_m=float(np.mean([r['feedback']['tracking']['target_error_m'] for r in part])),
                blocked_steps=sum(r['feedback']['decision']['blocked'] for r in part))


def analyze(folder):
    from PIL import Image
    rs, summary, reset = rows(folder/'steps.jsonl'), read(folder/'summary.json'), read(folder/'reset.json')
    final_images = list(folder.glob('final-robot*.png'))
    assert len(final_images) == 3 and all(np.asarray(Image.open(p)).max() > 0 for p in final_images)
    keys = list(reset['fridge']['qpos'])
    angles = np.rad2deg([[r['fridge']['qpos'][k] for k in keys] for r in [reset]+rs])
    m = episode_metrics(folder)
    m['worker_seconds'] = read(folder.parent/f'metrics-{folder.name}.json')['worker_seconds']
    events = [r for r in rs if r['feedback']['decision']['trigger'] and r['feedback']['decision']['enabled']]
    assert summary['inference_calls']*50 == (len(rs) + rs[-1]['queue_after_feedback'] +
           sum(r['feedback']['decision']['actions_discarded'] for r in events)), 'Action queue accounting mismatch'
    m.update(inference_calls=summary['inference_calls'], replan_events=len(events),
             discarded_actions=sum(r['feedback']['decision']['actions_discarded'] for r in events),
             inference_total_seconds=sum(summary['inference_seconds']), simulated_seconds=summary['simulated_seconds'],
             guard_activations=[dict(step=r['step'], failures=r['guard_failures']) for r in rs if r['guard_failures']],
             doors={k: dict(initial_deg=float(angles[0,i]), final_deg=float(angles[-1,i]),
                            best_closing_deg=float((angles[0,i]-angles[:,i]).max())) for i,k in enumerate(keys)},
             mean_target_error_m=float(np.mean([r['feedback']['tracking']['target_error_m'] for r in rs])),
             sum_target_delta_m=float(sum(np.linalg.norm(r['feedback']['tracking']['command_target_delta_m']) for r in rs)),
             sum_achieved_delta_m=float(sum(r['feedback']['tracking']['achieved_motion_m'] for r in rs)),
             contact_peaks_N={k: max(r['feedback']['contacts']['peak_normal_N'][k] for r in rs)
                              for k in ('robot_fridge', 'robot_self', 'robot_environment')},
             fridge_loaded_actions=sum(r['feedback']['contacts']['peak_normal_N']['robot_fridge'] >= 1 for r in rs),
             blocked_actions=sum(r['feedback']['decision']['blocked'] for r in rs))
    catalog = {g['id']: g for g in read(folder/'geometry-catalog.json')}
    m['peak_contact_details'] = {}
    for kind in ('robot_fridge', 'robot_self', 'robot_environment'):
        candidates = [(p['peak_normal_N'], r['step'], p['geom_ids']) for r in rs
                      for p in r['feedback']['contacts']['pairs'] if p['kind'] == kind]
        if candidates:
            force, step, ids = max(candidates)
            m['peak_contact_details'][kind] = dict(step=step, normal_N=force, geom_ids=ids,
                                                  geoms=[catalog[g]['name'] for g in ids])
        else:
            m['peak_contact_details'][kind] = None
    rule = Rule(**read(folder/'feedback-contract.json')['rule'])
    replay = Supervisor(rule)
    m['criteria_counts'] = {}
    for label, part in [('whole_episode', rs), ('last_100_actions', rs[-100:])]:
        m['criteria_counts'][label] = dict(actions=len(part),
            loaded_fridge=sum(r['feedback']['contacts']['peak_normal_N']['robot_fridge'] >= rule.loaded_normal_N for r in part),
            large_target_residual=sum(r['feedback']['tracking']['target_error_m'] >= rule.target_error_m for r in part),
            low_eef_motion=sum(r['feedback']['tracking']['achieved_motion_m'] <= rule.max_motion_m for r in part),
            all_three=sum(r['feedback']['decision']['blocked'] for r in part),
            longest_all_three_run=max((sum(1 for _ in group) if blocked else 0 for blocked,group in
                itertools.groupby(r['feedback']['decision']['blocked'] for r in part)), default=0))
    for r in rs:
        d = r['feedback']['decision']
        # Reconstruct decisions without timing, door joints, success, or RNG.
        again = replay.update(r['step'], r['feedback']['tracking'], r['feedback']['contacts'],
                              r['queue_after'], safe=not r['guard_failures'] and
                              not r['terminated'] and not r['truncated'] and r['step'] != 600)
        for key in again:
            assert again[key] == d[key], (r['step'], key)
    m['deterministic_decision_replay_exact'] = True
    analyses = []
    for r in events:
        step = r['step']
        decision = r['feedback']['decision']
        assert step < len(rs) and rs[step]['inference'] and rs[step]['queue_before'] == 0
        next_chunk = next(read(p) for p in folder.glob('chunk-*.json') if read(p)['before_step'] == step+1)
        old, new = np.array(decision['discarded_actions']), np.array(next_chunk['all_actions'])
        n = len(old)
        action_diff = new[:n]-old
        contact_pairs = []
        for p in r['feedback']['contacts']['pairs']:
            contact_pairs.append(dict(**p, geoms=[catalog[g]['name'] for g in p['geom_ids']],
                                     bodies=[catalog[g]['body_ancestors'] for g in p['geom_ids']]))
        analyses.append(dict(step=step, simulation_time=decision['simulation_time'],
             reason=decision['reason'], discarded_actions=n,
             preceding_five=[dict(step=x['step'], tracking=x['feedback']['tracking'], contacts=x['feedback']['contacts'])
                             for x in rs[step-5:step]],
             contact_pairs=contact_pairs, old_next_action=old[0].tolist(), new_next_action=new[0].tolist(),
             replacement_translation_rms_public_units=float(np.sqrt(np.mean(action_diff[:,5:8]**2))),
             replacement_all_dimensions_max_abs=float(np.max(np.abs(action_diff))),
             before_20=window(rs, angles, max(0,step-20),step),
             after_20=window(rs, angles, step,min(len(rs),step+20))))
    # Every executed action must be the corresponding prediction row (including early refills).
    chunk = None
    j = 0
    chunks = {read(p)['before_step']: read(p) for p in folder.glob('chunk-*.json')}
    for r in rs:
        if r['inference']:
            chunk = chunks[r['step']]['all_actions']
            j = 0
        np.testing.assert_array_equal(r['action'], chunk[j])
        j += 1
    m['predicted_executed_actions_exact'] = True
    m['final_cameras_nonblack'] = True
    return m, analyses, rs, angles


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args()
    out=args.output.resolve()
    pairs = (read(out/'result.json')['gate']['pairs'] if (out/'result.json').exists()
             else read(out/'partial-pairs.json'))
    valid_seeds = [p['seed'] for p in pairs]
    complete = valid_seeds == [43,44,45,46]
    excluded = read(out/'result.json').get('excluded_attempts', []) if (out/'result.json').exists() else []
    for p in out.glob('*/summary.json'):
        s=read(p)
        if s['termination_reason'] not in ('success','registered_horizon_600'):
            excluded.append(dict(episode=p.parent.name, termination=s['termination_reason'],
                                 actions=s['simulator_steps'], inference_calls=s['inference_calls'],
                                 exception=s.get('exception')))
    status = dict(complete=complete, planned_pairs=4, completed_pairs=len(pairs),
                  valid_seeds=valid_seeds, missing_seeds=sorted(set([43,44,45,46])-set(valid_seeds)),
                  excluded_attempts=excluded, candidate_46_executed=(out/'candidate-46/run-started.json').exists())
    gate = acceptance(pairs, read(out/'frozen-config.json')['acceptance'])
    dump(out/('analyzed-result.json' if complete else 'partial-result.json'), dict(status=status, gate=gate))
    all_metrics, event_analysis, prefixes = {}, {}, {}
    for seed in valid_seeds:
        records={}
        for arm in ('baseline','candidate'):
            folder=ROOT/next(p for p in pairs if p['seed'] == seed)[arm]['path']
            m, events, rs, angles=analyze(folder)
            all_metrics[f'{arm}-{seed}']=m
            event_analysis[f'{arm}-{seed}']=events
            records[arm]=rs
            with (out/f'events-{arm}-{seed}.jsonl').open('w') as f:
                for e in events:
                    f.write(json.dumps(e)+'\n')
        first=event_analysis[f'candidate-{seed}']
        n=min(first[0]['step'] if first else 600, *(len(v) for v in records.values()))
        prefixes[str(seed)] = dict(compared_actions=n,
             first_action_max_abs=float(np.max(np.abs(np.array(records['baseline'][0]['action'])-
                                                       np.array(records['candidate'][0]['action'])))),
             max_action_difference=float(np.max(np.abs(np.array([r['action'] for r in records['baseline'][:n]])-
                                                       np.array([r['action'] for r in records['candidate'][:n]])))),
             max_qpos_difference=float(np.max(np.abs(np.array([r['physical']['qpos'] for r in records['baseline'][:n]])-
                                                     np.array([r['physical']['qpos'] for r in records['candidate'][:n]])))))
    dump(out/'analysis.json', dict(metrics=all_metrics, events=event_analysis, prefixes=prefixes,
                                  status=status, acceptance=gate, preservation=read(out/'preservation.json')))
    fields=['success','actions','closing_deg','best_closing_deg','eef_path_m','peak_contact_normal_N',
            'fridge_loaded_actions','inference_calls','replan_events','discarded_actions','inference_total_seconds',
            'rollout_seconds','worker_seconds','simulated_seconds','peak_cuda_allocated']
    with (out/'paired-results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['episode','final_angles_deg']+fields)
        writer.writeheader()
        for key,m in all_metrics.items():
            writer.writerow(dict(episode=key,final_angles_deg=json.dumps(m['final_angles_deg']),**{k:m[k] for k in fields}))
    print(json.dumps(dict(metrics=all_metrics,prefixes=prefixes,gate=gate['reasons'],status=status),indent=2))


if __name__ == '__main__':
    main()
