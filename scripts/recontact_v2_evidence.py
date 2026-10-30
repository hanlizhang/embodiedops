"""Independent, read-only contact and restored-boundary evidence audits.

No environment or policy is imported. Prefix contacts inform causal recontact
history only; every reported force peak is restricted to newly executed actions.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
PEAKS = ('all', 'fridge', 'other_fridge', 'self', 'target_hand_door')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _compare(actual, expected, path, tolerance=1e-8):
    if isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys(), path + ': keys differ'
        return max((_compare(v, expected[k], path + '.' + k, tolerance) for k, v in actual.items()), default=0.)
    if isinstance(actual, (list, tuple)):
        assert isinstance(expected, (list, tuple)) and len(actual) == len(expected), path + ': length differs'
        return max((_compare(a, b, path + '[' + str(i) + ']', tolerance) for i, (a, b) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        assert isinstance(expected, (int, float)) and not isinstance(expected, bool), path + ': numeric type differs'
        if actual == expected:
            return 0.  # Equal static controller +/-infinity is permitted; NaN never is.
        error = abs(float(actual) - float(expected))
        assert math.isfinite(error) and error <= tolerance, f'{path}: actual={actual}, expected={expected}, error={error}'
        return error
    assert actual == expected, f'{path}: actual={actual}, expected={expected}'
    return 0.


def _geometry(folder):
    g = _read(folder / 'contact-geometry.json')
    spec = _read(folder / 'geometry-spec.json')
    catalog = _read(folder / 'geometry-catalog.json')
    ids = [int(c['id']) for c in catalog]
    assert len(ids) == len(set(ids)), 'Duplicate catalog geometry identity'
    by_id = {int(c['id']): c for c in catalog}
    robot = set(g['robot_geom_ids']); hand = set(g['hand_geom_ids']); fridge = set(g['fridge_geom_ids'])
    doors = {d: set(gs) for d, gs in g['task_door_geom_ids'].items()}
    target = g['selected_target_door']
    assert robot and hand and fridge and doors and hand <= robot
    assert robot == {c['id'] for c in catalog if c['robot']}, 'Robot IDs differ from saved catalog'
    assert hand == set(spec['gripper_geoms']), 'Hand IDs differ from geometry spec'
    assert doors == {d: set(v['geoms']) for d, v in spec['doors'].items()}, 'Door IDs differ from geometry spec'
    assert fridge == {c['id'] for c in catalog if c['name'] in set(g['fridge_geom_names'])}, 'Fridge names/IDs differ'
    assert all(gs <= by_id.keys() and not hand & gs for gs in doors.values())
    assert robot <= by_id.keys() and fridge <= by_id.keys()
    assert target is None or target in doors
    metric = {target: doors[target]} if target else doors
    assert set(g['metric_target_geom_ids']) == set().union(*metric.values())
    return dict(robot=robot, hand=hand, fridge=fridge, doors=doors, metric=metric, target=target, by_id=by_id)


def _action_contact(samples, geometry):
    """Recompute from active solver rows, independently of rollout helpers."""
    robot, hand, fridge, doors, metric = (geometry[k] for k in ('robot', 'hand', 'fridge', 'doors', 'metric'))
    peaks = dict.fromkeys(PEAKS, 0.)
    per_door = dict.fromkeys(doors, 0.)
    first = dict.fromkeys(doors, None)
    raw_peak = 0.
    active_rows = inactive_rows = 0
    previous_time = None
    for sample in samples:
        t = float(sample['time'])
        assert math.isfinite(t) and (previous_time is None or t > previous_time), 'Nonmonotonic contact substep time'
        previous_time = t
        for c in sample['robot_contacts']:
            pair = c['geom_ids']
            assert len(pair) == 2 and set(pair) <= geometry['by_id'].keys(), 'Unknown contact geometry identity'
            ids = set(pair)
            assert ids & robot, 'Nonrobot pair included in robot_contacts'
            normal = float(c['force_N_contact_frame'][0])
            assert math.isfinite(normal), 'Nonfinite contact normal force'
            force = max(0., normal); raw_peak = max(raw_peak, force)
            if c['constraint_address'] < 0:
                inactive_rows += 1
                continue
            active_rows += 1
            peaks['all'] = max(peaks['all'], force)
            if ids <= robot:
                peaks['self'] = max(peaks['self'], force)
            hit_metric = False
            for door, door_ids in doors.items():
                if ids & hand and ids & door_ids:
                    per_door[door] = max(per_door[door], force)
                    if force >= 1. and first[door] is None:
                        first[door] = t
                    if door in metric:
                        hit_metric = True
                        peaks['target_hand_door'] = max(peaks['target_hand_door'], force)
            if ids & fridge:
                peaks['fridge'] = max(peaks['fridge'], force)
                if not hit_metric:
                    peaks['other_fridge'] = max(peaks['other_fridge'], force)
    loaded = {d: first[d] is not None for d in doors}
    saved_contact = dict(peaks=peaks, loaded={d: loaded[d] for d in metric},
        first={d: first[d] for d in metric}, per_door_N={d: per_door[d] for d in metric},
        task_door_loaded=loaded, task_door_first=first, task_door_peaks_N=per_door)
    return dict(saved=saved_contact, per_door=per_door, first=first, loaded=loaded,
                raw_peak=raw_peak, active_rows=active_rows, inactive_rows=inactive_rows)


def _camera_pair(actual_path, expected_path):
    with np.load(actual_path, allow_pickle=False) as actual, np.load(expected_path, allow_pickle=False) as expected:
        assert actual['state'].shape == expected['state'].shape == (16,)
        assert actual['state'].dtype == expected['state'].dtype == np.float32
        error = float(np.max(np.abs(actual['state'].astype(np.float64) - expected['state'].astype(np.float64))))
        assert math.isfinite(error) and error <= 1e-7, f'Boundary state differs: {error}'
        cameras = {}
        for camera in CAMERAS:
            a, b = actual[camera], expected[camera]
            assert a.dtype == b.dtype == np.uint8 and a.shape == b.shape == (256, 256, 3)
            delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
            cameras[camera] = dict(max=int(delta.max()), mean=float(delta.mean()))
            assert cameras[camera]['max'] <= 1 and cameras[camera]['mean'] <= .001, f'Boundary camera gate: {camera}: {cameras[camera]}'
    return dict(actual=str(actual_path), reference=str(expected_path), actual_sha256=_sha(actual_path),
                reference_sha256=_sha(expected_path), state_max_error=error, cameras=cameras)


def audit_boundary(result):
    job = result['job']; folder = Path(job['folder'])
    diagnostic = dict(passed=None, camera_pairs=[], controller_peer=None, errors=[],
        observable_clocks=dict(recorded=False,comparison='not_recorded',peer=None))
    if not result.get('restoration_validity'):
        diagnostic['reason'] = 'Restoration not declared valid; no boundary acceptance claim.'
        return diagnostic
    try:
        actual = folder / 'actual-boundary.npz'; common = folder / 'common-policy-input.npz'
        assert _sha(actual) == result['actual_boundary_sha256'], 'Actual boundary hash differs'
        assert _sha(common) == result['common_policy_input_sha256'], 'Policy input hash differs'
        diagnostic['camera_pairs'].append(_camera_pair(actual, common))
        reference = job.get('reference')
        if reference:
            historical = Path(reference) / 'observations' / f"{int(job.get('prefix', 0))+1:04d}.npz"
            diagnostic['camera_pairs'].append(_camera_pair(actual, historical))
        boundary = _read(folder / 'boundary-controller.json')
        assert boundary['timestep'] == int(job.get('prefix', 0)), 'Boundary controller counter differs'
        _compare(boundary['physics_time'], result['origin_time'], 'Boundary physics origin')
        clock_path = folder / 'boundary-observable-clocks.json'
        clock_record = diagnostic['observable_clocks']
        clock_record['path'] = str(clock_path)
        clocks = None
        if clock_path.exists():
            clocks = _read(clock_path)
            assert isinstance(clocks, dict), 'Observable clock snapshot must be a mapping'
            clock_record.update(recorded=True,sha256=_sha(clock_path),comparison='no_paired_reference')
        else:
            clock_record['basis'] = 'This older attempt did not record an observable clock snapshot; no clock equality is claimed.'
        peer = job.get('paired_boundary_reference')
        if peer:
            peer_path = Path(peer)
            if peer_path.is_dir():
                peer_controller = peer_path / 'boundary-controller.json'
                diagnostic['camera_pairs'].append(_camera_pair(actual, peer_path / 'actual-boundary.npz'))
            else:
                peer_controller = peer_path
            diagnostic['controller_peer'] = dict(path=str(peer_controller), sha256=_sha(peer_controller),
                max_error=_compare(boundary, _read(peer_controller), 'Peer controller'))
            peer_clock_path = peer_controller.parent / 'boundary-observable-clocks.json'
            clock_record['peer'] = dict(path=str(peer_clock_path),recorded=peer_clock_path.exists())
            if peer_clock_path.exists():
                clock_record['peer']['sha256'] = _sha(peer_clock_path)
                if clocks is not None:
                    clock_record.update(comparison='passed',tolerance=1e-8,
                        max_error=_compare(clocks,_read(peer_clock_path),'Peer observable clocks',1e-8))
                else:
                    clock_record['comparison'] = 'not_compared_current_not_recorded'
            else:
                clock_record.update(comparison='not_recorded_in_peer',
                    basis='The older frozen peer did not record observable clocks; no clock equality is claimed. Existing controller, physical/state and raw-camera checks retain their original thresholds; historical prefix state gates are unchanged.')
        diagnostic['passed'] = True
    except Exception as exc:
        diagnostic.update(passed=False, errors=[f'{type(exc).__name__}: {exc}'])
    return diagnostic


def audit_contacts(result):
    """Return diagnostics; missing/corrupt evidence never raises out of this API."""
    job = result['job']; folder = Path(job['folder']); prefix = int(job.get('prefix', 0))
    diagnostic = dict(id=job['id'], passed=None, errors=[], newly_executed_actions=int(result.get('actions', 0)),
        prefix_actions_excluded_from_peaks=prefix,
        definition='Max nonnegative normal force of active solver constraints; effective hand-door contact >=1N at any substep; exact saved geom IDs.')
    if not result.get('actions'):
        diagnostic['reason'] = 'No newly executed action recorded; no post-boundary contact claim to audit.'
        return diagnostic
    try:
        geometry = _geometry(folder)
        assert geometry['target'] == job.get('target_door'), 'Selected target differs from fixed job'
        steps = _rows(folder / 'steps.jsonl'); raw = _rows(folder / 'contacts.jsonl')
        count = int(result['actions']); end = prefix + count
        assert [s['step'] for s in steps] == list(range(prefix+1, end+1)), 'Saved action coverage differs'
        assert [s['local_step'] for s in steps] == list(range(1, count+1)), 'Saved local action numbering differs'
        assert [s['step'] for s in raw] == list(range(1, end+1)), 'Raw contacts must cover one complete reset/replay/continuation sequence'
        seen = dict.fromkeys(geometry['doors'], False); gap = dict.fromkeys(geometry['doors'], 0)
        peaks = dict.fromkeys(PEAKS, 0.); per_door = dict.fromkeys(geometry['doors'], 0.)
        first_contact = {}; first_recontact = {}; recontact_actions = {}; recontact_events = []
        raw_peak = 0.; active_rows = inactive_rows = 0; substeps = 0; max_error = 0.
        origin = float(result['origin_time'])
        for item in raw:
            index = item['step']; samples = item['samples']
            assert len(samples) == 25, f'Action {index}: expected25 contact substeps, got{len(samples)}'
            con = _action_contact(samples, geometry)
            if index > prefix:
                row = steps[index-prefix-1]
                max_error = max(max_error, _compare(con['saved'], row['contact'], f'action{index}.contact', 1e-9))
                assert abs(float(samples[-1]['time']) - float(row['physical']['time'])) <= 1e-8
                for k, value in con['saved']['peaks'].items():
                    peaks[k] = max(peaks[k], value)
                for d, force in con['per_door'].items():
                    per_door[d] = max(per_door[d], force)
                    if con['first'][d] is not None:
                        first_contact.setdefault(d, con['first'][d] - origin)
                        if seen[d] and gap[d] >= 5:
                            seconds = con['first'][d] - origin
                            first_recontact.setdefault(d, seconds)
                            recontact_actions.setdefault(d, index-prefix)
                            recontact_events.append(dict(door=d, local_action=index-prefix,
                                global_action=index, prior_gap_actions=gap[d], seconds=seconds))
                raw_peak = max(raw_peak, con['raw_peak'])
                active_rows += con['active_rows']; inactive_rows += con['inactive_rows']; substeps += len(samples)
            for d in seen:
                if con['loaded'][d]:
                    seen[d] = True; gap[d] = 0
                else:
                    gap[d] += 1
        max_error = max(max_error, _compare(peaks, result['peaks'], 'result.peaks', 1e-9))
        if raw_peak > 1500.:
            assert result.get('hard_stop'), 'Observed >1500N without recorded hard stop'
        max_error = max(max_error, _compare(first_contact, result['first_contact_seconds'], 'result.first_contact_seconds', 1e-8))
        max_error = max(max_error, _compare(per_door, result['per_door_peaks_N'], 'result.per_door_peaks_N', 1e-9))
        max_error = max(max_error, _compare(first_recontact, result['first_recontact_seconds'], 'result.first_recontact_seconds', 1e-8))
        _compare(recontact_actions, result['first_recontact_action'], 'result.first_recontact_action', 0.)
        _compare(recontact_events, result['recontact_events'], 'result.recontact_events', 1e-8)
        diagnostic.update(passed=True, computed_peaks=peaks, computed_task_door_peaks_N=per_door,
            computed_task_door_first_contact_seconds=first_contact,
            computed_task_door_first_recontact_seconds=first_recontact,
            computed_task_door_first_recontact_action=recontact_actions,
            computed_recontact_events=recontact_events,
            contact_summary_max_error=max_error, active_contact_rows=active_rows,
            inactive_contact_rows_excluded=inactive_rows, new_contact_substeps=substeps,
            raw_robot_normal_peak_including_inactive_N=raw_peak,
            files={name:_sha(folder/name) for name in ['steps.jsonl','contacts.jsonl','geometry-spec.json','geometry-catalog.json','contact-geometry.json']})
    except Exception as exc:
        diagnostic.update(passed=False, errors=[f'{type(exc).__name__}: {exc}'])
    diagnostic['boundary'] = audit_boundary(result)
    if diagnostic['boundary']['passed'] is False:
        diagnostic['passed'] = False
        diagnostic['errors'].extend(diagnostic['boundary']['errors'])
    return diagnostic
