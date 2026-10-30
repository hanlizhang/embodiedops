"""Read-only, whole-trajectory v2 split and causal held-out state selection.

prepare() reads immutable v1 artifacts and returns serializable structures. It
does not write files, import a policy, instantiate a simulator, or collect data.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

V1 = Path('D:/EmbodiedOpsOutputs/recontact-self-imitation-v1')
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(value, key=None):
    if isinstance(value, dict):
        return {k: canonical(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if key == 'groups_containing_sampled_obj':
            return sorted(value)
        return [canonical(v) for v in value]
    return value


def array_hash(*arrays):
    """Shape-framed C-order little-endian arrays, independent of JSON formatting."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(str(a.shape).encode('ascii'))
        h.update(str(a.dtype).encode('ascii'))
        h.update(a.tobytes(order='C'))
    return h.hexdigest()


def source_audit(source, plan):
    p = Path(source['folder'])
    for filename, expected in source['input_hashes'].items():
        assert sha(Path(filename)) == expected, f'Changed v1 input: {filename}'
    manifest = read(p/'manifest.json')
    reset = read(p/'reset.json')
    scene = read(p/'scene.json')
    rows = records(p/'steps.jsonl')
    assert rows and len(rows) == source['actions'] <= 600
    assert all(r['step'] == i+1 for i, r in enumerate(rows))
    assert source['valid'] and source['success'] and source['qualified']
    assert not any(r['fridge']['success'] for r in rows[:-1]) and rows[-1]['fridge']['success']
    assert all(not r['guard_failures'] for r in rows)
    if source['historical']:
        assert manifest['model_sha256'] == plan['model_hashes']['frozen']
        assert manifest['resolved_instruction'] == 'CloseFridge'
        assert manifest['model_revision'] == plan['revision']
        origin = p.resolve()
    else:
        job = manifest['job']
        assert job['model'] == 'frozen' and job['kind'] == 'collection'
        assert job['prefix'] == 0 and job['reference'] is None
        assert manifest['checkpoint_sha256'] == plan['model_hashes']['frozen']
        origin = Path(job['folder']).resolve()
    actions = np.asarray([r['action'] for r in rows], dtype='<f4')
    states = np.asarray([r.get('returned_state', r.get('state')) for r in rows], dtype='<f4')
    qpos = np.asarray([r['physical']['qpos'] for r in rows], dtype='<f8')
    qvel = np.asarray([r['physical']['qvel'] for r in rows], dtype='<f8')
    assert actions.shape == (len(rows), 12) and states.shape == (len(rows), 16)
    assert all(np.isfinite(a).all() for a in [actions, states, qpos, qvel])
    commands = []
    for a, row in zip(actions, rows):
        command = np.concatenate([a[5:11], [1 if a[11] >= .5 else -1],
                                  a[:4], [1 if a[4] >= .5 else -1]])
        if not source['historical']:
            assert np.array_equal(command, row['controller_input'])
        commands.append(command)
    commands = np.asarray(commands, dtype='<f8')
    reset_arrays = [np.asarray(reset['physical'][k], dtype='<f8') for k in ['qpos', 'qvel']]
    reset_arrays.append(np.asarray(reset['state'], dtype='<f4'))
    # Historical exports also carry presentation-only top-level robot metadata.
    # Compare the reset scene and controller fields shared by both formats.
    scene_identity = {k: scene[k] for k in ('ep_meta', 'controller')}
    scene_hash = hashlib.sha256(json.dumps(canonical(scene_identity), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    reset_hash = array_hash(*reset_arrays)
    a_hash = array_hash(actions)
    c_hash = array_hash(commands)
    st = (p/'steps.jsonl').stat()
    fingerprints = dict(length=len(rows), resolved_source=str(p.resolve()), origin=str(origin),
        file_identity=[st.st_dev, st.st_ino], actions_sha256=a_hash, controller_commands_sha256=c_hash,
        states_sha256=array_hash(states), complete_trace_sha256=array_hash(actions, commands, states, qpos, qvel),
        reset_sha256=reset_hash, scene_controller_sha256=scene_hash,
        action_reset_playback_sha256=hashlib.sha256((a_hash+reset_hash+scene_hash).encode()).hexdigest(),
        controller_reset_playback_sha256=hashlib.sha256((c_hash+reset_hash+scene_hash).encode()).hexdigest(),
        steps_file_sha256=sha(p/'steps.jsonl'), manifest_sha256=sha(p/'manifest.json'))
    sample_signatures = {}
    for sample in source['samples']:
        assert sample['source'] == source['id'] and sample['seed'] == source['seed']
        assert sample['kind'] in ('recontact_anchor', 'completion_after_qualified_recontact')
        assert sample['instruction'] == 'CloseFridge' and sample['source_success']
        pre, end, n = sample['pre_action'], sample['last_action'], sample['valid_actions']
        assert 1 <= pre <= end <= len(rows) and n == end-pre+1 <= 50
        assert end == min(pre+49, len(rows))
        action = np.asarray(sample['actions'], dtype='<f4')
        assert np.array_equal(action, actions[pre-1:end]) and action.shape == (n, 12)
        assert hashlib.sha256(action.tobytes()).hexdigest() == sample['action_bytes_sha256']
        assert sample['action_is_pad'] == [False]*n+[True]*(50-n)
        assert sample['internal_action_dimension_is_pad'] == [False]*12+[True]*20
        observation = Path(sample['observation'])
        assert sha(observation) == sample['observation_sha256']
        h = hashlib.sha256()
        with np.load(observation) as ob:
            assert ob['state'].shape == (16,) and ob['state'].dtype == np.float32
            assert np.array_equal(ob['state'], states[pre-2] if pre > 1 else reset_arrays[-1])
            for camera in CAMERAS:
                assert ob[camera].shape == (256, 256, 3) and ob[camera].dtype == np.uint8
                assert ob[camera].max() > 0
            for key in sorted(ob.files):
                h.update(key.encode()); h.update(ob[key].tobytes())
        h.update(action.tobytes())
        sample_signatures[sample['id']] = h.hexdigest()
    return dict(fingerprints=fingerprints, rows=rows, reset=reset, scene=scene,
                sample_signatures=sample_signatures)


def first_causal_saved_anchor(source, audited):
    """Recompute raw active contacts; selection uses no future window outcome."""
    p = Path(source['folder'])
    spec = read(p/'geometry-spec.json')
    hand = set(spec['gripper_geoms'])
    doors = {k: set(v['geoms']) for k, v in spec['doors'].items()}
    seen = {d: False for d in doors}
    gap = {d: 0 for d in doors}
    last_contact = {d: None for d in doors}
    observed = {int(f.stem): f for f in (p/'observations').glob('*.npz')}
    contacts = records(p/'contacts.jsonl')
    rows = audited['rows']
    assert len(contacts) == len(rows)
    selected = None
    for i, (row, contact) in enumerate(zip(rows, contacts), 1):
        assert row['step'] == contact['step'] == i and len(contact['samples']) == 25
        before_success = rows[i-2]['fridge']['success'] if i > 1 else audited['reset']['fridge']['success']
        candidates = sorted(d for d in doors if seen[d] and gap[d] >= 5)
        if selected is None and not before_success and i in observed and candidates:
            d = candidates[0]
            selected = dict(seed=source['seed'], source_id=source['id'], reference=str(p.resolve()),
                pre_action=i, prefix=i-1, target_door=d, doors=[d], observation=str(observed[i].resolve()),
                observation_sha256=sha(observed[i]),
                past_contact_evidence=dict(last_effective_contact_action=last_contact[d],
                    consecutive_absent_actions=gap[d], absent_action_interval=[i-gap[d], i-1],
                    last_five_action_indices=list(range(i-5, i)), success_before_anchor=False,
                    normal_force_threshold_N=1., active_constraint_required=True,
                    hand_geom_ids=sorted(hand), target_door_geom_ids=sorted(doors[d]),
                    contacts_sha256=sha(p/'contacts.jsonl'),
                    boundary_simulation_time=rows[i-2]['physical']['time'],
                    boundary_door_angle_deg=float(np.rad2deg(rows[i-2]['fridge']['qpos'][d])),
                    selection='First time-ordered saved causal anchor; no future recontact/closure/student outcome used.'))
        loaded = {d: False for d in doors}
        for sample in contact['samples']:
            for c in sample['robot_contacts']:
                if c['constraint_address'] < 0 or c['force_N_contact_frame'][0] < 1.:
                    continue
                ids = set(c['geom_ids'])
                for d, door_ids in doors.items():
                    if ids & hand and ids & door_ids:
                        loaded[d] = True
        assert loaded == row['contact']['loaded'], 'Saved contact flag disagrees with raw physics contacts'
        for d in doors:
            if loaded[d]:
                seen[d] = True; gap[d] = 0; last_contact[d] = i
            else:
                gap[d] += 1
    assert selected is not None, 'Selected whole held-out source has no recorded causal anchor'
    with np.load(selected['observation']) as ob:
        expected = np.asarray(rows[selected['prefix']-1]['state'], dtype=np.float32)
        assert np.array_equal(ob['state'], expected)
        for camera in CAMERAS:
            assert ob[camera].shape == (256, 256, 3) and ob[camera].dtype == np.uint8
    return selected


def prepare():
    plan = read(V1/'registered-plan.json')
    result = read(V1/'result.json')
    index = read(V1/'data-index.json')
    ledger = records(V1/'execution-list.jsonl')
    status = read(V1/'collection-status.json')
    assert sha(V1/'data-index.json') == status['index_sha256']
    assert result['outcome'] == 'data_insufficient_no_training' and result['new_optimizer_updates'] == 0
    expected_order = [(r['seed'], r['replicate']) for r in plan['collection_order']]
    actual_order = [(r['job']['seed'], r['job']['replicate']) for r in ledger]
    assert actual_order == expected_order and len(ledger) == 24
    assert all(r['job']['kind'] == 'collection' and r['job']['model'] == 'frozen' for r in ledger)
    distribution = []
    for seed in sorted({r['job']['seed'] for r in ledger}):
        group = [r['result'] for r in ledger if r['job']['seed'] == seed]
        distribution.append(dict(seed=seed, attempts=len(group), raw_successes=sum(r['success'] for r in group),
            valid=sum(r['execution_validity'] for r in group), hard_stops=sum(r['hard_stop'] for r in group)))
    assert all(r['attempts'] == 4 for r in distribution)
    assert sum(r['raw_successes'] for r in distribution) == result['new_collection_successes'] == 11
    assert {r['seed'] for r in distribution if r['raw_successes']} == {44, 47, 49}
    qualified = [s for s in index['sources'] if s['qualified']]
    assert len(qualified) == index['qualified_source_count'] == result['qualified_sources']
    ids = [s['id'] for s in qualified]
    assert len(set(ids)) == len(ids)
    audited = {s['id']: source_audit(s, plan) for s in qualified}
    parents = {s: s for s in ids}
    def find(k):
        while parents[k] != k:
            parents[k] = parents[parents[k]]; k = parents[k]
        return k
    merges = []
    for i, a in enumerate(ids):
        fa = audited[a]['fingerprints']
        for b in ids[i+1:]:
            fb = audited[b]['fingerprints']
            fields = ['resolved_source', 'origin', 'file_identity', 'complete_trace_sha256',
                      'action_reset_playback_sha256', 'controller_reset_playback_sha256']
            reasons = [k for k in fields if fa[k] == fb[k]]
            if reasons:
                parents[find(b)] = find(a); merges.append(dict(a=a, b=b, reasons=reasons))
    grouped = defaultdict(list)
    for k in ids: grouped[find(k)].append(k)
    selected = []
    for seed in [44, 47, 49]:
        for run_index, entry in enumerate(ledger, 1):
            source = next((s for s in qualified if s['id'] == entry['job']['id'] and s['seed'] == seed and not s['historical']), None)
            if source is not None:
                selected.append(dict(seed=seed, source_id=source['id'], original_collection_run_index=run_index,
                                     replicate=entry['job']['replicate'])); break
        else: raise AssertionError(f'No qualified new source for held-out seed {seed}')
    held_roots = {find(s['source_id']) for s in selected}
    selected_ids = [s['source_id'] for s in selected]
    groups = []
    for root, members in grouped.items():
        split = 'holdout' if root in held_roots else 'training'
        preferred = [s for s in selected_ids if s in members]
        representative = preferred[0] if preferred else members[0]
        gid = 'source-group-'+hashlib.sha256(json.dumps(sorted(members)).encode()).hexdigest()[:16]
        groups.append(dict(group_id=gid, members=members, representative=representative, split=split))
    group_by_source = {s:g for g in groups for s in g['members']}
    source_by_id = {s['id']:s for s in qualified}
    def subset(name):
        sources = [deepcopy(source_by_id[g['representative']]) for g in groups if g['split'] == name]
        for source in sources:
            source['deduplication_group'] = group_by_source[source['id']]['group_id']
        samples = [deepcopy(s) for source in sources for s in source['samples']]
        kinds = Counter(s['kind'] for s in samples)
        return dict(version='recontact-self-imitation-v2', split=name, sources=sources, samples=samples,
            qualified_source_count=len(sources), qualified_seeds=sorted({s['seed'] for s in sources}),
            sample_count=len(samples), event_count=kinds['recontact_anchor'],
            recontact_window_count=kinds['recontact_anchor'], completion_sample_count=kinds['completion_after_qualified_recontact'],
            valid_action_labels_total=sum(s['valid_actions'] for s in samples),
            gate_passed=True, gate_basis='v2 explicitly removes the v1 four-scene training entry gate; only data integrity and grouped split checks apply.',
            caveat='Overlapping windows and completion samples do not add independent source trajectories or reset scenes.')
    train = subset('training'); hold = subset('holdout')
    assert train['sources'], 'Cannot sample a recontact mixture with zero training sources after grouped exclusion'
    train_sigs = {audited[s['source']]['sample_signatures'][s['id']] for s in train['samples']}
    hold_sigs = {audited[s['source']]['sample_signatures'][s['id']] for s in hold['samples']}
    assert not train_sigs & hold_sigs, 'Exact observation/action sample crosses whole-trajectory split'
    states = [first_causal_saved_anchor(source_by_id[s['source_id']], audited[s['source_id']]) for s in selected]
    for state in states: state['source_group'] = group_by_source[state['source_id']]['group_id']
    split = dict(rule='Per seed44,47,49 choose first qualified NEW source in original collection run order; exclude its entire origin/duplicate group. De-duplicate each remaining group to one representative.',
        selected_holdout_new_sources=selected, groups=groups, duplicate_merge_evidence=merges,
        source_assignments=[dict(source_id=s, group_id=group_by_source[s]['group_id'], split=group_by_source[s]['split'],
            representative=group_by_source[s]['representative'] == s) for s in ids],
        fingerprints={s:audited[s]['fingerprints'] for s in ids},
        fingerprint_method='SHA256 over shape/dtype-framed little-endian C-order actions float32, forward controller commands float64, state float32, qpos/qvel float64; exact replay grouping also uses complete actions or controller commands plus reset arrays and canonical scene/controller identity.',
        training_source_ids=[s['id'] for s in train['sources']], holdout_source_ids=[s['id'] for s in hold['sources']],
        holdout_all_alias_source_ids=[s for g in groups if g['split']=='holdout' for s in g['members']],
        training_sample_ids=[s['id'] for s in train['samples']], holdout_sample_ids=[s['id'] for s in hold['samples']],
        counts={name:{k:subset_index[k] for k in ['qualified_source_count','sample_count','recontact_window_count','completion_sample_count','valid_action_labels_total']}
                for name,subset_index in [('training',train),('holdout',hold)]},
        exact_cross_split_sample_duplicates=0, no_cross_split_source_groups=True)
    test_audit = dict(test_seeds=[60,61,62,63,64,65], conflicts=[],
        training_recontact_reset_seeds=train['qualified_seeds'], v1_optimizer_updates=result['new_optimizer_updates'],
        v1_executed_job_kinds=sorted({r['job']['kind'] for r in ledger}),
        v1_test_conditions_executed=sum(bool(r['models']) for r in result['full_task']),
        v1_test_seeds_were_planned_only=True,
        evidence_scope='Current v1 source data, frozen plan, result and complete execution ledger; no broad dataset or unrelated checkpoint search.',
        limitation=plan['split_audit']['limitation'])
    assert not set(test_audit['test_seeds']) & set(train['qualified_seeds'])
    assert test_audit['v1_test_conditions_executed'] == 0
    provenance = dict(v1=str(V1), files={str(V1/n):sha(V1/n) for n in ['result.json','data-index.json','registered-plan.json','execution-list.jsonl','collection-status.json']},
        sources={s['id']:dict(folder=s['folder'], resolved_folder=audited[s['id']]['fingerprints']['resolved_source'],
            historical=s['historical'], seed=s['seed'], input_hashes=s['input_hashes']) for s in qualified},
        v1_outcome_unchanged=result['outcome'], physical_or_model_execution=False)
    return dict(training_index=train, holdout_index=hold, split=split, recovery_states=states,
                distribution=distribution, source_provenance=provenance, test_seed_audit=test_audit)


if __name__ == '__main__':
    prepared = prepare()
    print(json.dumps(dict(distribution=prepared['distribution'], counts=prepared['split']['counts'],
        selected_holdout=prepared['split']['selected_holdout_new_sources'],
        anchors=[{k:s[k] for k in ['seed','source_id','pre_action','prefix','target_door']} for s in prepared['recovery_states']],
        duplicate_groups=[g for g in prepared['split']['groups'] if len(g['members'])>1]), indent=2))
