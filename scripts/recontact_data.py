"""Causal, geometry-specific recontact filtering of saved executed trajectories."""
from pathlib import Path
import hashlib
import numpy as np
from instruction_control_audit import read, dump, sha
from recontact_self_imitation import OUT, CAMERAS, rows, degrees, contacts, eligible, advance


def window_test(records, loaded, pre, door):
    """pre is one-based BEFORE action pre; outcomes never affect anchor eligibility."""
    end=min(pre+49,len(records))
    start_angle=degrees(records[pre-2]['fridge'])[door]
    final_angle=degrees(records[end-1]['fridge'])[door]
    recon=[i for i in range(pre,end+1) if loaded[i-1][door]]
    success=[i for i in range(pre,end+1) if records[i-1]['fridge']['success']]
    closure=abs(start_angle)-abs(final_angle)
    return dict(start=pre,end=end,executed_actions=end-pre+1,first_recontact=recon[0] if recon else None,
        start_deg=start_angle,end_deg=final_angle,net_closure_deg=closure,
        success_in_window=bool(success),first_success=success[0] if success else None,
        positive=bool(recon) and (closure>=2. or bool(success)))


def inspect_source(folder, historical=False):
    plan=read(OUT/'registered-plan.json');records=rows(folder/'steps.jsonl')
    manifest=read(folder/'manifest.json');reset=read(folder/'reset.json')
    catalog=read(folder/'geometry-catalog.json')
    if historical:
        seed=manifest['seed'];identity='historical-A-ID-'+str(seed)
        assert manifest['resolved_instruction']=='CloseFridge' and manifest['model_sha256']==plan['model_hashes']['frozen']
        assert all(manifest['checkpoint_file_hashes'][n]==h for n,h in plan['processors'].items())
        assert manifest['execution_horizon']==50
        summary=read(folder/'summary.json')
        valid=summary['termination_reason']=='success' and summary['final_fridge']['success']
        valid=valid and all(r['finite'] and not r['guard_failures'] for r in records)
        observed={int(p.stem.split('-')[-1]):p for p in folder.glob('observation-*.npz')}
        hand={g['id'] for g in catalog if g['name'].startswith('gripper0_right')}
        doors={d:{g['id'] for g in catalog if d.removesuffix('_joint') in g['body_ancestors']} for d in reset['fridge']['qpos']}
        fridge={g['id'] for g in catalog if g['name'] in set(read(folder/'study-fixture.json')['fridge_contact_geoms'])}
        robot={g['id'] for g in catalog if g['robot']}
        contact_rows=rows(folder/'contacts.jsonl');assert len(contact_rows)==len(records)
        contact=[]
        for i,r in enumerate(contact_rows,1):
            assert r['step']==i and len(r['samples'])==25
            contact.append(contacts(r['samples'],robot,fridge,doors,hand))
        valid=valid and max(c['peaks']['all'] for c in contact)<=1500
        predicted=[]
        for p in sorted(folder.glob('chunk-*.json')):
            chunk=read(p);assert chunk['before_step']==len(predicted)+1
            predicted.extend(chunk['all_actions'])
        assert np.array_equal(np.array(predicted[:len(records)],np.float32),np.array([r['action'] for r in records],np.float32))
        relevant=[folder/n for n in ['manifest.json','summary.json','scene.json','reset.json','steps.jsonl','contacts.jsonl','geometry-catalog.json','executed-script.py.txt']]
    else:
        result=read(folder/'result.json');seed=result['job']['seed'];identity=result['job']['id']
        assert result['job']['model']=='frozen' and manifest['checkpoint_sha256']==plan['model_hashes']['frozen']
        valid=result['execution_validity'] and not result['hard_stop']
        doors=read(folder/'geometry-spec.json')['doors']
        contact=[r['contact'] for r in records]
        observed={int(p.stem):p for p in (folder/'observations').glob('*.npz')}
        predicted=[]
        for p in sorted(folder.glob('actions-*.npy')):predicted.extend(np.load(p))
        assert np.array_equal(np.array(predicted[:len(records)],np.float32),np.array([r['action'] for r in records],np.float32))
        relevant=[folder/n for n in ['manifest.json','result.json','scene.json','reset.json','steps.jsonl','contacts.jsonl','geometry-spec.json']]
    assert all(r['step']==i+1 for i,r in enumerate(records))
    assert len(records)<=600
    success=bool(records[-1]['fridge']['success'])
    loaded=[c['loaded'] for c in contact]
    seen={d:False for d in doors};gap={d:0 for d in doors};events=[];candidates=[]
    for pre,r in enumerate(records,1):
        ds=eligible(seen,gap,records[pre-2]['fridge']['success'] if pre>1 else reset['fridge']['success'])
        if ds and pre in observed:
            for d in ds:
                event=dict(pre_action=pre,door=d,prior_no_contact_actions=gap[d],**window_test(records,loaded,pre,d))
                event['qualified']=valid and success and event['positive']
                candidates.append(event)
                if event['qualified']:events.append(event)
        advance(seen,gap,loaded[pre-1])
    samples=[];sample_pres={e['pre_action']:'recontact_anchor' for e in events}
    for event in events:
        for pre in observed:
            if pre>event['end'] and pre<=len(records) and (pre-1)%50==0:
                sample_pres.setdefault(pre,'completion_after_qualified_recontact')
    for pre,kind in sorted(sample_pres.items()):
        p=observed[pre];end=min(pre+49,len(records));window=records[pre-1:end]
        with np.load(p) as ob:
            assert ob['state'].shape==(16,) and ob['state'].dtype==np.float32
            expected=records[pre-2].get('returned_state',records[pre-2].get('state')) if pre>1 else reset['state']
            assert np.array_equal(ob['state'],np.asarray(expected,np.float32))
            for cam in CAMERAS:
                assert ob[cam].shape==(256,256,3) and ob[cam].dtype==np.uint8 and ob[cam].max()>0
                if historical:
                    expected_hash=read(folder/f'input-{pre:04d}.json')['cameras'][cam]['sha256']
                    assert hashlib.sha256(ob[cam].tobytes()).hexdigest()==expected_hash
        actions=np.asarray([r['action'] for r in window],dtype=np.float32)
        assert actions.shape==(end-pre+1,12) and np.isfinite(actions).all()
        samples.append(dict(id=f'{identity}-pre{pre}',source=identity,seed=seed,kind=kind,
            observation=str(p),observation_sha256=sha(p),observation_encoding='original lossless uint8 npz',
            pre_action=pre,last_action=end,valid_actions=len(actions),actions=actions.tolist(),
            action_is_pad=[False]*len(actions)+[True]*(50-len(actions)),
            internal_action_dimension_is_pad=[False]*12+[True]*20,
            action_bytes_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
            instruction='CloseFridge',source_success=True,success_in_window=any(r['fridge']['success'] for r in window),
            action_representation='saved postprocessor public output; verified equal to executed action; no inverse threshold/clipping; normalize once in original preprocessor'))
    return dict(id=identity,folder=str(folder),seed=seed,historical=historical,valid=bool(valid),
        success=success,actions=len(records),initial_deg=degrees(reset['fridge']),final_deg=degrees(records[-1]['fridge']),
        qualified=bool(events),candidates=candidates,events=events,samples=samples,
        input_hashes={str(p):sha(p) for p in relevant},
        scene=read(folder/'scene.json')['ep_meta'],video=str(folder/'rollout.mp4'))


def build_index():
    plan=read(OUT/'registered-plan.json');sources=[];invalid=[]
    for path in plan['historical_sources']:sources.append(inspect_source(Path(path),True))
    for entry in plan['collection_order']:
        p=OUT/'collection'/f'collection-{entry["seed"]}-r{entry["replicate"]}-frozen'
        if not (p/'result.json').exists():continue
        r=read(p/'result.json')
        if not r['execution_validity']:
            invalid.append(dict(path=str(p),result=r));continue
        sources.append(inspect_source(p))
    good=[s for s in sources if s['qualified']];seeds=sorted({s['seed'] for s in good})
    samples=[x for s in good for x in s['samples']]
    assert len({s['id'] for s in sources})==len(sources)
    result=dict(qualified_source_count=len(good),qualified_seeds=seeds,independent_reset_scene_count=len(seeds),
        gate_passed=len(good)>=8 and len(seeds)>=4,event_count=sum(len(s['events']) for s in good),
        sample_count=len(samples),sources=sources,samples=samples,invalid_attempts=invalid,
        caveat='Completion samples are not additional causal recovery anchors or source trajectories; original dataset scene overlap unknown.')
    dump(OUT/'data-index.json',result)
    return result


if __name__=='__main__':
    result=build_index();print({k:v for k,v in result.items() if k not in ['sources','samples','invalid_attempts']})
