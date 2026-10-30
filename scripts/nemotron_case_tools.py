"""Five bounded tools for the real seed-64 Nemotron demonstration.

Read-only archived evidence by default. The sole mutating tool can launch exactly
one pre-registered two-episode pair; no model-selected shell, path or threshold.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('D:/EmbodiedOpsOutputs/recontact-self-imitation-v2')
MODELS = ('frozen', 'recontact_mix_300')
CASE = {'type': 'string', 'enum': ['seed64']}
SOURCE = {'type': 'string', 'enum': ['archived', 'new']}


def schema(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required,
                       'additionalProperties': False}}}


SCHEMAS = [
    schema('inspect_case', 'Read seed64 experiment identity, fixed model protocol and available evidence. No physics is run.', {'case': CASE}, ['case']),
    schema('inspect_trajectory', 'Read measured door/contact time series and geometry identity sets in an archived trajectory. No simulation.',
           {'case': CASE, 'model': {'type': 'string', 'enum': list(MODELS)},
            'segment': {'type': 'string', 'enum': ['all', 'early', 'middle', 'late']}}, ['case','model','segment']),
    schema('compare_candidates', 'Read actual matched frozen/candidate outcomes; distinguish archived evidence from this run. No simulation.',
           {'case': CASE, 'source': SOURCE}, ['case','source']),
    schema('run_bounded_verification', 'Optional: execute one fixed frozen/candidate seed64 pair, at most 2 episodes of 600 actions and 600 seconds each, after integrity preflight. Use only if archived evidence leaves a concrete verification need; cannot retry.',
           {'case': CASE, 'rationale': {'type': 'string', 'minLength': 10, 'maxLength': 500}}, ['case','rationale']),
    schema('check_acceptance', 'Invoke the independent fixed acceptance gate for the selected evidence. The model cannot change or override its decision. Archived evidence can be sufficient.',
           {'case': CASE, 'source': SOURCE}, ['case','source'])]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def validate_arguments(name, args):
    tool = next((x['function'] for x in SCHEMAS if x['function']['name'] == name), None)
    if tool is None or not isinstance(args, dict):
        raise ValueError('Unknown tool or invalid argument object')
    spec = tool['parameters']
    if set(args) != set(spec['required']):
        raise ValueError('Unexpected or missing tool arguments')
    for key, value in args.items():
        rule = spec['properties'][key]
        if not isinstance(value, str) or ('enum' in rule and value not in rule['enum']):
            raise ValueError('Tool argument outside fixed allowlist')
        if len(value) > rule.get('maxLength', 40) or len(value) < rule.get('minLength', 1):
            raise ValueError('Tool string length outside limit')
    return args


def compact_condition(c):
    fields = ('id','model','seed','execution_validity','restoration_validity','raw_success',
              'qualified_success','hard_stop','actions','initial_deg','final_deg','min_deg',
              'peaks','first_success_action','first_success_seconds','first_contact_seconds',
              'first_recontact_seconds','relative_ceiling_N','relative_pass','stop_reason',
              'inference_seconds','execution_seconds','worker_seconds','video')
    return {k:c.get(k) for k in fields}


class CaseTools:
    schemas = SCHEMAS

    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.plan = read(ARCHIVE/'registered-plan.json')
        self.models = read(ARCHIVE/'frozen-models.json')
        self.result = read(ARCHIVE/'result.json')
        self.conditions = {c['model']:c for c in self.result['conditions']
                           if c['seed']==64 and c['kind']=='full' and c['model'] in MODELS}
        assert set(self.conditions)==set(MODELS)
        self.audit = read(ARCHIVE/'execution-audit.json')
        self.verified_archive = self._match(self.conditions)
        checked={x['id']:x for x in self.audit['checked']}
        self.verified_archive['archived_contact_audit_passed']=all(checked[c['id']]['passed'] for c in self.conditions.values())
        self.verified_archive['passed'] &= self.verified_archive['archived_contact_audit_passed']
        self.policy = read(ROOT/'environment/agent-workflow-v1.json')['acceptance']
        self.gate_hash = sha(ROOT/'scripts/agent_workflow.py')
        self.last_acceptance = None
        self.new_conditions = None
        self.verification_attempted = False
        self.physical = self.output/'physical'
        self._register_verification()

    def _match(self, conditions):
        from phase1d_rollout import canonical_scene
        from recontact_v2_evidence import _camera_pair, _compare, audit_boundary
        a,b = (Path(conditions[m]['attempt_folder']) for m in MODELS)
        ra,rb = (read(p/'result.json') for p in (a,b))
        checks = {'same_scene_controller': canonical_scene(read(a/'scene.json'))==canonical_scene(read(b/'scene.json')),
                  'same_geometry': read(a/'geometry-spec.json')==read(b/'geometry-spec.json') and read(a/'geometry-catalog.json')==read(b/'geometry-catalog.json'),
                  'same_noise_seed_stream': ra['job']['noise_seeds']==rb['job']['noise_seeds'],
                  'same_task_budget': all(r['horizon']==600 and r['job']['seed']==64 for r in (ra,rb))}
        checks['reset_max_error'] = _compare(read(a/'reset.json'),read(b/'reset.json'),'pair reset',1e-8)
        checks['controller_max_error'] = _compare(read(a/'boundary-controller.json'),read(b/'boundary-controller.json'),'pair controller',1e-8)
        checks['camera_gate'] = _camera_pair(b/'actual-boundary.npz',a/'actual-boundary.npz')
        checks['boundaries'] = {m:audit_boundary(r) for m,r in zip(MODELS,(ra,rb))}
        common = min(len(ra['inferences']),len(rb['inferences']))
        checks['same_consumed_native_noise'] = all(ra['inferences'][i]['noise_sha256']==rb['inferences'][i]['noise_sha256'] for i in range(common))
        checks['passed'] = all(checks[k] for k in ['same_scene_controller','same_geometry','same_noise_seed_stream','same_task_budget','same_consumed_native_noise']) and all(x['passed'] for x in checks['boundaries'].values())
        return checks

    def _register_verification(self):
        self.physical.mkdir(exist_ok=False)
        original = deepcopy(self.plan)
        expected = dict(original['source_hashes'])
        for line in (ARCHIVE/'runtime-source-amendments.jsonl').read_text().splitlines():
            item = json.loads(line)
            assert expected[item['path']]==item['old_sha256']
            expected[item['path']] = item['new_sha256']
        for name in ['nemotron_case_tools.py','nemotron_verification_worker.py','agent_workflow.py']:
            expected[str(ROOT/'scripts'/name)]=sha(ROOT/'scripts'/name)
        expected[str(ROOT/'environment/agent-workflow-v1.json')]=sha(ROOT/'environment/agent-workflow-v1.json')
        original.update(version='nemotron-agent-fixed-seed64-verification-v1', source_hashes=expected,
                        output=str(self.physical), registered_unix=time.time(), deadline_unix=time.time()+1800)
        jobs=[]
        for model in MODELS:
            cond = deepcopy(next(c for c in self.plan['conditions'] if c['id']==f'full-64-r1-{model}'))
            folder = self.physical/cond['id']
            reference = Path(self.conditions['frozen']['attempt_folder']) if model=='frozen' else self.physical/'full-64-r1-frozen'
            job = dict(cond, checkpoint=self.models['paths'][model],checkpoint_sha256=self.models['hashes'][model],
                       folder=str(folder),reference=str(reference),paired_boundary_reference=str(reference))
            jobs.append(job)
        original['conditions']=[{k:v for k,v in j.items() if k not in ['folder','checkpoint','checkpoint_sha256','reference','paired_boundary_reference']} for j in jobs]
        original['verification_note']='One optional fixed pair; same v2 seed64/checkpoints/processors/noise/actions. Only output root and fresh wall-clock deadline differ. No training, scene search, retry or model selection.'
        write(self.physical/'registered-plan.json',original)
        write(self.physical/'jobs.json',jobs)
        self.jobs=jobs

    def evidence_manifest(self):
        inputs = [ARCHIVE/n for n in ['registered-plan.json','result.json','frozen-models.json','execution-audit.json']]
        for c in self.conditions.values():
            inputs += [Path(c['attempt_folder'])/n for n in ['result.json','steps.jsonl','reset.json','scene.json','contact-geometry.json','geometry-catalog.json']]
        return {'case':'seed64','archive_root':str(ARCHIVE),'archive_inputs':{str(p):sha(p) for p in inputs},
                'verification_plan':str(self.physical/'registered-plan.json'),
                'verification_plan_sha256':sha(self.physical/'registered-plan.json'),
                'checkpoint_paths_and_hashes':self.models,
                'matched_archive':self.verified_archive,'allowed_tools':[x['function']['name'] for x in SCHEMAS],
                'limits':{'max_new_episodes':2,'max_pair_attempts':1,'actions_per_episode':600,'worker_timeout_seconds':600,'hard_stop_N':1500},
                'acceptance_policy':read(ROOT/'environment/agent-workflow-v1.json')['acceptance']}

    def dispatch(self, name, args):
        validate_arguments(name,args)
        return getattr(self,name)(**args)

    def inspect_case(self, case):
        return {'case':case,'task':'CloseFridge','scene_seed':64,'case_type':'archived regression; human-selected case',
                'models':{m:{'checkpoint':self.models['paths'][m],'sha256':self.models['hashes'][m]} for m in MODELS},
                'protocol':{k:self.plan[k] for k in ['revision','instruction','H','flow_steps','public_hz','processors']},
                'actions_cap':600,'available_evidence':['actual paired results','lossless boundary RGB/state','door/contact time series','geometry identities','independent contact/boundary audit','local videos'],
                'archive_matches':self.verified_archive['passed'],
                'optional_new_verification':'A fresh pair is allowed only if evidence leaves a concrete need; archived comparison and acceptance can suffice.',
                'not_proven':['mechanistic root cause','generalization','successful learned recovery']}

    def inspect_trajectory(self, case, model, segment):
        c=self.conditions[model];folder=Path(c['attempt_folder'])
        with (folder/'steps.jsonl').open(encoding='utf-8') as f:
            rows=[json.loads(line) for line in f if line.strip()]
        lo,hi={'all':(1,600),'early':(1,200),'middle':(201,400),'late':(401,600)}[segment]
        selected=[r for r in rows if lo<=r['step']<=hi]
        g=read(folder/'contact-geometry.json');catalog={x['id']:x for x in read(folder/'geometry-catalog.json')}
        windows=[]
        for start in range(lo,hi+1,50):
            part=[r for r in selected if start<=r['step']<start+50]
            if not part:continue
            before = rows[part[0]['step']-2]['fridge']['qpos'] if part[0]['step']>1 else read(folder/'reset.json')['fridge']['qpos']
            windows.append({'actions':[part[0]['step'],part[-1]['step']],
                'start_deg':{d:math.degrees(x) for d,x in before.items()},
                'end_deg':{d:math.degrees(x) for d,x in part[-1]['fridge']['qpos'].items()},
                'target_contact_actions':sum(any(r['contact']['task_door_loaded'].values()) for r in part),
                'peaks_N':{k:max(r['contact']['peaks'][k] for r in part) for k in ['target_hand_door','other_fridge','self','all']},
                'success':any(r['fridge']['success'] for r in part)})
        intervals=[]
        for loaded in [True,False]:
            start=last=None
            for r in selected:
                active=any(r['contact']['task_door_loaded'].values())
                if active==loaded:
                    if start is None:start=r['step']
                    last=r['step']
                elif start is not None:
                    if last-start+1>=5:intervals.append({'start':start,'end':last,'effective_target_contact':loaded})
                    start=last=None
            if start is not None and last-start+1>=5:intervals.append({'start':start,'end':last,'effective_target_contact':loaded})
        return {'source':'archived','model':model,'segment':segment,'summary':compact_condition(c),'windows':windows,
                'contact_intervals_at_least_5_actions':sorted(intervals,key=lambda x:x['start'])[:40],
                'recontact_events':[e for e in c['recontact_events'] if lo<=e['global_action']<=hi],
                'geometry':{'door_joint_names':list(g['task_door_geom_ids']),
                            'hand_geoms':[{k:catalog[i][k] for k in ['id','name']} for i in g['hand_geom_ids']],
                            'target_door_geom_ids':g['metric_target_geom_ids']},
                'contact_definition':g['effective_contact'],'other_fridge_definition':g['other_fridge'],
                'limitations':'Geometry sets identify the measured categories. These summaries alone do not identify a unique causal collision or prove why the learned action changed.'}

    def _conditions_for(self, source):
        if source=='archived':return self.conditions,self.verified_archive
        if self.new_conditions is None:raise ValueError('No complete valid new verification pair exists')
        return self.new_conditions,self.new_match

    def compare_candidates(self, case, source):
        conditions,match=self._conditions_for(source)
        return {'case':case,'source':source,'new_simulation_in_this_tool':False,'matched_pair':match['passed'],
                'conditions':{m:compact_condition(conditions[m]) for m in MODELS},
                'relative_contact_gate':'candidate all robot active normal peak <=1.25*matched frozen peak+25 N; hard ceiling1500 N',
                'scope':'One fixed scene regression; no claim of benchmark/generalization improvement.'}

    def check_acceptance(self, case, source):
        from agent_workflow import acceptance
        assert sha(ROOT/'scripts/agent_workflow.py')==self.gate_hash,'Independent gate source changed'
        if self.verification_attempted and source=='archived':
            raise ValueError('New verification was attempted; acceptance must use source=new, including invalid outcomes')
        if self.verification_attempted and source=='new' and self.new_conditions is None:
            invalid={'valid':False,'success':False,'guard_failure':False,'peak_contact_normal_N':None,'closing_deg':0.0}
            gate=acceptance([{'baseline':invalid,'candidate':invalid,'matched_reset':False}],self.policy)
            out={'case':case,'source':'new','decision':'rejected','accepted':False,
                 'reasons':gate['reasons']+['new_verification_incomplete_or_invalid'],
                 'execution_validity':False,'independent_gate':'scripts/agent_workflow.py:acceptance',
                 'raw_gate_result':gate,'conditions':{},'llm_cannot_override':True,
                 'new_episodes_executed':self._new_episode_count()}
            write(self.output/'acceptance.json',out);self.last_acceptance=out
            return out
        conditions,match=self._conditions_for(source)
        policy=self.policy
        assert sha(ROOT/'scripts/agent_workflow.py')==self.gate_hash,'Independent gate source changed'
        assert policy['max_peak_force_ratio']==1.25 and policy['max_peak_force_slack_N']==25.0
        metrics={}
        for m,c in conditions.items():
            metrics[m]={'valid':bool(c['execution_validity'] and c['restoration_validity']),
                'success':bool(c['raw_success'] and not c['hard_stop']),
                'guard_failure':c['hard_stop'],'peak_contact_normal_N':c['peaks']['all'],
                'closing_deg':sum(c['initial_deg'][d]-c['final_deg'][d] for d in c['initial_deg'])}
        decision=acceptance([{'baseline':metrics['frozen'],'candidate':metrics['recontact_mix_300'],
                              'matched_reset':match['passed']}],policy)
        ceiling=1.25*conditions['frozen']['peaks']['all']+25
        qualified={m:bool(metrics[m]['valid'] and match['passed'] and metrics[m]['success'] and
                    conditions[m]['peaks']['all']<=ceiling) for m in MODELS}
        out={'case':case,'source':source,'decision':'accepted' if decision['passed'] else 'rejected',
             'accepted':decision['passed'],'reasons':decision['reasons'],'independent_gate':'scripts/agent_workflow.py:acceptance',
             'gate_source_sha256':sha(ROOT/'scripts/agent_workflow.py'),'policy':policy,
             'raw_gate_result':decision,'v2_qualified_success':qualified,'relative_ceiling_N':ceiling,
             'matched_evidence':match,'conditions':{m:compact_condition(c) for m,c in conditions.items()},
             'llm_cannot_override':True,'new_episodes_executed':self._new_episode_count()}
        write(self.output/'acceptance.json',out);self.last_acceptance=out
        return out

    def _new_episode_count(self):
        ledger=self.physical/'execution.json'
        return len(read(ledger)) if ledger.exists() else 0

    def run_bounded_verification(self, case, rationale):
        if self.verification_attempted or (self.physical/'attempt.json').exists():
            raise ValueError('One fixed verification pair already attempted; retries prohibited')
        started=time.monotonic()
        try:
            result=self._execute_verification(case,rationale)
            result['status']='completed' if result['pair_complete'] else 'technical_invalid'
        except Exception as exc:
            result={'source':'new','status':'technical_invalid','pair_complete':False,
                    'failure':type(exc).__name__, 'failure_phase':getattr(self,'verification_phase','preflight'),
                    'failure_note':'Preflight or independent worker failed; preserved local records contain the attempted stage.',
                    'episodes_attempted':self._new_episode_count(),'new_physical_verification':True}
        result['elapsed_seconds']=time.monotonic()-started
        write(self.physical/'verification-result.json',result)
        return result

    def _execute_verification(self, case, rationale):
        marker=self.physical/'attempt.json'
        if self.verification_attempted or marker.exists():raise ValueError('One fixed verification pair already attempted; retries prohibited')
        self.verification_attempted=True
        self.last_acceptance=None
        write(marker,{'rationale':rationale,'started_unix':time.time(),'source':'new','jobs':self.jobs})
        plan=read(self.physical/'registered-plan.json')
        self.verification_phase='budget_and_source_preflight'
        assert time.time()+1260<plan['deadline_unix'],'Insufficient fixed verification wall-clock budget'
        assert Path(sys.prefix).name=='embodiedops-robocasa'
        assert shutil.disk_usage(ROOT).free>10*1024**3,'Insufficient workspace disk reserve'
        assert all(sha(Path(p))==h for p,h in plan['source_hashes'].items()),'Inherited execution source changed'
        self.verification_phase='checkpoint_and_processor_preflight'
        for job in self.jobs:
            model=Path(job['checkpoint'])
            assert sha(model/'model.safetensors')==job['checkpoint_sha256'],'Checkpoint integrity'
            assert all(sha(model/n)==h for n,h in plan['processors'].items()),'Processor integrity'
        write(self.physical/'preflight.json',{'passed':True,'time':time.time(),'models_and_processors':'hash verified','source_hashes':'verified','same_protocol':True})
        runs=[];new={}
        for job in self.jobs:
            jobpath=self.physical/(job['id']+'-job.json');write(jobpath,job)
            env={k:v for k,v in os.environ.items() if not k.startswith(('NEBIUS_','EMBODIEDOPS_REASONING_'))}
            started=time.monotonic();status='completed'
            self.verification_phase='independent_worker'
            resultpath=Path(job['folder'])/'result.json'
            record={'id':job['id'],'returncode':None,'seconds':None,'status':'running',
                    'started_unix':time.time(),'result_path':str(resultpath)}
            runs.append(record);write(self.physical/'execution.json',runs)
            with (self.physical/(job['id']+'-worker.log')).open('x',encoding='utf-8') as log:
                try:
                    process=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/nemotron_verification_worker.py'),
                        '--root',str(self.physical),'--job',str(jobpath)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                    returncode=process.returncode
                except subprocess.TimeoutExpired:
                    returncode=None;status='worker_timeout_no_retry'
            elapsed=time.monotonic()-started
            resultpath=Path(job['folder'])/'result.json'
            record.update(returncode=returncode,seconds=elapsed,status=status)
            write(self.physical/'execution.json',runs)
            if not resultpath.exists() or returncode!=0 or elapsed>600:break
            r=read(resultpath)
            self.verification_phase='independent_contact_and_boundary_audit'
            from recontact_v2_evidence import audit_contacts
            audit=audit_contacts(r);write(Path(job['folder'])/'independent-audit.json',audit)
            if not r['execution_validity'] or not audit['passed']:break
            c=dict(r,id=job['id'],model=job['model'],seed=64,attempt_folder=job['folder'],
                raw_success=r['raw_all_door_success'],qualified_success=None,video=str(Path(job['folder'])/'rollout.mp4'))
            new[job['model']]=c
        if set(new)==set(MODELS):
            self.new_match=self._match(new);self.new_conditions=new
            ceiling=1.25*new['frozen']['peaks']['all']+25
            for c in new.values():
                c.update(relative_ceiling_N=ceiling,relative_pass=c['peaks']['all']<=ceiling)
                c['qualified_success']=bool(c['execution_validity'] and c['restoration_validity'] and self.new_match['passed'] and c['policy_success'] and c['relative_pass'])
            write(self.physical/'pair-result.json',{'conditions':new,'matched':self.new_match})
        return {'source':'new','new_physical_verification':True,'rationale':rationale,
                'pair_complete':self.new_conditions is not None,'episodes_attempted':len(runs),'workers':runs,
                'comparison':self.compare_candidates(case,'new') if self.new_conditions is not None else None,
                'failure_policy':'Infrastructure failure preserved; no retries or substitution; acceptance cannot approve an incomplete pair.'}
