"""Build the recorded Nemotron Demo from existing artifacts only.

No model loading, API requests, training, or simulation. Public HTML never links
local seed-64 videos; the local HTML uses verified original video files.
"""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'docs/demo/index.html'
PUBLIC_EVIDENCE = 'https://github.com/hanlizhang/embodiedops/tree/main/docs/evidence/nemotron-agent-demo-v1'
REPORT_URL = 'https://github.com/hanlizhang/embodiedops/blob/main/environment/nemotron-agent-demo-v1.md'
START, END = '<!-- NEMOTRON_AGENT_DEMO_START -->', '<!-- NEMOTRON_AGENT_DEMO_END -->'
STYLE_START, STYLE_END = '/* NEMOTRON_AGENT_STYLE_START */', '/* NEMOTRON_AGENT_STYLE_END */'
STYLES = '''
.agent-banner{background:var(--ink);color:white;border-radius:8px;padding:22px 25px;display:flex;justify-content:space-between;gap:24px;align-items:center;margin-bottom:24px}.agent-banner .eyebrow{color:var(--lime)}.agent-banner h3{margin-top:6px;font-size:24px}.agent-banner p{font-size:13px;color:#d4dfd6;max-width:66ch}.agent-status{font:12px var(--mono);padding:7px 10px;background:#f6e8df;color:#934529;border-radius:4px;white-space:nowrap}.agent-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.agent-kpi{padding:16px 18px;border:1px solid var(--line);border-radius:6px;background:white}.agent-kpi b{display:block;font-size:24px;letter-spacing:-.035em}.agent-kpi span{font:10px var(--mono);color:var(--muted)}.agent-pair{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:22px 0}.agent-pair video{display:block;width:100%;aspect-ratio:8/3;background:#15221b}.agent-pair .player-caption{align-items:flex-start;display:block}.agent-pair .player-caption span{display:block;margin-top:5px}.agent-pair .media-note{padding:20px;background:#edf1e8;min-height:100px;font-size:13px;color:var(--muted)}.agent-trace{list-style:none;margin:24px 0;padding:0}.agent-trace li{position:relative;padding:0 0 20px 35px;border-left:1px solid #b8cabd;margin-left:10px}.agent-trace li:last-child{padding-bottom:0}.agent-trace li:before{content:'';position:absolute;left:-5px;top:7px;width:9px;height:9px;border-radius:50%;background:var(--green)}.agent-trace h3{font:14px var(--mono);overflow-wrap:anywhere}.agent-trace p{font-size:13px;color:var(--muted);margin-top:7px}.agent-trace details{margin-top:11px}.agent-trace summary{padding:11px 15px;font-size:12px}.agent-json{max-height:330px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.6 var(--mono);padding:0 15px 16px;margin:0}.agent-subhead{font-size:20px;margin-top:30px}.agent-response{font-size:14px;white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);padding:0 22px 22px;max-width:100%}.agent-limits{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:25px}.agent-limits ul{padding-left:18px;margin:10px 0 0;font-size:13px;color:var(--muted)}.agent-limits li{margin:5px 0}.agent-top-link{margin:0 0 16px;font-size:13px}.agent-correction{border-left:4px solid var(--warm)!important;background:#fbefe6!important}.agent-public-note{font-size:13px;color:var(--muted);padding:18px;background:#edf1e8;border:1px solid var(--line);border-radius:6px;margin:22px 0}
@media(max-width:700px){.agent-banner{align-items:flex-start;flex-direction:column}.agent-kpis{grid-template-columns:1fr 1fr}.agent-pair,.agent-limits{grid-template-columns:1fr}.agent-kpi b{font-size:21px}.agent-trace li{padding-left:23px}.agent-banner h3{font-size:21px}}
'''


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def escaped(value):
    return html.escape(str(value), quote=True)


def json_html(value):
    return escaped(json.dumps(value, indent=2, ensure_ascii=False))


def number(value, digits=2):
    return 'not recorded' if value is None else f'{float(value):.{digits}f}'


def angle_map(value):
    return ' / '.join(number(v, 3) + '°' for v in value.values())


def archive_case():
    source = ROOT / 'docs/evidence/recontact-self-imitation-v2/result.json'
    record = read(source)
    pairs = {c['model']: c for c in record['conditions']
             if c['kind'] == 'full' and c['seed'] == 64 and c['model'] in ('frozen', 'recontact_mix_300')}
    if set(pairs) != {'frozen', 'recontact_mix_300'} or not all(c['execution_validity'] for c in pairs.values()):
        raise ValueError('Expected the two valid archived seed-64 conditions')
    if not pairs['frozen']['qualified_success'] or pairs['recontact_mix_300']['qualified_success']:
        raise ValueError('Unexpected archived seed-64 outcomes')
    return pairs, source


def replace_block(text, start, end, block, before):
    if start in text:
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError('Ambiguous generated Demo markers')
        a, remaining = text.split(start, 1)
        _, b = remaining.split(end, 1)
        return a + block + b
    if text.count(before) != 1:
        raise ValueError('Missing unique Demo insertion marker')
    return text.replace(before, block + '\n' + before, 1)


def strip_machine_paths(value):
    """Keep evidence values; public inline preview replaces local-only path strings."""
    if isinstance(value, dict):
        return {k: strip_machine_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_machine_paths(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'[A-Za-z]:[\\/][^\n\r\"<>]*', '[local artifact path; see evidence index]', value)
    return value



def new_verification_html(result, local, archived):
    verification = result.get('physical_verification')
    if not verification:
        return ''
    rows = verification.get('workers', [])
    comparison = verification.get('comparison') or {}
    conditions = comparison.get('conditions') or {}
    complete = bool(verification.get('pair_complete'))
    pieces = ['<h3 class="agent-subhead">This run · agent-requested physical verification</h3>',
        f'<p class="small muted">{escaped(verification.get("rationale", ""))}</p>',
        f'<p class="experiment-note">Pair complete: {str(complete).lower()} · {verification.get("episodes_attempted", 0)} episode(s) attempted. Same registered seed, models, processor, noise and action cap; each model runs at most once. These results do not replace the archived comparison above.</p>']
    if complete:
        if set(conditions) != {'frozen', 'recontact_mix_300'}:
            raise ValueError('Complete new verification must contain exactly the fixed pair')
        changed = [name for name, c in conditions.items() if bool(c['raw_success']) != bool(archived[name]['raw_success'])]
        if changed:
            pieces.append('<p class="agent-public-note"><strong>The new task outcomes differ from the archive.</strong> Changed condition(s): '+escaped(', '.join(changed))+'. This run does not reproduce the archived success/failure pair. Matching registered inputs and initial-state tolerances does not establish bitwise trajectory repeatability; both records remain visible.</p>')
        pieces.append('<div class="table-wrap"><table><thead><tr><th>This-run condition</th><th>True / qualified success</th><th>Door opening · start → final (min)</th><th>Target / other fridge / self / all peak</th><th>Worker wall time</th></tr></thead><tbody>')
        for name, c in conditions.items():
            pieces.append(f'<tr><td>{escaped(name)}<br><span class="small muted">{c["actions"]} actions · valid: {str(c["execution_validity"]).lower()}</span></td><td>{str(c["raw_success"]).lower()} / {str(c["qualified_success"]).lower()}<br><span class="small muted">hard stop: {str(c["hard_stop"]).lower()}</span></td><td>{angle_map(c["initial_deg"])} → {angle_map(c["final_deg"])}<br><span class="small muted">minimum {angle_map(c["min_deg"])}</span></td><td>{" / ".join(number(c["peaks"][k],2) for k in ["target_hand_door","other_fridge","self","all"])} N</td><td>{number(c["worker_seconds"],3)} s</td></tr>')
        pieces.append('</tbody></table></div>')
        pieces.append(f'<p class="experiment-note">This-run relative ceiling: {number(conditions["frozen"]["relative_ceiling_N"],3)} N from the matched frozen execution. Hard ceiling remains 1500 N. Independent acceptance below states its evidence source explicitly.</p>')
        if local:
            pieces.append('<details><summary>Open this run’s two original verification videos</summary><div class="case-body"><div class="agent-pair">')
            for name, c in conditions.items():
                video = Path(c['video'])
                if not video.is_file():
                    raise FileNotFoundError(f'New verification recording missing: {video}')
                with video.open('rb') as stream:
                    if stream.read(12)[4:8] != b'ftyp':
                        raise ValueError('Expected an original verification MP4 container')
                pieces.append(f'<article class="player"><video controls playsinline preload="metadata" src="{escaped(video.as_uri())}" aria-label="This run {escaped(name)}"></video><div class="player-caption"><b>{escaped(name)} · this run</b><span>seed 64 · task success {str(c["raw_success"]).lower()} · {c["actions"]} actions</span><a href="{escaped(video.as_uri())}" target="_blank" rel="noopener">Open original video ↗</a></div></article>')
            pieces.append('</div></div></details>')
        else:
            pieces.append('<p class="experiment-note">The new verification videos are also connected only in the local Demo; they have no verified public media URL.</p>')
    else:
        pieces.append('<p class="experiment-note"><strong>The new pair did not complete validly.</strong> Worker failures are retained below; they do not count as accepted repair evidence.</p>')
    worker_summary = [{k: w.get(k) for k in ('id','status','returncode','seconds')} for w in rows]
    pieces.append(f'<details><summary>Actual independent worker attempts and latency</summary><pre class="agent-json">{json_html(worker_summary)}</pre></details>')
    return '\n'.join(pieces)


def build_section(result, trace, acceptance, pairs, local, narrative_review):
    if not isinstance(trace.get('events'), list) or not isinstance(trace.get('messages'), list):
        raise ValueError('Expected real structured API messages and tool events')
    sequence = [e['tool'] for e in trace['events']]
    if sequence != result.get('tool_sequence'):
        raise ValueError('Displayed tool sequence differs from the recorded run')
    requests = trace.get('requests', [])
    if result['api_requests'] != len(requests) or result['api_requests'] > 10:
        raise ValueError('API request count differs or exceeds budget')
    calls = sum(len(m.get('tool_calls') or []) for m in trace['messages'] if m.get('role') == 'assistant')
    if calls != len(sequence):
        raise ValueError('Tool events differ from actual assistant calls')
    model = result['model']
    if model != 'nvidia/Nemotron-3_5-Lightning':
        raise ValueError('Unexpected agent model')
    complete = result['status'] == 'completed'
    label = 'Recorded Nemotron loop' if complete else 'Incomplete recorded run'
    episodes = result['new_episodes']
    episode_note = ('The agent used archived verification; no new physical episodes were run.' if episodes == 0 else
                    f'The agent requested {episodes} new physical episode(s). Archived and current results remain separate in the trace.')
    cost = result.get('estimated_cost_usd')
    cost_label = 'unknown' if cost is None else f'${cost:.6f}'
    usage = result.get('api_usage', {})
    pieces = [START, '<section class="section" id="agentic">',
        '<div class="section-head"><div><p class="eyebrow">LIVE MODEL · RECORDED EVIDENCE</p><h2>Agentic Diagnosis<br>&amp; Verification.</h2></div><p>NVIDIA Nemotron chooses bounded tools, receives their real results, and investigates a SmolVLA regression. An independent gate owns the final candidate decision.</p></div>',
        f'<div class="agent-banner"><div><p class="eyebrow">{escaped(label)} · SEED 64</p><h3>A regression, investigated with tools.</h3><p>In the archived case, frozen closes the fridge and the 300-update recontact mixture leaves it open. Nemotron inspects that evidence and decides whether the candidate needs further verification.</p></div><span class="agent-status">{escaped(acceptance.get("decision", "see gate decision"))}</span></div>',
        f'<div class="agent-kpis"><div class="agent-kpi"><b>{result["api_requests"]}</b><span>REAL NEBIUS REQUESTS / MAX 10</span></div><div class="agent-kpi"><b>{len(sequence)}</b><span>ACTUAL TOOL CALLS</span></div><div class="agent-kpi"><b>{cost_label}</b><span>USAGE-BASED ESTIMATE · USD</span></div><div class="agent-kpi"><b>{episodes}</b><span>NEW PHYSICAL EPISODES / MAX 2</span></div></div>',
        f'<p class="small muted">Model: <code>{escaped(model)}</code> · {escaped(episode_note)} Tool order below comes from the API trace, not a scripted route.</p>']
    if local:
        pieces.append('<div class="agent-pair">')
        for name, title in [('frozen', 'Frozen reference · task completed'), ('recontact_mix_300', 'Recontact mixture · rejected candidate')]:
            c = pairs[name]
            p = Path(c['video'])
            if not p.is_file():
                raise FileNotFoundError(f'Original seed-64 video missing: {p}')
            with p.open('rb') as stream:
                if stream.read(12)[4:8] != b'ftyp':
                    raise ValueError('Expected original MP4 container')
            pieces.append(f'<article class="player"><video controls playsinline preload="metadata" src="{escaped(p.as_uri())}" aria-label="{escaped(title)}"></video><div class="player-caption"><b>{escaped(title)}</b><span>Archived v2 · seed 64 · {c["actions"]} actions · {angle_map(c["final_deg"])}</span><a href="{escaped(p.as_uri())}" target="_blank" rel="noopener">Open original video ↗</a></div></article>')
        pieces.append('</div>')
    else:
        pieces.append('<div class="agent-public-note"><strong>Seed 64 video availability.</strong> The exact frozen/candidate pair is connected in the local Demo on the evidence machine. These recordings are not publicly hosted, so this page does not substitute another scene or link a local drive. The <a href="#execution">public robot recordings below</a> are explicitly labelled historical seed 44/43 examples.</div>')
    pieces.append('<div class="table-wrap"><table><thead><tr><th>Archived seed-64 condition</th><th>True task success</th><th>Door opening · start → final (min)</th><th>Target hand–door / other fridge / self / all peak</th><th>Hard stop / relative gate</th></tr></thead><tbody>')
    for name, c in pairs.items():
        peaks = c['peaks']
        pieces.append(f'<tr><td>{escaped(name)}<br><span class="small muted">{c["actions"]} actions · valid</span></td><td>{str(c["raw_success"]).lower()}<br><span class="small muted">qualified: {str(c["qualified_success"]).lower()}</span></td><td>{angle_map(c["initial_deg"])} → {angle_map(c["final_deg"])}<br><span class="small muted">minimum {angle_map(c["min_deg"])}</span></td><td>{" / ".join(number(peaks[k], 2) for k in ["target_hand_door", "other_fridge", "self", "all"])} N</td><td>{str(c["hard_stop"]).lower()} / {"pass" if c["relative_pass"] else "fail"}</td></tr>')
    pieces.append('</tbody></table></div>')
    pieces.append(f'<p class="experiment-note">Same registered scene, checkpoint identities, processors and matched inference noise. H=50, 10 flow steps, 20 Hz, 600-action cap. Relative all-robot contact ceiling: {number(pairs["frozen"]["relative_ceiling_N"],3)} N; hard ceiling: 1500 N. Contact values come from geometry-identified simulator records.</p>')
    pieces.append(f'<p class="experiment-note">Archived physical worker wall time: frozen {number(pairs["frozen"]["worker_seconds"],3)} s; mixture {number(pairs["recontact_mix_300"]["worker_seconds"],3)} s. These are historical execution times, separate from this agent run. New verification timing, if requested, appears in its returned tool result below.</p>')
    pieces.append(new_verification_html(result, local, pairs))
    pieces.append('<h3 class="agent-subhead">The actual tool-calling path</h3><ol class="agent-trace">')
    for index, event in enumerate(trace['events'], 1):
        tool_result = event['result'] if local else strip_machine_paths(event['result'])
        args = event.get('arguments', {})
        rationale = args.get('rationale') or args.get('reason') or ''
        returned = event.get('returned_to_model_request')
        delivery = f'Returned to model request {returned}' if returned is not None else 'Model delivery not recorded'
        pieces.append(f'<li><h3>{index:02d} · {escaped(event["tool"])}</h3><p>Decision step {escaped(event.get("step", "?"))} · {number(event.get("seconds"),3)} s tool latency · {escaped(delivery)}.</p>')
        if rationale:
            pieces.append(f'<p>{escaped(rationale)}</p>')
        pieces.append(f'<details><summary>Arguments and returned evidence</summary><pre class="agent-json">{json_html({"arguments": args, "result": tool_result})}</pre></details></li>')
    pieces.append('</ol>')
    gate = acceptance if local else strip_machine_paths(acceptance)
    pieces.append(f'<details open><summary>Final independent acceptance decision · evidence source included</summary><pre class="agent-json">{json_html(gate)}</pre></details>')
    if narrative_review.get('model_narrative_fully_accurate') is False:
        corrections = ''.join('<li>'+escaped(issue['correction'])+'</li>' for issue in narrative_review['issues'])
        pieces.append('<div class="agent-public-note agent-correction" role="note"><strong>Factual correction to the model’s final narrative.</strong><p>The tool-calling protocol completed, but the raw model explanation contains errors. The measured results and independent gate take precedence:</p><ul>'+corrections+'</ul><p>The unedited response remains below. These errors did not override the rejection.</p></div>')
    pieces.append(f'<details><summary>Raw Nemotron final response · contains the corrected errors above</summary><p class="agent-response">{escaped(result.get("final_answer", "No final response recorded."))}</p></details>')
    pieces.append(f'<p class="fineprint">Actual API usage: {usage.get("prompt_tokens", "unknown")} input + {usage.get("completion_tokens", "unknown")} output tokens; reasoning tokens: {usage.get("reasoning_tokens", "unknown")} (included in output). API latency total: {number(result.get("api_latency_seconds"),3)} s. Whole agent run: {number(result.get("elapsed_seconds"),3)} s. Cost is an estimate from returned usage and the recorded Nebius price, not an invoice or a provider billing cap.</p>')
    pieces.append('<div class="agent-limits"><div class="mini"><h3>Verified in this recorded run</h3><ul><li>Real NVIDIA Nemotron Chat Completions with native tool calls and matching tool results.</li><li>Bounded access to existing robot evidence; independent candidate acceptance.</li><li>Tool choices, result delivery, usage and latency preserved in the trace.</li></ul></div><div class="mini"><h3>Limits of the demonstration</h3><ul><li>One preselected development regression; no evidence of reliable autonomous diagnosis across tasks.</li><li>No new VLA training, learned recovery gain or successful policy repair.</li><li>No hardware deployment; contact limits are simulator checks.</li></ul></div></div>')
    if not complete:
        pieces.append('<p class="fineprint"><strong>This run did not complete. The recorded partial tool trace does not establish a successful closed loop.</strong></p>')
    pieces.append(f'<div class="sources"><a href="{PUBLIC_EVIDENCE}" target="_blank" rel="noopener">Complete tool trace &amp; source evidence ↗</a><a href="{REPORT_URL}" target="_blank" rel="noopener">中文执行报告 ↗</a></div></section>{END}')
    return '\n'.join(pieces)


def install_section(template, section):
    template = replace_block(template, STYLE_START, STYLE_END, STYLE_START+'\n'+STYLES+'\n'+STYLE_END, '</style>')
    template = replace_block(template, START, END, section, '<section class="section" id="execution">')
    template = template.replace('href="#execution">Explore the demo', 'href="#agentic">Explore the demo')
    template = template.replace('href="#execution">Demo', 'href="#agentic">Agent Demo')
    old = 'The executed decision backend is deterministic. The model-driven extension is being developed with NVIDIA Nemotron; this demo presents the verified tool and robot-learning foundation.'
    new = 'Historical experiment workflows used deterministic rules. The recorded Nemotron investigation above adds a real native tool-calling loop around a fixed regression case; candidate acceptance remains independent.'
    template = template.replace(old, new)
    old = '<span class="eyebrow">IN DEVELOPMENT</span><h3>NVIDIA Nemotron</h3></div><p>Connecting a reasoning model to the existing diagnostic tools for hypothesis updates and probe selection. An endpoint adapter exists; live model-driven decisions are not yet part of the verified demo. Completed workflow executions use evidence-based rules.</p>'
    new = '<span class="eyebrow">RECORDED AGENT INVESTIGATION</span><h3>NVIDIA Nemotron</h3></div><p>The seed-64 investigation above records native tool calls, returned physical evidence and an independent acceptance decision. Earlier learning studies remain human-designed and rule-orchestrated. This bounded integration does not establish autonomous learning-based repair.</p>'
    template = template.replace(old, new)
    return template


def build(run):
    run = run.resolve()
    if not run.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Expected an existing local agent run under outputs')
    inputs = [run/'result.json', run/'trace.json', run/'acceptance.json']
    result, trace, acceptance = [read(p) for p in inputs]
    review_path = run/'narrative-review.json'
    narrative_review = read(review_path)
    if narrative_review.get('trace_sha256') != digest(run/'trace.json'):
        raise ValueError('Narrative review does not bind to this actual trace')
    reviewed_gate = narrative_review['authoritative_decision']
    if any(reviewed_gate[k] != acceptance[k] for k in ('accepted','source','reasons')):
        raise ValueError('Narrative review differs from the independent gate')
    inputs.append(review_path)
    if result.get('independent_acceptance') != acceptance:
        raise ValueError('Result and independent acceptance artifact disagree')
    pairs, source = archive_case()
    template = TEMPLATE.read_text(encoding='utf-8')
    public = install_section(template, build_section(result, trace, acceptance, pairs, False, narrative_review))
    local = install_section(template, build_section(result, trace, acceptance, pairs, True, narrative_review))
    # Existing public recordings stay public on Pages. Connect their authentic local
    # counterparts in the standalone local presentation without copying videos.
    local_sources = {
        'success': ROOT/'outputs/agent-workflow-v1/learning-5000-v1/diagnostic-44-episode_balanced/rollout.mp4',
        'incomplete': ROOT/'outputs/agent-workflow-v1/learning-5000-v1/diagnostic-43-episode_balanced/rollout.mp4',
        'candidate': ROOT/'outputs/agent-workflow-v1/terminal-window-v1/diagnostic-43/rollout.mp4'}
    if all(p.is_file() for p in local_sources.values()):
        marker = '<script id="video-sources" type="application/json">'
        before, rest = local.split(marker, 1)
        _, after = rest.split('</script>', 1)
        local = before+marker+json.dumps({k:p.as_uri() for k,p in local_sources.items()})+'</script>'+after
    if re.search(r'(?:src|href)=[\"\'](?:file:|[A-Za-z]:[\\/])', public, re.I):
        raise ValueError('Public page contains a local file media/link URL')
    local_path = run/'index.html'
    if local_path.exists():
        raise FileExistsError('Existing local Demo is preserved; do not overwrite it')
    local_path.write_text(local, encoding='utf-8')
    TEMPLATE.write_text(public, encoding='utf-8')
    new_conditions = ((result.get('physical_verification') or {}).get('comparison') or {}).get('conditions') or {}
    all_video_records = {('archived/' + k): v for k, v in pairs.items()}
    all_video_records.update({('this_run/' + k): v for k, v in new_conditions.items()})
    provenance = {'purpose':'Recorded artifact presentation; no new API, ML or physics calls',
        'inputs':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in inputs+[source]},
        'videos':{k:{'path':v['video'],'bytes':Path(v['video']).stat().st_size} for k,v in all_video_records.items()},
        'builder_sha256':digest(Path(__file__)), 'public_html_sha256':digest(TEMPLATE),
        'local_html_sha256':digest(local_path)}
    (run/'demo-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    return local_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/nemotron-agent-demo-v1')
    print(build(parser.parse_args().run))
