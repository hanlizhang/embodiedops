"""Build an artifact-only local presentation. No ML, simulator or network imports."""
import argparse
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/agent-workflow-v1/terminal-window-v1"
NATIVE = ROOT / "outputs/agent-workflow-v1/learning-5000-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(out):
    out = out.resolve()
    if not out.is_relative_to(ROOT / "outputs") or out.exists():
        raise ValueError("Choose a fresh directory under outputs; existing files are never replaced.")
    summary_path = ROOT / "environment/terminal-window-experiment-v1-results.json"
    videos = [
        ("success", "A real success · Episode-balanced / seed 44",
         NATIVE / "diagnostic-44-episode_balanced", True,
         "Task success at action 220. Final opening 0.31°. This selected example is not a success-rate estimate."),
        ("failure", "Incomplete closure · Episode-balanced / seed 43",
         NATIVE / "diagnostic-43-episode_balanced", False,
         "600 actions; 13.11° remains open. Loaded door contact ends after action 251; closure settles near 285."),
        ("candidate", "Rejected repair · Terminal-window / seed 43",
         RUN / "diagnostic-43", False,
         "600 actions; 90.00° remains open. Peak recorded normal force 575.12 N, including robot self-contact."),
    ]
    plots = [RUN / "analysis/loss-comparison.svg", RUN / "analysis/door-progress.svg"]
    required = [summary_path, RUN / "result.json", RUN / "candidate/selected-checkpoint.json",
                ROOT / "docs/DEMO.md", ROOT / "docs/INTERVIEW.md"]
    required += plots
    required += [folder / name for _, _, folder, _, _ in videos
                 for name in ("rollout.mp4", "summary.json", "manifest.json")]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing authentic artifacts; no substitutes generated:\n" + "\n".join(missing))
    evidence, raw = read(summary_path), read(RUN / "result.json")
    if sha(RUN / "result.json") != evidence["source_sha256"]["result.json"]:
        raise ValueError("Original result hash differs from the reviewed numerical record")
    if raw["accepted"] or raw["final_seeds_evaluated"] or raw["simulator_episodes"] != 4:
        raise ValueError("Unexpected decision/evaluation contract")
    if read(RUN / "candidate/selected-checkpoint.json") != evidence["selected"]:
        raise ValueError("Checkpoint selection differs from reviewed evidence")
    pairs = evidence["diagnostic_pairs"]
    if [p["seed"] for p in pairs] != [43, 44, 45, 46]:
        raise ValueError("Diagnostic matrix differs")
    for pair in pairs:
        for side in ("baseline", "candidate"):
            metric = pair[side]
            summary = read(ROOT / metric["path"] / "summary.json")
            if summary["final_fridge"]["success"] != metric["success"]:
                raise ValueError("Recorded task success differs")
            angles = [math.degrees(x) for x in summary["final_fridge"]["qpos"].values()]
            if len(angles) != len(metric["final_angles_deg"]) or any(
                    not math.isclose(x, y, abs_tol=1e-8) for x, y in zip(angles, metric["final_angles_deg"])):
                raise ValueError("Recorded door angle differs")
            required.append(ROOT / metric["path"] / "summary.json")
    for _, _, folder, success, _ in videos:
        if read(folder / "summary.json")["final_fridge"]["success"] != success:
            raise ValueError("Selected video outcome differs")
        if read(folder / "manifest.json")["resolved_instruction"] != "CloseFridge":
            raise ValueError("Selected video has a different instruction")
        with (folder / "rollout.mp4").open("rb") as stream:
            if stream.read(12)[4:8] != b"ftyp":
                raise ValueError("Expected original MP4 container")
    out.mkdir(parents=True, exist_ok=False)

    def link(path):
        return quote(os.path.relpath(path, out).replace("\\", "/"), safe="/:")

    def video(item):
        ident, title, folder, success, caption = item
        return f"""<article id="{ident}" class="card">
<p class="eyebrow">{'RECORDED SUCCESS' if success else 'RECORDED FAILURE'}</p>
<h3>{escape(title)}</h3>
<video controls preload="metadata" playsinline src="{link(folder/'rollout.mp4')}">
Your browser cannot play this recording. Use the original video link below.</video>
<p>{escape(caption)}</p><a href="{link(folder/'rollout.mp4')}">Original MP4</a>
<span class="muted"> · <a href="{link(folder/'summary.json')}">Original metrics</a></span></article>"""

    table = []
    for p in pairs:
        b, c = p["baseline"], p["candidate"]
        angles = lambda m: " / ".join(f"{x:.2f}°" for x in m["final_angles_deg"])
        table.append(f"<tr><td>{p['seed']}</td><td>{'success' if b['success'] else 'fail'} → "
                     f"{'success' if c['success'] else 'fail'}</td><td>{angles(b)} → {angles(c)}</td>"
                     f"<td>{b['peak_contact_normal_N']:.1f} → {c['peak_contact_normal_N']:.1f} N</td></tr>")
    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EmbodiedOps · A repair that did not pass</title>
<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#172737;background:#f2f5f8}
*{box-sizing:border-box}body{margin:0}header{background:#102b3a;color:white;padding:48px max(5vw,24px)}
main{max-width:1200px;margin:auto;padding:10px 24px 60px}h1{font-size:clamp(30px,4vw,52px);margin:12px 0}
h2{font-size:27px;margin-top:0}h3{font-size:19px}p,li{line-height:1.6}
.eyebrow{font-size:12px;letter-spacing:.14em;font-weight:700;color:#38657a}
header .eyebrow{color:#9fdae5}header p{max-width:840px}nav a{color:#bce7ed;margin-right:25px}
a{color:#13637c}section{margin-top:40px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.card{background:white;border:1px solid #d9e2e8;border-radius:10px;padding:22px;min-width:0}
video{width:100%;background:#111;display:block}.muted{color:#62717b;font-size:13px}
.badge{display:inline-block;color:#8b2b18;background:#ffe6dc;padding:7px 12px;border-radius:6px;font-weight:700}
.stats{display:flex;gap:36px;flex-wrap:wrap;margin:22px 0}.stats b{display:block;font-size:27px}
figure{margin:20px 0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#566771}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px 8px;text-align:left;border-bottom:1px solid #dde4e9}
.overflow{overflow:auto}.callout{border-left:4px solid #b34b2f;padding:10px 20px;background:#fff3ed}
@media(max-width:760px){.grid{grid-template-columns:1fr}header{padding:30px 24px}}
@media print{header{background:white;color:black}video{max-height:180px}section{break-inside:avoid}}
</style><header><p class="eyebrow">EMBODIEDOPS / VERIFIED SIMULATION CASE STUDY</p>
<h1>A repair that did not pass.</h1>
<p>Real SmolVLA training. Real RoboCasa execution. A lower terminal imitation loss
that failed to improve learned manipulation.</p>
<nav><a href="#execution">01 Execution</a><a href="#learning">02 Learning</a>
<a href="#verification">03 Verification</a><a href="#decision">04 Decision</a></nav></header>
<main><section id="execution"><h2>01 · Success is visible. Failure must be measured.</h2>
<p>Original recordings from the same Episode-balanced checkpoint. Three-camera views,
task-ID <code>CloseFridge</code>, H=50, 10 flow steps. These are selected illustrations;
the full diagnostic matrix appears below. Playback follows simulation time, not wall-clock inference time.</p>
<div class="grid">""" + video(videos[0]) + video(videos[1]) + """</div></section>
<section id="learning"><h2>02 · Turn the failure into one controlled learning test.</h2>
<div class="card"><p><b>Evidence → hypothesis → training → independent verification.</b>
Human analysis found incomplete closure in seeds 43/45. The hypothesis was that
emphasizing recorded successful terminal actions would help finish closing.</p>
<ul><li>513 demonstrations audited; 461 train / 52 held-out episodes.</li>
<li>2,438 anchor replacements with full chunks spanning first recorded positive reward.
These are not verified contact events or recovery demonstrations.</li>
<li>One 5,000-update run from original pretrained weights, matching native precision,
Flow Matching, schedule, noise seeds and saved processors.</li>
<li>Step 5,000 selected by original global held-out loss before robot evaluation.
Five independent content reload checks passed; one exceeded its runtime cap.</li></ul>
<p class="muted">The implemented tool loop uses deterministic evidence rules.
This targeted experiment was human-directed and executed by a registered runner;
it was not discovered by an LLM.</p></div>
<div class="stats"><div><b>−3.51%</b>terminal held-out loss</div>
<div><b>+2.72%</b>global held-out loss</div><div><b>109 min</b>training worker time</div>
<div><b>1.73 GiB</b>peak allocated CUDA memory</div></div>
<figure><img src=\"""" + link(plots[0]) + """\" alt="Actual fixed training, global validation and terminal validation losses">
<figcaption>Measured fixed-noise probes. Native terminal reference exists only at selected step 5,000;
no native terminal curve is invented. Terminal-window training increased valid action-target coverage as well as terminal exposure.</figcaption></figure></section>
<section id="verification"><h2>03 · Execute the selected weights on the registered scenes.</h2>
<p>Actual success stayed <b>1/4 → 1/4</b>. All four new episodes were valid. Final seeds 120–123
were gated and never executed. The archived native seed 46 reference is its valid replacement,
not the excluded camera-blackout attempt.</p>
<div class="card overflow"><table><thead><tr><th>Seed</th><th>Native → targeted</th>
<th>Final door openings</th><th>Peak normal contact force</th></tr></thead><tbody>""" + "".join(table) + """</tbody></table></div>
<p class="muted">Positive active solver contact-normal components, including self-contact.
These values are not net hand forces, hinge torques or hardware safety certification.</p>
""" + video(videos[2]) + """
<figure><img src=\"""" + link(plots[1]) + """\" alt="Paired per-door opening traces on all four diagnostic scenes">
<figcaption>Saved joint traces from all diagnostic scenes, including the unsuccessful outcomes.
No policy or physics rerun was used to build this page.</figcaption></figure></section>
<section id="decision"><h2>04 · Reject the candidate. Preserve the evidence.</h2>
<span class="badge">REJECTED</span>
<p>The preregistered criteria required at least 2/4 successes, a repair of seed 43 or45,
retention of 44 success, acceptable per-door progress, contact regressions and offline losses.
Lower terminal loss alone did not satisfy them.</p>
<div class="callout"><b>What this establishes:</b> a completed train-and-evaluate workflow,
not an improved robot policy. Successful-state imitation may not cover policy-generated
failure states; this remains a hypothesis, not a proven causal explanation.</div>
<p>Motion-mixture also failed (0/4). FP32 masters fixed measured BF16 update accumulation
but worsened optimization; full reload timed out and no robot evaluation was performed.</p>
<p class="muted">Protocol deviation: terminal reload 4000 took 23.6 min against a 15 min cap.
Content passed, but the runtime bound was not enforced. The original trace is preserved;
a post-run deadline guard does not retroactively repair this deviation.</p>
<p class="muted">Four development scenes; one training seed per arm; pretraining overlap unknown.
No official benchmark reproduction, autonomous LLM, world model, hardware deployment or
proven success-rate improvement. Active GPU-hours and energy were not measured.</p>
<p><a href="provenance.json">Media / evidence SHA256 provenance</a> ·
<a href=\"""" + link(summary_path) + """\">Compact numerical evidence</a> ·
<a href=\"""" + link(ROOT / "docs/DEMO.md") + """\">Speaker script</a> ·
<a href=\"""" + link(ROOT / "docs/INTERVIEW.md") + """\">English / 中文 interview notes</a></p>
</section></main></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    provenance = {
        "scope": "Existing authentic artifacts only; no inference, training, physics or network",
        "selection": "Illustrative success44 / incomplete43 / rejected43; full diagnostic matrix included",
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"):
                  {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(set(required))},
        "builder_sha256": sha(Path(__file__)),
        "html_sha256": sha(out / "index.html"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return out / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(build(parser.parse_args().output))
