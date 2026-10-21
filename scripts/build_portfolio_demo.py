"""Build an artifact-only local presentation. No ML, simulator or network imports."""
import argparse
import hashlib
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
                ROOT / "docs/DEMO.md", ROOT / "docs/demo/index.html"]
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

    # Use the same maintained presentation for local and published viewing.
    # Inject actual local media only after the original evidence checks above pass.
    template = (ROOT / "docs/demo/index.html").read_text(encoding="utf-8")
    sources = {
        "success": link(videos[0][2] / "rollout.mp4"),
        "incomplete": link(videos[1][2] / "rollout.mp4"),
        "candidate": link(videos[2][2] / "rollout.mp4"),
    }
    marker = '<script id="video-sources" type="application/json">'
    if template.count(marker) != 1:
        raise ValueError("Demo template must contain exactly one video-sources block")
    before, after = template.split(marker, 1)
    _, after = after.split("</script>", 1)
    media_json = json.dumps(sources, ensure_ascii=False).replace("<", "\\u003c")
    html = before + marker + media_json + "</script>" + after
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

