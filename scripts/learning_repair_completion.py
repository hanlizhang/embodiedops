"""Read-only evidence consolidation after the bounded camera-corrected continuation."""
import argparse
import csv
import json
import shutil
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
from agent_workflow import ROOT, read, dump, sha, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--continuation", type=Path, required=True)
    args = ap.parse_args()
    old, out = args.source.resolve(), args.continuation.resolve()
    result = read(out / "result.json")
    audit = read(old / "audited-summary.json")
    assert audit["passed"] and read(out / "preservation-after.json")["all_historical_files_unchanged"]
    selected = read(old / "selected-checkpoints.json")
    camera_checks = {}
    summaries = {}
    for key, metrics in result["metrics"].items():
        folder = ROOT / metrics["path"]
        manifest = read(folder / "manifest.json")
        seed, arm = key.split("-", 1)
        assert manifest["seed"] == int(seed) and manifest["resolved_instruction"] == "CloseFridge"
        assert manifest["execution_horizon"] == 50
        expected = ("96ae6628166a7dc1b41f8a2d92176c48e394988294b0b44755409acbb3f9bc2b"
                    if arm == "pretrained" else selected[arm]["sha256"])
        assert manifest["model_sha256"] == expected
        if int(seed) == 46 and arm != "pretrained":
            reference_manifest = read(old / f"diagnostic-45-{arm}/manifest.json")
            assert manifest["checkpoint_file_hashes"] == reference_manifest["checkpoint_file_hashes"]
            assert manifest["renderer_owner_cleanup"]["sha256"] == sha(out / "renderer_lifecycle.py.txt")
        script = folder / "executed-script.py.txt"
        assert manifest["script_sha256"] == sha(script)
        checks = {}
        for p in folder.glob("input-*.json"):
            assert all(c["max"] > 0 for c in read(p)["cameras"].values())
        for p in folder.glob("final-robot0_*.png"):
            image = np.asarray(Image.open(p))
            checks[p.name] = dict(shape=list(image.shape), dtype=str(image.dtype),
                                 min=int(image.min()), max=int(image.max()),
                                 nonzero_pixels=int(np.count_nonzero(np.any(image != 0, axis=-1))))
            assert image.max() > 0
        assert len(checks) == 3
        camera_checks[key] = dict(final=checks, input_records=len(list(folder.glob("input-*.json"))),
                                 validity="runner guards every action input; saved replan inputs and final PNGs checked")
        if arm != "pretrained":
            assert read(folder / "paired-reset-validation.json")["passed"]
        summaries[key] = read(folder / "summary.json")
    aggregates = {}
    for arm in ["pretrained", "episode_balanced", "motion_mixture"]:
        episodes = [m for k, m in result["metrics"].items() if k.endswith("-" + arm)]
        latencies = [v for k, s in summaries.items() if k.endswith("-" + arm) for v in s["inference_seconds"]]
        aggregates[arm] = dict(successes=sum(m["success"] for m in episodes), episodes=len(episodes),
                               mean_closing_deg=mean(m["closing_deg"] for m in episodes),
                               peak_normal_N=max(m["peak_contact_normal_N"] for m in episodes),
                               mean_inference_seconds=mean(latencies), inference_calls=len(latencies),
                               total_rollout_seconds=sum(m["rollout_seconds"] for m in episodes),
                               peak_cuda_allocated_bytes=max(m["peak_cuda_allocated"] for m in episodes),
                               guard_failures=sum(m["guard_failure"] for m in episodes))
    combined = dict(training=audit["training"], checkpoint_selection=selected,
                    robot_aggregates=aggregates, evaluation=result, camera_checks=camera_checks,
                    original_audit_sha256=sha(old / "audited-summary.json"),
                    prior_worker_wall_hours=audit["gpu_worker_process_hours"],
                    continuation_worker_wall_hours=result["worker_wall_seconds"]/3600,
                    total_worker_wall_hours=audit["gpu_worker_process_hours"]+result["worker_wall_seconds"]/3600,
                    cost_limitations="Worker wall time includes CPU/I/O, not active GPU time. Camera probes excluded. No energy telemetry.",
                    active_gpu_hours=None, energy_kWh=None)
    old_first = read(old / "diagnostic-46-episode_balanced/first-action-trace.json")
    new_first = read(out / "diagnostic-46-episode_balanced/first-action-trace.json")
    combined["replacement_first_action_exact"] = old_first["raw_action"] == new_first["raw_action"]
    combined["replacement_first_step_qpos_exact"] = old_first["after"]["qpos"] == new_first["after"]["qpos"]
    assert combined["replacement_first_action_exact"] and combined["replacement_first_step_qpos_exact"]
    dump(out / "completion-audit.json", combined)
    shutil.copyfile(old / "loss-comparison.png", out / "loss-comparison.png")
    fields = ["seed_arm", "success", "actions", "closing_deg", "best_closing_deg", "final_angles_deg",
              "peak_contact_normal_N", "loaded_actions", "eef_path_m", "mean_inference_seconds", "path"]
    with (out / "paired-results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key, metrics in result["metrics"].items():
            writer.writerow(dict(seed_arm=key, **{k: metrics[k] for k in fields[1:]}))
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
