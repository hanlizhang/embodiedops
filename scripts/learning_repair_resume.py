"""Finish only the two missing diagnostic pairs; no training or final-set access."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agent_workflow import ROOT, read, dump, sha, episode_metrics, acceptance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--camera-evidence", type=Path, required=True)
    args = ap.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    assert out.is_relative_to(ROOT / "outputs/agent-workflow-v1")
    evidence = read(args.camera_evidence / "owner-cleanup/trace.json")
    complete = next(e for e in evidence if e["stage"] == "complete")
    assert complete["saved_first_step_qpos_exact"] and not any(complete["warnings"])
    assert all(e["images"]["rgb"]["max"] > 0 for e in evidence if e["stage"] == "mujoco_read_pixels")
    plan = read(source / "registered-plan.json")
    assert plan["diagnostic_seeds"] == [43, 44, 45, 46]
    selected = read(source / "selected-checkpoints.json")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    old_files = {str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()}
    dump(out / "historical-files-before.json", old_files)
    for name in ("learning_repair_resume.py", "phase1d_rollout.py", "renderer_lifecycle.py", "camera_lifecycle_probe.py"):
        shutil.copyfile(ROOT / "scripts" / name, out / (name + ".txt"))
    invalid = source / "diagnostic-46-episode_balanced"
    assert read(invalid / "summary.json")["termination_reason"] == "invalid_observation_black_camera"
    dump(out / "invalid-attempt-archive.json", {
        "path": str(invalid), "preserved_in_place": True, "excluded_from_robot_success_denominator": True,
        "reason": "All three raw observations black after first action; no inference on black images",
        "files": {str(p.relative_to(invalid)): sha(p) for p in invalid.rglob("*") if p.is_file()}})
    policy = read(ROOT / plan["acceptance_policy_source"])["acceptance"]
    # Latest user instruction requires the original diagnostic acceptance gate
    # before final scenes, superseding the sustained runner's unconditional loop.
    dump(out / "continuation-plan.json", {
        "source": str(source), "max_optimizer_updates": 0, "max_policy_episodes": 2,
        "max_actions_per_episode": 600, "episode_timeout_seconds": 420,
        "max_worker_seconds": 840, "max_wall_seconds": 1200,
        "matrix": [{"seed": 46, "arm": a, "checkpoint": selected[a]} for a in plan["arms"]],
        "camera_correction": "renderer-owner-cleanup; no image substitution or retry",
        "final_gate": "Latest user instruction: require diagnostic acceptance and validation improvement",
        "final_seeds_preserved": plan["final_seeds"], "acceptance_policy": policy})

    def event(kind, **values):
        with (out / "trace.jsonl").open("a") as f:
            f.write(json.dumps(dict(event=kind, elapsed_seconds=time.monotonic()-started, **values)) + "\n")
        print(kind, json.dumps(values)[:400], flush=True)

    worker_seconds = 0
    try:
        for arm in plan["arms"]:
            checkpoint = Path(selected[arm]["checkpoint"])
            assert sha(checkpoint / "model.safetensors") == selected[arm]["sha256"]
            folder = out / f"diagnostic-46-{arm}"
            command = [sys.executable, "-B", "-u", "scripts/phase1d_rollout.py",
                       "--output", str(folder), "--seed", "46", "--checkpoint", str(checkpoint),
                       "--instruction", plan["instruction"], "--workflow-diagnostics",
                       "--renderer-owner-cleanup", "--reference", str(source / "diagnostic-46-pretrained")]
            event("evaluation_start", arm=arm, command=command)
            tick = time.monotonic()
            with (out / (arm + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=min(420, 1200-(time.monotonic()-started)))
            worker_seconds += time.monotonic()-tick
            assert result.returncode in (0, 2)
            metrics = episode_metrics(folder)
            event("evaluation_result", arm=arm, metrics=metrics)
            assert metrics["valid"] or metrics["guard_failure"], "Invalid episode: no automatic retry"
            assert read(folder / "paired-reset-validation.json")["passed"]
            assert read(folder / "manifest.json")["model_sha256"] == selected[arm]["sha256"]
        gates = {}
        all_metrics = {}
        for seed in plan["diagnostic_seeds"]:
            for arm in ["pretrained"] + plan["arms"]:
                directory = (out if seed == 46 and arm != "pretrained" else source) / f"diagnostic-{seed}-{arm}"
                all_metrics[f"{seed}-{arm}"] = episode_metrics(directory)
        for arm in plan["arms"]:
            pairs = [dict(seed=s, baseline=all_metrics[f"{s}-pretrained"],
                          candidate=all_metrics[f"{s}-{arm}"],
                          matched_reset=read((out if s == 46 else source) /
                                            f"diagnostic-{s}-{arm}/paired-reset-validation.json")["passed"])
                     for s in plan["diagnostic_seeds"]]
            gate = acceptance(pairs, policy)
            reasons = gate["reasons"][:]
            if not selected[arm]["validation_improved"]:
                reasons.append("validation_regression_blocks_promotion")
            gates[arm] = dict(diagnostic=gate, accepted=False, qualifies_for_final=not reasons,
                             reasons=reasons)
        # No final-set execution in this bounded continuation. Existing per-seed
        # regressions cannot be erased by the last diagnostic result.
        assert not any(g["qualifies_for_final"] for g in gates.values())
        dump(out / "result.json", dict(
            technically_completed_diagnostic_learning_evaluation=True,
            demonstrated_improvement=False, decisions=gates, metrics=all_metrics,
            final=dict(status="not_run_no_candidate_qualified", seeds=plan["final_seeds"]),
            valid_policy_episodes=12, invalid_excluded_attempts=1, new_policy_episodes=2,
            worker_wall_seconds=worker_seconds, wall_seconds=time.monotonic()-started,
            active_gpu_hours=None, energy_kWh=None))
        event("decision", decisions=gates)
    except Exception as exc:
        dump(out / "blocked.json", dict(error=str(exc)))
        raise
    finally:
        unchanged = all((source / p).exists() and sha(source / p) == digest for p, digest in old_files.items())
        dump(out / "preservation-after.json", dict(all_historical_files_unchanged=unchanged,
                                                  count=len(old_files)))
        assert unchanged


if __name__ == "__main__":
    main()
