"""Execute one preregistered terminal-window learning repair with existing tools."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from agent_workflow import ROOT, BASE, read, rows, dump, sha, episode_metrics, acceptance


def validate_worker_completion(name, returncode, elapsed, limit, allowed=(0,)):
    """A successful exit must not silently authorize work after an elapsed deadline.

    Post-run correction: terminal-window-v1's fourth reload returned successfully
    after its subprocess timeout allowance. Its original trace/source are retained.
    This check stops subsequent workers even if an OS wait returns late.
    """
    if elapsed > limit:
        raise TimeoutError(f"{name} completed after its {limit:.3f}s wall limit ({elapsed:.3f}s); no continuation")
    if returncode not in allowed:
        raise RuntimeError(f"{name} failed with code {returncode}; no retry")


def per_door_gate(pairs, tolerance):
    reasons = []
    for pair in pairs:
        before, after = pair["baseline"]["final_angles_deg"], pair["candidate"]["final_angles_deg"]
        if len(before) != len(after):
            reasons.append("door_identity_mismatch")
        elif any(c > b+tolerance for b, c in zip(before, after)):
            reasons.append("individual_door_regression")
    return sorted(set(reasons))


def pilot_decision(native_pairs, pretrained_pairs, selected_global, terminal, native_terminal, plan, policy):
    native_gate = acceptance(native_pairs, policy)
    pretrained_gate = acceptance(pretrained_pairs, policy)
    reasons = ["native:"+r for r in native_gate["reasons"]]
    reasons += ["pretrained:"+r for r in pretrained_gate["reasons"]]
    reasons += per_door_gate(native_pairs, plan["max_per_door_regression_deg"])
    success = {p["seed"]: bool(p["candidate"]["success"]) for p in native_pairs}
    assert set(success) == set(plan["diagnostic_seeds"])
    if sum(success.values()) < plan["minimum_diagnostic_successes"]:
        reasons.append("insufficient_diagnostic_success")
    if not any(success[s] for s in plan["required_any_success_seeds"]):
        reasons.append("incomplete_closing_cases_not_repaired")
    if not all(success[s] for s in plan["required_retained_success_seeds"]):
        reasons.append("required_success_lost")
    if selected_global > plan["native_global_validation"]:
        reasons.append("global_validation_regression")
    if not terminal < native_terminal:
        reasons.append("terminal_validation_not_improved")
    return dict(passed=not reasons, reasons=sorted(set(reasons)),
                native_comparison=native_gate, pretrained_comparison=pretrained_gate,
                selected_global_validation=selected_global, selected_terminal_validation=terminal,
                native_terminal_validation=native_terminal)


def compatible(reference, candidate, checkpoint_sha):
    """Require stored contract/reset evidence; never silently reuse invalid episodes."""
    a, b = read(reference/"manifest.json"), read(candidate/"manifest.json")
    for key in ("versions", "simulator_sources", "resolved_instruction", "execution_horizon", "rng_rule"):
        assert a[key] == b[key], key
    assert b["resolved_instruction"] == "CloseFridge" and b["execution_horizon"] == 50
    for key, value in a["checkpoint_file_hashes"].items():
        if key != "config.json":
            assert b["checkpoint_file_hashes"][key] == value
    assert b["model_sha256"] == checkpoint_sha
    ar, br = read(reference/"reset.json"), read(candidate/"reset.json")
    assert ar["physical"] == br["physical"] and ar["state"] == br["state"]
    assert list(ar["fridge"]["qpos"]) == list(br["fridge"]["qpos"])
    assert not (reference/"invalid-camera.json").exists()
    assert not (candidate/"invalid-camera.json").exists()
    assert read(candidate/"paired-reset-validation.json")["passed"]
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    args = parser.parse_args()
    out, cache = args.output.resolve(), args.dataset_cache.resolve()
    assert out.is_relative_to(ROOT/"outputs/agent-workflow-v1")
    assert cache.is_relative_to(ROOT/"outputs")
    assert not out.exists() and not cache.exists(), "Exclusive run and cache required; no restart"
    plan = read(ROOT/"environment/terminal-window-experiment-v1.json")
    assert plan["optimizer_updates"] == 5000 and plan["max_training_runs"] == 1
    assert plan["diagnostic_seeds"] == [43, 44, 45, 46]
    assert plan["gated_final_seeds"] == [120, 121, 122, 123]
    assert shutil.disk_usage(ROOT).free >= plan["minimum_free_disk_bytes"]
    out.mkdir(parents=True)
    cache.mkdir()
    started = time.monotonic()
    dump(out/"registered-plan.json", plan)
    baseline = ROOT/plan["baseline"]
    history = baseline.parent
    resume = ROOT/"outputs/agent-workflow-v1/learning-5000-resume-v1"
    fp32 = ROOT/"outputs/agent-workflow-v1/optimizer-precision-v1"
    settings = dict(read(baseline/"settings.json"), sampling=plan["sampling"])
    cfg = out/"training-config.json"
    dump(cfg, settings)
    names = ["terminal_window_experiment.py", "terminal_sampling.py", "phase2_train_v1.py",
             "learning_sampling.py", "phase1d_rollout.py", "agent_workflow.py",
             "control_diagnostics.py", "renderer_lifecycle.py"]
    for name in names:
        shutil.copyfile(ROOT/"scripts"/name, out/(name+".txt"))
    shutil.copyfile(ROOT/plan["proposal"], out/"preregistered-proposal.md")
    protected = [BASE/"model.safetensors", ROOT/plan["proposal"]]
    preservation = {str(p.relative_to(ROOT)): sha(p)
                    for folder in (history, resume, fp32, ROOT/"outputs/terminal-window-audit-v1/final")
                    for p in folder.rglob("*") if p.is_file()}
    preservation.update(read(fp32/"preservation-before.json"))
    preservation.update({str(p.relative_to(ROOT)): sha(p) for p in protected})
    dump(out/"preservation-before.json", preservation)
    times, episodes = {}, 0

    def event(kind, **values):
        record = dict(event=kind, elapsed_seconds=time.monotonic()-started, **values)
        with (out/"trace.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
        print(kind, json.dumps(values)[:450], flush=True)

    def run(command, name, timeout, allowed=(0,)):
        remaining = plan["max_wall_seconds"]-(time.monotonic()-started)
        assert remaining > 0, "Registered total wall budget exhausted"
        limit = min(timeout, remaining)
        event("worker_start", name=name, command=command, timeout_seconds=limit)
        tick, code = time.monotonic(), None
        try:
            with (out/(name+".log")).open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
            code = result.returncode
        finally:
            times[name] = time.monotonic()-tick
            event("worker_end", name=name, seconds=times[name], returncode=code)
        validate_worker_completion(name, code, times[name], limit, allowed)

    def train_command(folder):
        return [sys.executable, "-B", "-u", "scripts/phase2_train_v1.py", "--config", str(cfg),
                "--output", str(folder), "--dataset-cache", str(cache)]

    def rollout(seed, name, checkpoint, reference=None):
        nonlocal episodes
        episodes += 1
        assert episodes <= plan["max_policy_episodes"]
        folder = out/name
        command = [sys.executable, "-B", "-u", "scripts/phase1d_rollout.py",
                   "--output", str(folder), "--seed", str(seed), "--checkpoint", str(checkpoint),
                   "--instruction", "CloseFridge", "--workflow-diagnostics", "--renderer-owner-cleanup"]
        if reference:
            command += ["--reference", str(reference)]
        run(command, name, plan["episode_timeout_seconds"], allowed=(0, 2))
        metric = episode_metrics(folder)
        assert metric["valid"] or metric["guard_failure"], "Invalid observation/runtime; stop without replacement"
        assert not (folder/"invalid-camera.json").exists()
        if reference:
            compatible(reference, folder, sha(checkpoint/"model.safetensors"))
        event("robot_result", seed=seed, name=name, metrics=metric)
        return folder, metric

    try:
        # Validate all archived diagnostic references before spending candidate compute.
        for seed in plan["diagnostic_seeds"]:
            pre = history/f"diagnostic-{seed}-pretrained"
            native = (resume if seed == 46 else history)/f"diagnostic-{seed}-episode_balanced"
            assert episode_metrics(pre)["valid"] and episode_metrics(native)["valid"]
            compatible(pre, native, sha(baseline/"checkpoints/005000/pretrained_model/model.safetensors"))
        event("archived_references_validated")
        pf = out/"preflight"
        run(train_command(pf)+["--terminal-preflight"], "preflight", plan["preflight_timeout_seconds"])
        assert read(pf/"preflight-passed.json")["passed"]
        event("preflight_passed", sampling=read(pf/"terminal-sampling-audit.json"))
        reference_folder = out/"native-terminal-reference"
        run(train_command(reference_folder)+["--terminal-reference", str(baseline/"checkpoints/005000")],
            "native-terminal-reference", plan["reference_probe_timeout_seconds"])
        reference_probe = read(reference_folder/"reference-probes.json")
        assert reference_probe["passed"]
        assert abs(reference_probe["global_validation"]["task_id"]["mean"]-plan["native_global_validation"]) < 1e-6
        candidate = out/"candidate"
        command = train_command(candidate)+["--preflight-evidence", str(pf/"preflight-passed.json")]
        run(command, "training", plan["training_timeout_seconds"])
        updates = rows(candidate/"updates.jsonl")
        assert len(updates) == 5000 and all(r["all_trainable_gradients_finite"] for r in updates)
        audit = read(candidate/"terminal-sampling-audit.json")
        assert sum(r["valid_action_rows"] for r in updates) == audit["candidate"]["valid_action_rows"]
        assert sum(r["valid_action_rows"] < 50 for r in updates) == audit["candidate"]["padded_chunks"]
        for step in plan["checkpoint_steps"]:
            run(command+["--verify", str(candidate/"checkpoints"/f"{step:06d}")],
                f"reload-{step}", plan["reload_timeout_seconds"])
            assert read(candidate/f"reload-{step:06d}.json")["passed"]
        choice = read(candidate/"selected-checkpoint.json")
        checkpoint = candidate/choice["checkpoint"]/"pretrained_model"
        selected = dict(**choice, path=str(checkpoint), sha256=sha(checkpoint/"model.safetensors"))
        dump(out/"selected-before-robot-evaluation.json", selected)
        event("checkpoint_frozen", **selected)
        native_pairs, pretrained_pairs = [], []
        for seed in plan["diagnostic_seeds"]:
            pre = history/f"diagnostic-{seed}-pretrained"
            native = (resume if seed == 46 else history)/f"diagnostic-{seed}-episode_balanced"
            folder, metric = rollout(seed, f"diagnostic-{seed}", checkpoint, pre)
            matched = compatible(native, folder, selected["sha256"])
            native_pairs.append(dict(seed=seed, baseline=episode_metrics(native), candidate=metric, matched_reset=matched))
            pretrained_pairs.append(dict(seed=seed, baseline=episode_metrics(pre), candidate=metric, matched_reset=matched))
            dump(out/"diagnostic-progress.json", dict(native_pairs=native_pairs, pretrained_pairs=pretrained_pairs))
        policy = read(ROOT/plan["acceptance_policy_source"])["acceptance"]
        chosen_probe = next(r for r in rows(candidate/"probes.jsonl") if r["step"] == choice["step"])
        gate = pilot_decision(native_pairs, pretrained_pairs, choice["score"],
                              chosen_probe["terminal_validation"]["task_id"]["mean"],
                              reference_probe["terminal_validation"]["task_id"]["mean"], plan, policy)
        dump(out/"diagnostic-decision.json", gate)
        event("diagnostic_decision", passed=gate["passed"], reasons=gate["reasons"])
        final_gate, final_pairs, final_seeds = None, [], []
        if gate["passed"]:
            for seed in plan["gated_final_seeds"]:
                native_checkpoint = baseline/"checkpoints/005000/pretrained_model"
                reference, native_metric = rollout(seed, f"final-{seed}-native", native_checkpoint)
                _, metric = rollout(seed, f"final-{seed}-candidate", checkpoint, reference)
                final_pairs.append(dict(seed=seed, baseline=native_metric, candidate=metric, matched_reset=True))
                final_seeds.append(seed)
                dump(out/"final-progress.json", dict(pairs=final_pairs))
            final_gate = acceptance(final_pairs, policy)
            final_gate["reasons"] += per_door_gate(final_pairs, plan["max_per_door_regression_deg"])
            final_gate["passed"] = not final_gate["reasons"]
        accepted = bool(gate["passed"] and final_gate and final_gate["passed"])
        result = dict(accepted=accepted, diagnostic=gate, final=final_gate, final_seeds_evaluated=final_seeds,
                      final_skipped_reason=None if gate["passed"] else "preregistered_diagnostic_gate_failed",
                      selected=selected, optimizer_updates=5000, simulator_episodes=episodes,
                      worker_seconds=times, total_wall_seconds=time.monotonic()-started,
                      gpu_worker_hours=sum(v for k, v in times.items() if k != "preflight")/3600,
                      active_gpu_hours=None, energy_kWh=None,
                      cost_note="GPU worker wall-hours include CPU/I/O; active compute and power not metered")
        dump(out/"result.json", result)
        event("acceptance_decision", accepted=accepted, final_seeds_evaluated=final_seeds)
    except Exception as exc:
        dump(out/"blocked.json", dict(error=str(exc), traceback=traceback.format_exc(),
                                     worker_seconds=times, simulator_episodes=episodes))
        event("blocked", error=str(exc))
        raise
    finally:
        changed = [p for p, value in preservation.items() if sha(ROOT/p) != value]
        dump(out/"preservation-after.json", dict(passed=not changed, files=len(preservation), changed=changed))
        assert not changed, "Historical artifact modification"


if __name__ == "__main__":
    main()
