"""One approved master-precision candidate using the existing training/eval tools."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from agent_workflow import ROOT, MODEL, read, rows, dump, sha, episode_metrics, acceptance


def supervise_existing_training(out, training_pid, supervisor_pid, event, timeout):
    """Windows-only handoff: suspend the old timeout supervisor, not its worker.

    Keep the old supervisor alive until the training child exits, so its process
    job/pipe lifetime cannot interrupt training. Never resume or restart training.
    """
    import ctypes as ct
    from ctypes import wintypes as wt
    assert sys.platform == "win32" and training_pid != supervisor_pid
    command = ("Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -in "
               f"{training_pid},{supervisor_pid}" +
               " } | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json")
    records = json.loads(subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", command], text=True))
    processes = {r["ProcessId"]: r for r in records}
    assert processes[training_pid]["ParentProcessId"] == supervisor_pid
    assert "phase2_train_v1.py" in processes[training_pid]["CommandLine"]
    assert str(out/"candidate") in processes[training_pid]["CommandLine"]
    assert "optimizer_precision_experiment.py" in processes[supervisor_pid]["CommandLine"]
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel.OpenProcess.restype = wt.HANDLE
    kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    kernel.GetExitCodeProcess.argtypes = [wt.HANDLE, ct.POINTER(wt.DWORD)]
    kernel.TerminateProcess.argtypes = [wt.HANDLE, wt.UINT]
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    kernel.GetProcessTimes.argtypes = [wt.HANDLE] + [ct.POINTER(wt.FILETIME)]*4
    worker = kernel.OpenProcess(0x100000 | 0x1000 | 0x1, False, training_pid)
    supervisor = kernel.OpenProcess(0x800 | 0x1000 | 0x1, False, supervisor_pid)
    assert worker and supervisor, ct.get_last_error()
    stamps = [wt.FILETIME() for _ in range(4)]
    assert kernel.GetProcessTimes(worker, *[ct.byref(t) for t in stamps])
    epoch = ((stamps[0].dwHighDateTime << 32) + stamps[0].dwLowDateTime)/1e7-11644473600
    elapsed = time.time()-epoch
    assert 0 <= elapsed < timeout
    suspend = ct.WinDLL("ntdll").NtSuspendProcess
    suspend.argtypes = [wt.HANDLE]
    suspend.restype = wt.LONG
    assert suspend(supervisor) == 0
    event("runtime_supervision_handoff", training_pid=training_pid,
          previous_supervisor_pid=supervisor_pid, elapsed_training_seconds=elapsed,
          training_restarted=False, previous_supervisor_suspended=True)
    try:
        while True:
            remaining = timeout-(time.time()-epoch)
            if remaining <= 0:
                assert kernel.TerminateProcess(worker, 124)
                raise TimeoutError("Approved extended training limit exhausted")
            status = kernel.WaitForSingleObject(worker, int(min(30, remaining)*1000))
            if status == 0:
                break
            assert status == 258, ct.get_last_error()
        code = wt.DWORD()
        assert kernel.GetExitCodeProcess(worker, ct.byref(code))
        assert code.value == 0, f"Existing training exited {code.value}; no restart"
        return time.time()-epoch
    finally:
        # Training has exited (or has explicitly been stopped at its time limit).
        # Terminate the suspended obsolete supervisor so it cannot act on timeout.
        kernel.TerminateProcess(supervisor, 0)
        kernel.CloseHandle(supervisor)
        kernel.CloseHandle(worker)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--supervise-training-pid", type=int)
    ap.add_argument("--previous-supervisor-pid", type=int)
    args = ap.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT / "outputs/agent-workflow-v1")
    handoff = args.supervise_training_pid is not None
    assert handoff == (args.previous_supervisor_pid is not None)
    if handoff:
        assert (out/"preflight-passed.json").exists() and not (out/"result.json").exists()
        assert not (out/"blocked.json").exists() and not (out/"runtime-amendment.json").exists()
        plan = read(out/"registered-plan.json")
        amendment = dict(reason="Measured sustained throughput exceeds original time allowance",
                         authorization="User explicitly approved extending the runtime caps",
                         original_training_seconds=plan["training_timeout_seconds"],
                         original_main_seconds=plan["max_main_wall_seconds"],
                         training_timeout_seconds=9000, max_main_wall_seconds=12600,
                         optimizer_updates=5000, max_policy_episodes=4, training_restarted=False)
        dump(out/"runtime-amendment.json", amendment)
        plan.update(training_timeout_seconds=9000, max_main_wall_seconds=12600)
        shutil.copyfile(Path(__file__), out/"runtime-supervisor.py.txt")
    else:
        out.mkdir(parents=True, exist_ok=False)
        plan = read(ROOT / "environment/optimizer-precision-v1.json")
        dump(out / "registered-plan.json", plan)
    baseline = ROOT / plan["baseline"]
    history = baseline.parent
    completed = ROOT / "outputs/agent-workflow-v1/learning-5000-resume-v1"
    settings = dict(read(baseline / "settings.json"),
                    optimizer_precision="fp32_master", extra_probe_steps=plan["extra_probe_steps"])
    if handoff:
        assert read(out/"training-config.json") == settings
    else:
        dump(out / "training-config.json", settings)
    for name in ["phase2_train_v1.py", "fp32_master.py", "learning_sampling.py",
                 "optimizer_precision_experiment.py", "phase1d_rollout.py", "renderer_lifecycle.py",
                 "agent_workflow.py", "control_diagnostics.py"]:
        if not handoff:
            shutil.copyfile(ROOT / "scripts" / name, out / (name + ".txt"))
    if handoff:
        preservation = read(out/"preservation-before.json")
    else:
        preservation = {str(p.relative_to(ROOT)): sha(p) for folder in (history, completed)
                        for p in folder.rglob("*") if p.is_file()}
        preservation.update(read(history / "preservation-before.json"))
        dump(out / "preservation-before.json", preservation)
    started = time.monotonic()
    main_started = None
    worker_times = {}

    def event(kind, **values):
        with (out / "trace.jsonl").open("a") as f:
            f.write(json.dumps(dict(event=kind, elapsed_seconds=time.monotonic()-started, **values))+"\n")
        print(kind, json.dumps(values)[:500], flush=True)

    def run(command, name, limit, allowed=(0,)):
        if main_started is not None:
            limit = min(limit, plan["max_main_wall_seconds"]-(time.monotonic()-main_started))
        else:
            limit = min(limit, plan["preflight_timeout_seconds"]-(time.monotonic()-started))
        assert limit > 0, "Registered wall budget exhausted"
        event("worker_start", name=name, command=command, timeout=limit)
        tick = time.monotonic()
        with (out / (name+".log")).open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=limit)
        worker_times[name] = time.monotonic()-tick
        event("worker_end", name=name, seconds=worker_times[name], returncode=result.returncode)
        assert result.returncode in allowed, f"{name} failed: stop, no automatic retry"

    def training_command(folder):
        return [sys.executable, "-B", "-u", "scripts/phase2_train_v1.py",
                "--config", str(out/"training-config.json"), "--output", str(folder),
                "--dataset-cache", str(ROOT/"outputs/pmc-v1")]

    try:
        pf = out / "preflight"
        command = training_command(pf)+["--preflight-steps", "8"]
        if not handoff:
            run(command, "preflight", plan["preflight_timeout_seconds"])
            assert read(pf / "preflight-local.json")["passed"]
            run(command+["--verify", str(pf/"checkpoints/000008")], "preflight-reload",
                plan["preflight_timeout_seconds"])
            verification = read(pf/"reload-000008.json")
            assert verification["passed"] and verification["resumed_update_equal"]
            dump(out/"preflight-passed.json", dict(passed=True, local=read(pf/"preflight-local.json"),
                                                 reload=verification, disposable_updates=10))
            event("preflight_passed", **read(out/"preflight-passed.json"))
        main_started = time.monotonic()
        candidate = out/"candidate"
        command = training_command(candidate)+["--preflight-evidence", str(out/"preflight-passed.json")]
        if handoff:
            previous_trace = rows(out/"trace.jsonl")
            worker_times.update({r["name"]: r["seconds"] for r in previous_trace if r["event"] == "worker_end"})
            train_start = next(r["elapsed_seconds"] for r in previous_trace
                               if r["event"] == "worker_start" and r["name"] == "training")
            seconds = supervise_existing_training(out, args.supervise_training_pid,
                                                   args.previous_supervisor_pid, event,
                                                   plan["training_timeout_seconds"])
            worker_times["training"] = seconds
            main_started = time.monotonic()-seconds
            started = main_started-train_start
            event("worker_end", name="training", seconds=seconds, returncode=0, supervisor_handoff=True)
        else:
            run(command, "training", plan["training_timeout_seconds"])
        updates = rows(candidate/"updates.jsonl")
        assert len(updates) == 5000
        assert all(r["all_trainable_gradients_finite"] for r in updates)
        for step in plan["checkpoint_steps"]:
            run(command+["--verify", str(candidate/"checkpoints"/f"{step:06d}")],
                f"reload-{step}", plan["reload_timeout_seconds"])
            assert read(candidate/f"reload-{step:06d}.json")["passed"]
        selection = read(candidate/"selected-checkpoint.json")
        checkpoint = candidate/selection["checkpoint"]/"pretrained_model"
        selected = dict(**selection, path=str(checkpoint), sha256=sha(checkpoint/"model.safetensors"))
        dump(out/"selected-before-robot-evaluation.json", selected)
        event("checkpoint_selected", **selected)
        pairs, native_pairs, metrics = [], [], {}
        for seed in plan["diagnostic_seeds"]:
            reference = history/f"diagnostic-{seed}-pretrained"
            native = (completed if seed == 46 else history)/f"diagnostic-{seed}-episode_balanced"
            folder = out/f"diagnostic-{seed}"
            command = [sys.executable, "-B", "-u", "scripts/phase1d_rollout.py",
                       "--output", str(folder), "--seed", str(seed), "--checkpoint", str(checkpoint),
                       "--instruction", "CloseFridge", "--workflow-diagnostics", "--renderer-owner-cleanup",
                       "--reference", str(reference)]
            run(command, f"diagnostic-{seed}", plan["episode_timeout_seconds"], allowed=(0, 2))
            m = episode_metrics(folder)
            assert m["valid"] or m["guard_failure"], "Invalid observation/runtime; no automatic replacement"
            manifest = read(folder/"manifest.json")
            baseline_manifest = read(reference/"manifest.json")
            for k in ("versions", "simulator_sources", "resolved_instruction", "execution_horizon", "rng_rule"):
                assert manifest[k] == baseline_manifest[k], k
            for k, value in baseline_manifest["checkpoint_file_hashes"].items():
                if k != "config.json":
                    assert manifest["checkpoint_file_hashes"][k] == value
            assert manifest["model_sha256"] == selected["sha256"]
            matched = read(folder/"paired-reset-validation.json")["passed"]
            metrics[str(seed)] = m
            pairs.append(dict(seed=seed, baseline=episode_metrics(reference), candidate=m, matched_reset=matched))
            native_pairs.append(dict(seed=seed, baseline=episode_metrics(native), candidate=m, matched_reset=matched))
            event("robot_result", seed=seed, metrics=m)
        policy = read(ROOT/plan["acceptance_policy_source"])["acceptance"]
        gate = acceptance(pairs, policy)
        native_gate = acceptance(native_pairs, policy)
        reasons = gate["reasons"][:]
        if selection["score"] >= selection["pretrained_score"]:
            reasons.append("validation_regression_blocks_promotion")
        result = dict(selected=selected, pretrained_comparison=gate, native_precision_comparison=native_gate,
                      accepted=not reasons, reasons=reasons, metrics=metrics, worker_seconds=worker_times,
                      main_wall_seconds=time.monotonic()-main_started, total_wall_seconds=time.monotonic()-started,
                      final_seeds_evaluated=[], active_gpu_hours=None, energy_kWh=None)
        dump(out/"result.json", result)
        event("acceptance_decision", accepted=result["accepted"], reasons=reasons)
    except Exception as exc:
        dump(out/"blocked.json", dict(error=str(exc), worker_seconds=worker_times))
        event("blocked", error=str(exc))
        raise
    finally:
        unchanged = all(sha(ROOT/p) == value for p, value in preservation.items())
        dump(out/"preservation-after.json", dict(passed=unchanged, files=len(preservation)))
        assert unchanged


if __name__ == "__main__":
    main()
