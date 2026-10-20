"""Offline audit of the single optimizer-precision experiment; no inference or physics."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from agent_workflow import ROOT, BASE, read, rows, dump, sha, episode_metrics


def serialized_integrity(folder):
    """CPU-only file audit, not an independent policy/optimizer reload."""
    import torch
    from safetensors import safe_open
    target = read(folder/"verification-target.json")
    checks = [
        ("pretrained_model/model.safetensors", target["model"]),
        ("training_state/fp32_masters.safetensors", target["masters"]),
        ("training_state/optimizer_state.safetensors", {
            f"state/{i}/{name}": value
            for i, state in target["optimizer"]["state"].items()
            for name, value in state.items()}),
    ]
    counts = {}
    for filename, expected in checks:
        with safe_open(str(folder/filename), framework="pt", device="cpu") as f:
            assert set(f.keys()) == set(expected), filename
            for name in f.keys():
                t = f.get_tensor(name)
                digest = hashlib.sha256(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                assert dict(shape=list(t.shape), dtype=str(t.dtype), sha256=digest) == expected[name], name
            counts[filename] = len(expected)
    assert read(folder/"training_state/optimizer_param_groups.json") == target["optimizer"]["param_groups"]
    assert read(folder/"training_state/scheduler_state.json") == target["scheduler"]
    processors = list(BASE.glob("*processor*"))
    assert processors
    for p in processors:
        assert sha(p) == sha(folder/"pretrained_model"/p.name), p.name
    return dict(passed=True, scope="CPU serialized tensor/state hashes only; no policy reload or inference",
                tensor_counts=counts, processor_files=len(processors))


def chart(out, methods):
    """Static scientific figure; each dot is a measured fixed probe."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    title = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 27)
    image = Image.new("RGB", (1500, 690), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 22), "Optimizer precision: measured task-ID flow-matching loss", font=title, fill="#111827")
    colors = {"native": "#2563eb", "master": "#c2410c"}
    for panel, (key, label) in enumerate([
            ("train", "Fixed training probe (32 anchors)"),
            ("validation", "Held-out episodes (104 anchors)")]):
        left, top, width, height = 100+panel*740, 125, 610, 395
        values = [p[key]["task_id"]["mean"] for ps in methods.values() for p in ps]
        lo, hi = min(values)-.01, max(values)+.01
        xy = lambda s, v: (left+s/5000*width, top+height-(v-lo)/(hi-lo)*height)
        draw.text((left, 80), label, font=font, fill="#111827")
        for i in range(6):
            value = lo+(hi-lo)*i/5
            y = xy(0, value)[1]
            draw.line([(left, y), (left+width, y)], fill="#e5e7eb")
            draw.text((left-70, y-10), f"{value:.3f}", font=small, fill="#374151")
            draw.text((left+i*width/5-20, top+height+12), str(i*1000), font=small, fill="#374151")
        initial = methods["native"][0][key]["task_id"]["mean"]
        y = xy(0, initial)[1]
        for x in range(left, left+width, 16):
            draw.line([(x, y), (min(x+8, left+width), y)], fill="#6b7280", width=2)
        for arm, probes in methods.items():
            points = [xy(p["step"], p[key]["task_id"]["mean"]) for p in probes]
            draw.line(points, fill=colors[arm], width=3)
            for x, y in points:
                draw.ellipse((x-3, y-3, x+3, y+3), fill=colors[arm])
        draw.text((left+205, 565), "Optimizer updates", font=font, fill="#374151")
    draw.text((100, 615), "Blue: native optimizer     Orange: FP32 masters     Dashed: original pretrained loss", font=font, fill="#374151")
    draw.text((100, 647), "Both use native BF16/FP32 forward weights. Connecting lines interpolate measured probes.", font=small, fill="#374151")
    image.save(out/"precision-loss-comparison.png")


def contact_detail(folder):
    catalog = {g["id"]: g for g in read(folder/"geometry-catalog.json")}
    peak = None
    robot_external_peak = 0.
    with (folder/"contacts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for substep, sample in enumerate(r["samples"]):
                for c in sample["robot_contacts"]:
                    force = c["force_N_contact_frame"][0]
                    if c["constraint_address"] < 0 or force <= 1e-8:
                        continue
                    gs = [catalog[i] for i in c["geom_ids"]]
                    self_contact = all(g["robot"] for g in gs)
                    if not self_contact:
                        robot_external_peak = max(robot_external_peak, force)
                    if peak is None or force > peak["normal_N"]:
                        peak = dict(normal_N=force, step=r["step"], substep_zero_based=substep,
                                    geoms=[g["name"] for g in gs], robot_self_contact=self_contact)
    return dict(peak=peak, external_peak_normal_N=robot_external_peak)


def parameter_comparison(candidate, native, selected_step):
    import torch
    from safetensors import safe_open
    audit = read(native/"parameter-audit.json")
    old_unchanged = audit["unchanged_trainable"]
    names = list(audit["optimizer_state_dtypes"])
    files = {
        "base": BASE/"model.safetensors",
        "master": candidate/"checkpoints"/f"{selected_step:06d}"/"training_state/fp32_masters.safetensors",
        "forward": candidate/"checkpoints"/f"{selected_step:06d}"/"pretrained_model/model.safetensors",
    }
    native_file = native/"checkpoints"/f"{selected_step:06d}"/"pretrained_model/model.safetensors"
    native_context = safe_open(str(native_file), framework="pt", device="cpu") if native_file.exists() else nullcontext(None)
    result = {}
    with safe_open(str(files["base"]), framework="pt", device="cpu") as original, \
         safe_open(str(files["master"]), framework="pt", device="cpu") as masters, \
         safe_open(str(files["forward"]), framework="pt", device="cpu") as forward, native_context as native_weights:
        for name in names:
            b = original.get_tensor(name)
            f = forward.get_tensor(name)
            assert b.dtype == f.dtype
            entry = dict(native_dtype=str(b.dtype), elements=b.numel(),
                         forward_changed_elements=int(torch.count_nonzero(f != b)),
                         forward_max_abs_delta=float((f.float()-b.float()).abs().max()),
                         forward_l2_delta=float((f.float()-b.float()).norm()))
            if native_weights is not None:
                v = native_weights.get_tensor(name)
                entry.update(baseline_changed_elements=int(torch.count_nonzero(v != b)),
                             baseline_max_abs_delta=float((v.float()-b.float()).abs().max()),
                             baseline_l2_delta=float((v.float()-b.float()).norm()))
            if name in masters.keys():
                m = masters.get_tensor(name)
                assert m.dtype == torch.float32 and torch.equal(m.to(f.dtype), f)
                entry.update(master_changed_elements=int(torch.count_nonzero(m != b.float())),
                             master_max_abs_delta=float((m-b.float()).abs().max()),
                             master_l2_delta=float((m-b.float()).norm()))
            result[name] = entry
    formerly = {n: result[n] for n in old_unchanged}
    return dict(selected_step=selected_step, native_previously_unchanged=formerly,
                formerly_unchanged_count=len(formerly),
                formerly_unchanged_master_now_changed=sum(v.get("master_changed_elements", 0)>0 for v in formerly.values()),
                formerly_unchanged_forward_now_changed=sum(v["forward_changed_elements"]>0 for v in formerly.values()),
                forward_total_l2_delta=math.sqrt(sum(v["forward_l2_delta"]**2 for v in result.values())),
                baseline_same_step_total_l2_delta=(math.sqrt(sum(v["baseline_l2_delta"]**2 for v in result.values()))
                                                   if native_file.exists() else None),
                trainable=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Audit completed training after a recorded blocker; never invent evaluation results.")
    args = parser.parse_args()
    out = args.output.resolve()
    result = read(out/"result.json") if (out/"result.json").exists() else None
    if result is None:
        assert args.allow_incomplete and (out/"blocked.json").exists()
        assert not list(out.glob("diagnostic-*")), "Partial robot results need explicit review"
    plan = read(out/"registered-plan.json")
    candidate, native = out/"candidate", ROOT/plan["baseline"]
    methods = {name: rows(folder/"probes.jsonl") for name, folder in [("native", native), ("master", candidate)]}
    us = {name: rows(folder/"updates.jsonl") for name, folder in [("native", native), ("master", candidate)]}
    assert len(us["native"]) == len(us["master"]) == 5000
    assert read(native/"split.json") == read(candidate/"split.json")
    for old, new in zip(us["native"], us["master"]):
        assert all(old[k] == new[k] for k in ("step", "index", "episode", "sampling_component", "task", "lr_used"))
        assert new["all_trainable_gradients_finite"] and new["tensors_with_grad"] == 155
        assert new["gradient_clipped"] == (new["grad_norm"] > 10)
    draws = {p["step"]: p for p in methods["native"]}
    for p in methods["master"]:
        if p["step"] in draws:
            for split in ("train", "validation"):
                for language in ("natural", "task_id"):
                    assert p[split][language]["draws"] == draws[p["step"]][split][language]["draws"]
    best = min((p for p in methods["master"] if p["step"] in plan["checkpoint_steps"]),
               key=lambda p: (p["validation"]["task_id"]["mean"], p["step"]))
    selection = read(candidate/"selected-checkpoint.json")
    assert best["step"] == selection["step"]
    assert best["validation"]["task_id"]["mean"] == selection["score"]
    if result:
        assert selection["step"] == result["selected"]["step"]
        assert selection["score"] == result["selected"]["score"]
    checkpoint_hashes = {}
    reloads, integrity = {}, {}
    for step in plan["checkpoint_steps"]:
        reload_file = candidate/f"reload-{step:06d}.json"
        reload = read(reload_file) if reload_file.exists() else None
        if result:
            assert reload and reload["passed"] and reload["master_parameters_equal"]
        folder = candidate/"checkpoints"/f"{step:06d}"
        integrity[str(step)] = serialized_integrity(folder)
        checkpoint_hashes[str(step)] = {k: sha(folder/p) for k, p in {
            "forward": "pretrained_model/model.safetensors",
            "master": "training_state/fp32_masters.safetensors"}.items()}
        if reload:
            assert checkpoint_hashes[str(step)]["forward"] == reload["checkpoint_sha256"]
        reloads[str(step)] = reload or dict(passed=None, status=(
            "timed_out" if (out/f"reload-{step}.log").exists() else "not_attempted"))
    audit = read(candidate/"parameter-audit.json")
    assert audit["all_frozen_unchanged"]
    assert all(v["exp_avg"] == v["exp_avg_sq"] == "torch.float32" for v in audit["optimizer_state_dtypes"].values())
    norm = "model.vlm_with_expert.lm_expert.layers.0.input_layernorm.weight"
    initial = methods["native"][0]["validation"]["task_id"]["mean"]
    early = {name: max(p["validation"]["task_id"]["mean"] for p in ps
                      if p["step"] in plan["primary_optimization_test"]["common_early_steps"])
             for name, ps in methods.items()}
    reduction = 1-max(0., early["master"]-initial)/(early["native"]-initial)
    paired_probes = {}
    for p in methods["master"]:
        if p["step"] not in draws:
            continue
        diffs = [x-y for x, y in zip(p["validation"]["task_id"]["samples"],
                                     draws[p["step"]]["validation"]["task_id"]["samples"])]
        paired_probes[str(p["step"])] = dict(anchors=len(diffs), worse=sum(d>0 for d in diffs),
                                             median_difference=median(diffs), mean_difference=mean(diffs))
    training = {}
    for name, folder in [("native", native), ("master", candidate)]:
        summary = read(folder/"summary.json")
        visits = Counter(r["episode"] for r in us[name])
        training[name] = dict(
            probes=[dict(step=p["step"], train_ID=p["train"]["task_id"]["mean"],
                         validation_ID=p["validation"]["task_id"]["mean"]) for p in methods[name]],
            updates=len(us[name]), episodes=len(visits), distinct_anchors=len({r["index"] for r in us[name]}),
            visits_per_episode=[min(visits.values()), max(visits.values())],
            grad_norm_median=median(r["grad_norm"] for r in us[name]),
            grad_norm_max=max(r["grad_norm"] for r in us[name]),
            clipped_updates=sum(r["grad_norm"]>10 for r in us[name]),
            peak_allocated_GiB=summary["peak_allocated"]/2**30,
            peak_reserved_GiB=summary["peak_reserved"]/2**30,
            training_loop_seconds=summary["seconds_including_probes_and_checkpoints"],
            synchronized_update_seconds=sum(r["seconds"] for r in us[name]))
    master_updates = us["master"]
    norm_stats = dict(
        master_changed_updates=sum(r["updates"][norm]["changed_elements"]>0 for r in master_updates),
        forward_changed_updates=sum(r["native_forward_updates"][norm]["changed_elements"]>0 for r in master_updates),
        first_forward_change=next((r["step"] for r in master_updates if r["native_forward_updates"][norm]["changed_elements"]>0), None))
    robot = []
    if result:
        references = [("pretrained", result["pretrained_comparison"]),
                      ("native", result["native_precision_comparison"])]
    else:
        references = []
        for group, suffix in [("pretrained", "pretrained"), ("native", "episode_balanced")]:
            pairs = []
            for seed in plan["diagnostic_seeds"]:
                root = ROOT/"outputs/agent-workflow-v1"/(
                    "learning-5000-resume-v1" if group == "native" and seed == 46 else "learning-5000-v1")
                folder = root/f"diagnostic-{seed}-{suffix}"
                matched = True if group == "pretrained" else read(folder/"paired-reset-validation.json")["passed"]
                pairs.append(dict(seed=seed, baseline=episode_metrics(folder), matched_reset=matched))
            references.append((group, dict(pairs=pairs)))
    pretrained_reference = references[0][1]
    for group, gate in references:
        for pair in gate["pairs"]:
            assert pair["matched_reset"]
            m = dict(pair["baseline"], arm=group, seed=pair["seed"])
            if group == "native":
                ref = next(p["baseline"] for p in pretrained_reference["pairs"]
                           if p["seed"] == pair["seed"])
                a, b = read(ROOT/m["path"]/"manifest.json"), read(ROOT/ref["path"]/"manifest.json")
                assert all(a[k] == b[k] for k in (
                    "versions", "simulator_sources", "resolved_instruction", "execution_horizon", "rng_rule"))
                assert read(ROOT/m["path"]/"paired-reset-validation.json")["passed"]
            m["contact_detail"] = contact_detail(ROOT/m["path"])
            robot.append(m)
    for seed, values in (result["metrics"] if result else {}).items():
        m = dict(values, arm="master", seed=int(seed))
        folder = ROOT/m["path"]
        assert read(folder/"manifest.json")["model_sha256"] == result["selected"]["sha256"]
        assert read(folder/"paired-reset-validation.json")["passed"]
        m["contact_detail"] = contact_detail(folder)
        assert m["contact_detail"]["peak"]["normal_N"] == m["peak_contact_normal_N"]
        robot.append(m)
    groups = {}
    for arm in ("pretrained", "native", "master"):
        rs = [r for r in robot if r["arm"] == arm]
        if not rs and result is None and arm == "master":
            groups[arm] = dict(status="not_run", episodes=0, successes=None)
            continue
        assert sorted(r["seed"] for r in rs) == [43, 44, 45, 46]
        groups[arm] = dict(successes=sum(r["success"] for r in rs), episodes=len(rs),
            mean_closing_deg=mean(r["closing_deg"] for r in rs),
            mean_inference_seconds=mean(r["mean_inference_seconds"] for r in rs),
            guard_failures=sum(r["guard_failure"] for r in rs),
            valid_episodes=sum(r["valid"] for r in rs),
            peak_contact_normal_N=max(r["peak_contact_normal_N"] for r in rs))
    preserved = read(out/"preservation-after.json")
    assert preserved["passed"]
    assert all(sha(ROOT/p) == h for p, h in read(out/"preservation-before.json").items())
    params = parameter_comparison(candidate, native, best["step"])
    dump(out/"parameter-comparison.json", params)
    preflight_params = parameter_comparison(out/"preflight", native, 8)
    dump(out/"preflight-parameter-comparison.json", preflight_params)
    dump(out/"audited-episodes.json", robot)
    preflight_before = read(out/"preflight/parameters-before.json")
    preflight_after = read(out/"preflight/checkpoints/000008/verification-target.json")["model"]
    preflight_changes = [n for n in preflight_before if preflight_before[n] != preflight_after[n]]
    source_file = out/("result.json" if result else "blocked.json")
    source = read(source_file)
    trace = rows(out/"trace.jsonl")
    summary = dict(offline_audit_passed=True, workflow_completed=result is not None,
        source_result_sha256=sha(source_file), training=training, selection=selection,
        serialized_integrity=integrity, independent_reloads=reloads,
        preflight_native_changed_tensors=len(preflight_changes),
        preflight_native_BF16_changed_tensors=sum(preflight_after[n]["dtype"] == "torch.bfloat16" for n in preflight_changes),
        preflight_formerly_unchanged_master_now_changed=preflight_params["formerly_unchanged_master_now_changed"],
        preflight_formerly_unchanged_forward_now_changed=preflight_params["formerly_unchanged_forward_now_changed"],
        norm_update_counts=norm_stats, checkpoint_hashes=checkpoint_hashes,
        initial_validation=initial, early_peak_at_common_steps=early,
        heldout_paired_anchor_differences_vs_native=paired_probes,
        early_excess_reduction_fraction=reduction,
        primary_optimization_test_passed=reduction >= plan["primary_optimization_test"]["minimum_reduction_in_early_validation_excess"],
        formerly_unchanged_master_now_changed=params["formerly_unchanged_master_now_changed"],
        formerly_unchanged_forward_now_changed=params["formerly_unchanged_forward_now_changed"],
        selected_forward_total_l2_delta=params["forward_total_l2_delta"],
        native_same_step_total_l2_delta=params["baseline_same_step_total_l2_delta"],
        robot_groups=groups, accepted=result["accepted"] if result else None,
        acceptance_gate_status="completed" if result else "not_run",
        reasons=result["reasons"] if result else ["independent_reload_timeout", "validation_regression_blocks_promotion"],
        blocker=None if result else source["error"],
        worker_seconds=source["worker_seconds"],
        runtime_amendment=read(out/"runtime-amendment.json") if (out/"runtime-amendment.json").exists() else None,
        completed_gpu_worker_wall_hours=sum(source["worker_seconds"].values())/3600,
        elapsed_through_stop_seconds=trace[-1]["elapsed_seconds"],
        main_wall_seconds=result["main_wall_seconds"] if result else None,
        active_gpu_hours=None, energy_kWh=None,
        preservation=preserved, final_seeds_evaluated=result["final_seeds_evaluated"] if result else [])
    dump(out/"audited-summary.json", summary)
    chart(out, methods)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
