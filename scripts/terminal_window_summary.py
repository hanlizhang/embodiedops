"""Read-only completed-run analysis; uses the repository's dependency-free SVG style."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path

import numpy as np

from agent_workflow import ROOT, read, rows, dump


def svg_panels(path, panels, heading, xlabel):
    """Small static plots of measured points; no interpolation beyond straight segments."""
    width, height = 1050, 300*len(panels)+70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="70" y="28" font-family="Arial" font-size="20">{escape(heading)}</text>']
    for panel, (title, ylabel, series) in enumerate(panels):
        top = 65+panel*300
        points = [(x, y) for _, _, values in series for x, y in values]
        xmax = max(x for x, y in points) or 1
        ymin = min(0, min(y for x, y in points))
        ymax = max(y for x, y in points)*1.08 or 1
        for i in range(5):
            yy = top+180-i*45
            val = ymin+(ymax-ymin)*i/4
            xx = 75+i*230
            parts += [f'<path d="M75 {yy}H995" stroke="#ddd"/>',
                      f'<text x="8" y="{yy+5}" font-size="12">{val:.3g}</text>',
                      f'<text x="{xx}" y="{top+200}" font-size="12">{xmax*i/4:.0f}</text>']
        parts.append(f'<text x="75" y="{top-12}" font-size="16">{escape(title)} — {escape(ylabel)}</text>')
        parts.append(f'<text x="450" y="{top+219}" font-size="12">{escape(xlabel)}</text>')
        for k, (label, color, values) in enumerate(series):
            coordinates = " ".join(f"{75+x/xmax*920:.2f},{top+180-(y-ymin)/(ymax-ymin)*180:.2f}" for x, y in values)
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in values if len(values) <= 12 else []:
                parts.append(f'<circle cx="{75+x/xmax*920:.2f}" cy="{top+180-(y-ymin)/(ymax-ymin)*180:.2f}" r="3" fill="{color}"/>')
            parts.append(f'<text x="{75+(k%3)*305}" y="{top+240+(k//3)*17}" fill="{color}" font-size="12">{escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def physical_details(folder):
    reset, logs = read(folder/"reset.json"), rows(folder/"steps.jsonl")
    names = list(reset["fridge"]["qpos"])
    states = [reset]+logs
    angles = np.rad2deg([[r["fridge"]["qpos"][n] for n in names] for r in states])
    catalog = {g["id"]: g for g in read(folder/"geometry-catalog.json")}
    door_bodies = {n: n.removesuffix("_joint") for n in names}
    for body in door_bodies.values():
        assert any(body in g["body_ancestors"] for g in catalog.values())
    loaded = {n: set() for n in names}
    categories, identities = {}, {}
    for row in rows(folder/"contacts.jsonl"):
        for sample in row["samples"]:
            for contact in sample["robot_contacts"]:
                force = contact["force_N_contact_frame"][0]
                if contact["constraint_address"] < 0 or force <= 1e-8:
                    continue
                geoms = [catalog[g] for g in contact["geom_ids"]]
                external = [g for g in geoms if not g["robot"]]
                for name, body in door_bodies.items():
                    if any(body in g["body_ancestors"] for g in external):
                        loaded[name].add(row["step"])
                door = any(body in g["body_ancestors"] for body in door_bodies.values() for g in external)
                wall = any("wall" in " ".join([g["name"]]+g["body_ancestors"]).lower() for g in external)
                category = "self" if not external else ("door" if door else ("wall" if wall else "other_external"))
                categories[category] = max(categories.get(category, 0), force)
                identity = " / ".join(g["name"] for g in geoms)
                identities[identity] = max(identities.get(identity, 0), force)
    eef = np.asarray([r["physical"]["robot0_eef_pos"] for r in states])
    doors = {}
    for i, name in enumerate(names):
        plateau = next(k for k in range(len(states)) if np.ptp(angles[k:, i]) < .1)
        contacts = sorted(loaded[name])
        runs = []
        for step in contacts:
            if not runs or step-runs[-1][-1] > 20:
                runs.append([step])
            else:
                runs[-1].append(step)
        doors[name] = dict(initial_deg=float(angles[0, i]), final_deg=float(angles[-1, i]),
                           minimum_deg=float(angles[:, i].min()), minimum_step=int(angles[:, i].argmin()),
                           closure_deg=float(angles[0, i]-angles[-1, i]),
                           maximum_reopening_deg=float(np.max(angles[:, i]-np.minimum.accumulate(angles[:, i]))),
                           final_range_under_point1_deg_start=plateau,
                           final_plateau_seconds=(len(logs)-plateau)/20,
                           eef_world_path_after_plateau_m=float(np.linalg.norm(np.diff(eef[plateau:], axis=0), axis=1).sum()),
                           loaded_contact_actions=len(contacts),
                           first_loaded_contact=contacts[0] if contacts else None,
                           last_loaded_contact=contacts[-1] if contacts else None,
                           loaded_contact_runs_gap_over_1s=[[r[0], r[-1]] for r in runs])
    return dict(doors=doors, peak_normal_N_by_category=categories,
                top_loaded_geom_pairs=sorted(identities.items(), key=lambda x: -x[1])[:8],
                force_scope="positive contact-local normal solver force, not world wrench or hinge torque"), angles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    assert out.is_relative_to(run)
    result = read(run/"result.json")
    assert read(run/"preservation-after.json")["passed"]
    out.mkdir(exist_ok=False)
    baseline = ROOT/read(run/"registered-plan.json")["baseline"]
    candidate = run/"candidate"
    probes = {"Episode-balanced": rows(baseline/"probes.jsonl"), "Terminal-window": rows(candidate/"probes.jsonl")}
    colors = {"Episode-balanced": "#2563eb", "Terminal-window": "#c2410c"}
    panels = []
    for field, title in (("train", "Original fixed training anchors"), ("validation", "Original global held-out anchors")):
        series = [(name, colors[name], [(r["step"], r[field]["task_id"]["mean"]) for r in measured])
                  for name, measured in probes.items()]
        panels.append((title, "Flow Matching loss", series))
    native_terminal = read(run/"native-terminal-reference/reference-probes.json")["terminal_validation"]["task_id"]["mean"]
    panels.append(("Held-out terminal anchors (baseline measured only at selected step 5000)",
                   "Flow Matching loss", [
                       ("Terminal-window", colors["Terminal-window"],
                        [(r["step"], r["terminal_validation"]["task_id"]["mean"]) for r in probes["Terminal-window"]]),
                       ("Native selected reference only", colors["Episode-balanced"], [(5000, native_terminal)])]))
    svg_panels(out/"loss-comparison.svg", panels, "Terminal-window repair: actual fixed losses", "Optimizer update")
    curves = {name: [dict(step=r["step"], fixed_train=r["train"]["task_id"]["mean"],
                          global_validation=r["validation"]["task_id"]["mean"],
                          terminal_validation=r.get("terminal_validation", {}).get("task_id", {}).get("mean"))
                    for r in measured] for name, measured in probes.items()}
    dump(out/"loss-curves.json", curves)
    training = {}
    for name, folder in (("Episode-balanced", baseline), ("Terminal-window", candidate)):
        updates = rows(folder/"updates.jsonl")
        assert len(updates) == 5000
        training[name] = dict(summary=read(folder/"summary.json"),
                             clipped_updates=sum(r.get("gradient_clipped", r["grad_norm"] > 10) for r in updates),
                             clipping_count_source="recorded flag, or archived pre-clip norm > unchanged threshold10",
                             grad_norm_quantiles=np.quantile([r["grad_norm"] for r in updates], [0, .5, .95, 1]).tolist(),
                             finite_gradient_updates=sum(r["all_trainable_gradients_finite"] for r in updates),
                             lr_first=updates[0]["lr_used"], lr_peak=max(r["lr_used"] for r in updates),
                             lr_last=updates[-1]["lr_used"],
                             parameter_changes=read(folder/"parameter-audit.json"))
    native_updates, new_updates = rows(baseline/"updates.jsonl"), rows(candidate/"updates.jsonl")
    assert [r["episode"] for r in native_updates] == [r["episode"] for r in new_updates]
    assert [r["lr_used"] for r in native_updates] == [r["lr_used"] for r in new_updates]
    actual = Counter(r["sampling_component"] for r in new_updates)
    assert dict(actual) == read(candidate/"terminal-sampling-audit.json")["components"]
    physical, panels = {}, []
    for seed in (43, 44, 45, 46):
        series = []
        for name, folder in (
            ("pretrained", baseline.parent/f"diagnostic-{seed}-pretrained"),
            ("native", (ROOT/"outputs/agent-workflow-v1/learning-5000-resume-v1" if seed == 46 else baseline.parent)/f"diagnostic-{seed}-episode_balanced"),
            ("terminal", run/f"diagnostic-{seed}")):
            details, angles = physical_details(folder)
            physical[f"{seed}-{name}"] = details
            palette = {"pretrained": ["#999999", "#555555"], "native": ["#60a5fa", "#1d4ed8"], "terminal": ["#fb923c", "#c2410c"]}
            for i in range(angles.shape[1]):
                series.append((f"{name} door {i+1}", palette[name][i],
                               list(zip(np.arange(len(angles)).tolist(), angles[:, i].tolist()))))
        panels.append((f"Seed {seed}", "Door opening (degrees)", series))
    svg_panels(out/"door-progress.svg", panels, "Paired CloseFridge diagnostic scenes", "Executed action (20 Hz)")
    dump(out/"analysis.json", dict(training=training, physical=physical, decision=result,
                                  sample_distribution=read(candidate/"terminal-sampling-audit.json"),
                                  terminal_reference=native_terminal,
                                  labels_are_contacts=False, recovery_demonstrations_verified=False))
    print(json.dumps(dict(accepted=result["accepted"], curves=str(out/"loss-comparison.svg"),
                          physical_plot=str(out/"door-progress.svg"))))


if __name__ == "__main__":
    main()
