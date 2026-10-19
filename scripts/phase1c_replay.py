"""Validated, display-only replay of the recorded Phase 1C trajectory.

--validate reconstructs the scene once (including upstream reset settling),
checks historical evidence, and saves a compiled model. Default mode loads that
validated model only. Neither mode loads a policy or executes recorded actions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY = ROOT / "outputs/phase1c/retry-20261018"
DEFAULT_OUT = TRAJECTORY / "replay-ready"
CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
JUMPS = [250, 272, 285, 300, 335, 406, 462]
PINS = {"lerobot": "0.6.1", "robocasa": "1.0.0", "robosuite": "1.5.2",
        "mujoco": "3.3.1", "numpy": "2.2.5"}
REVISIONS = {"robocasa": "56e355ccc64389dfc1b8a61a33b9127b975ba681",
             "robosuite": "aaa8b9b214ce8e77e82926d677b4d61d55e577ab"}
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "outputs/phase1b/numba-cache")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    path.write_text(json.dumps(plain(value), indent=2), encoding="utf-8")


def load_states():
    reset = read(TRAJECTORY / "reset.json")
    rows = [json.loads(line) for line in (TRAJECTORY / "steps.jsonl").read_text().splitlines()]
    require([r["step"] for r in rows] == list(range(1, 601)), "Missing/reordered action endpoints")
    states = [reset["physical"]] + [r["physical"] for r in rows]
    for state in states:
        require(len(state["qpos"]) == 112 and len(state["qvel"]) == 108, "State dimensions differ")
        require(all(np.isfinite(state[k]).all() for k in ("qpos", "qvel", "time")), "Nonfinite recorded state")
    require(np.allclose(np.diff([s["time"] for s in states]), 0.05, atol=1e-12, rtol=0), "Unexpected sample timing")
    return reset, rows, states


def evidence_hashes():
    paths = [TRAJECTORY / n for n in ["reset.json", "steps.jsonl", "manifest.json"]]
    paths += [TRAJECTORY / f"reset-{cam}.png" for cam in CAMERAS]
    paths += [ROOT / "outputs/phase1b/reset.json"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def verify_sources():
    result = {}
    sources = ROOT / "outputs/phase1b/sources"
    for package, revision in REVISIONS.items():
        base = sources / f"{package}-{revision}" if package == "robocasa" else Path(sys.prefix) / "Lib/site-packages"
        count = 0
        with zipfile.ZipFile(sources / f"{package}.zip") as archive:
            for name in archive.namelist():
                relative = name.removeprefix(f"{package}-{revision}/")
                path = base / relative
                if relative.startswith(package + "/") and relative.endswith(".py") and path.is_file():
                    require(path.read_bytes() == archive.read(name), f"Source drift: {path}")
                    count += 1
        require(count == {"robocasa": 494, "robosuite": 204}[package], "Source file inventory differs")
        result[package] = {"revision": revision, "matching_python_files": count}
    return result


def model_inventory(model):
    import mujoco
    names = {}
    for kind, number in [("body", model.nbody), ("joint", model.njnt),
                         ("camera", model.ncam), ("site", model.nsite),
                         ("geom", model.ngeom), ("actuator", model.nu)]:
        enum = getattr(mujoco.mjtObj, "mjOBJ_" + kind.upper())
        names[kind] = [mujoco.mj_id2name(model, enum, i) for i in range(number)]
    return {"sizes": {k: int(getattr(model, k)) for k in
                       ("nq", "nv", "nu", "na", "nmocap", "nbody", "njnt", "ngeom", "nsite", "ncam")},
            "names": names, "body_parentid": model.body_parentid.copy(),
            "jnt_type": model.jnt_type.copy(), "jnt_bodyid": model.jnt_bodyid.copy(),
            "jnt_qposadr": model.jnt_qposadr.copy(), "jnt_dofadr": model.jnt_dofadr.copy(),
            "jnt_range": model.jnt_range.copy(), "cam_bodyid": model.cam_bodyid.copy()}


def apply_state(model, data, state):
    import mujoco
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    data.time = state["time"]
    # No historical forces are available. These values are deliberately not interpreted.
    data.ctrl[:] = 0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    # robosuite's two non-colliding UI targets are not part of the trajectory.
    for bid in np.flatnonzero(model.body_mocapid >= 0):
        mid = model.body_mocapid[bid]
        data.mocap_pos[mid] = model.body_pos[bid]
        data.mocap_quat[mid] = model.body_quat[bid]
    mujoco.mj_forward(model, data)


def geometry_spec(model, robot, low):
    import mujoco
    inv = model_inventory(model)["names"]
    spec = {"eef_site": int(robot.eef_site_id["right"]),
            "base_site": int(model.site(robot.robot_model.base.correct_naming("center")).id),
            "doors": {}, "gripper_geoms": []}
    for i, name in enumerate(inv["geom"]):
        if name and name.startswith("gripper0_right") and (model.geom_contype[i] or model.geom_conaffinity[i]):
            spec["gripper_geoms"].append(i)
    for joint in low.fxtr._fridge_door_joint_names:
        jid = int(model.joint(joint).id)
        body = int(model.jnt_bodyid[jid])
        descendants = {body}
        for bid in range(body + 1, model.nbody):
            if int(model.body_parentid[bid]) in descendants:
                descendants.add(bid)
        geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
        handles = [g for g in geoms if "handle" in (inv["geom"][g] or "").lower()]
        spec["doors"][joint] = {"joint_id": jid, "qpos_address": int(model.jnt_qposadr[jid]),
                                "body_id": body, "body_name": inv["body"][body],
                                "geoms": geoms, "handle_geoms": handles,
                                "handle_names": [inv["geom"][g] for g in handles]}
        require(geoms and handles, f"Missing identifiable door/handle geometry: {joint}")
    require(spec["gripper_geoms"], "Missing gripper collision geometry")
    spec["eef_site_name"] = inv["site"][spec["eef_site"]]
    spec["base_site_name"] = inv["site"][spec["base_site"]]
    return spec


def proximity(model, data, spec):
    """Geometry only. No use of contact forces or claims of measured past contact."""
    import mujoco
    point = data.site_xpos[spec["eef_site"]]
    result = {}
    for name, door in spec["doors"].items():
        distances = {}
        for label, targets in [("door", door["geoms"]), ("handle", door["handle_geoms"])]:
            best = (2.0, None)
            for hand in spec["gripper_geoms"]:
                for target in targets:
                    distance = mujoco.mj_geomDistance(model, data, hand, target, 2.0, None)
                    if distance < best[0]:
                        best = (float(distance), [model.geom(hand).name, model.geom(target).name])
            distances[f"gripper_to_{label}_signed_clearance_m"] = best[0]
            distances[f"gripper_to_{label}_nearest_pair"] = best[1]
        # Explicitly a point-to-body-origin distance, not surface clearance.
        distances["eef_to_door_body_origin_m"] = float(np.linalg.norm(point - data.xpos[door["body_id"]]))
        distances["angle_deg"] = float(np.rad2deg(data.qpos[door["qpos_address"]]))
        result[name] = distances
    return result


def validate(out, reset, rows, states):
    import mujoco
    from PIL import Image
    from lerobot.envs.robocasa import RoboCasaEnv
    baseline = read(ROOT / "outputs/phase1b/reset.json")
    report = {"passed": False, "sources": verify_sources(), "versions": PINS,
              "evidence_hashes": evidence_hashes(), "limitations": [
                  "Original full model/name inventory was not saved; checks use surviving metadata, images and trajectories.",
                  "20 Hz pose replay, not a full integration-state restart; intermediate physics states/forces are unavailable."]}
    env = RoboCasaEnv("CloseFridge", camera_name=CAMERAS)
    try:
        print("Reconstructing seed-42 scene; upstream initialization/reset settling only.", flush=True)
        np.random.seed(42)
        env._ensure_env()
        np.random.seed(42)
        obs, _ = env.reset(seed=42)
        low = env._env.env
        robot = low.robots[0]
        model = deepcopy(low.sim.model._model)
        require((model.nq, model.nv) == (112, 108), "Model state dimensions differ")
        require(model.na == 0, "Unrecorded actuator activation state required")
        mocap_ids = np.flatnonzero(model.body_mocapid >= 0)
        require({model.body(int(b)).name for b in mocap_ids} == {"left_eef_target", "right_eef_target"},
                "Unexpected unrecorded mocap bodies")
        for bid in mocap_ids:
            geoms = np.flatnonzero(model.geom_bodyid == bid)
            require(model.body_parentid[bid] == 0 and not np.any(model.body_parentid == bid), "Mocap target has moving descendants")
            require(not np.any(model.geom_contype[geoms]) and not np.any(model.geom_conaffinity[geoms]), "Mocap target collides")
            require(not np.any(model.eq_obj1id == bid) and not np.any(model.eq_obj2id == bid), "Mocap target constrained to scene")
        report["unrecorded_visual_mocap_targets"] = [model.body(int(b)).name for b in mocap_ids]
        report["reset_errors"] = {}
        for key in ("qpos", "qvel"):
            value = np.asarray(getattr(low.sim.data, key))
            expected = np.asarray(states[0][key])
            error = float(np.max(np.abs(value - expected)))
            report["reset_errors"][key] = {"max_abs": error, "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                                             "original_sha256": reset["hashes"][key]}
            require(np.allclose(value, expected, atol=1e-9, rtol=0), f"Reset {key} differs materially")
        require(np.isclose(low.sim.data.time, states[0]["time"], atol=1e-12), "Reset time differs")
        require(np.allclose(obs["agent_pos"], reset["state"], atol=1e-7, rtol=0), "Reset policy state differs")
        require(plain(low.get_ep_meta()) == baseline["ep_meta"], "Episode metadata differs")
        require(type(low.fxtr).__name__ == baseline["fixture_class"], "Fridge fixture differs")
        require(type(robot).__name__ == baseline["robot_class"], "Robot class differs")
        require(type(robot.robot_model).__name__ == baseline["robot_model"], "Robot model differs")
        require(plain(robot.composite_controller_config) == baseline["controller_config"], "Controller config differs")
        inv = model_inventory(model)
        require(inv["names"]["actuator"] == baseline["actuator_names"], "Actuator names/order differ")
        for name, config in baseline["ep_meta"]["cam_configs"].items():
            cid = int(model.camera(name).id)
            require(inv["names"]["body"][model.cam_bodyid[cid]] == config["parent_body"], f"Camera parent differs: {name}")
            require(np.allclose(model.cam_pos[cid], config["pos"], atol=1e-7), f"Camera position differs: {name}")
            quat = np.asarray(config["quat"]); quat /= np.linalg.norm(quat)
            require(np.allclose(model.cam_quat[cid], quat, atol=1e-7), f"Camera orientation differs: {name}")
        for name, axis in baseline["base_joint_axes"].items():
            require(np.array_equal(model.joint(name).axis, axis), f"Base joint axis differs: {name}")
        require(list(low.fxtr._fridge_door_joint_names) == baseline["fridge_joints"], "Fridge joints differ")
        require(np.isclose(model.opt.timestep, baseline["mujoco_timestep"]), "Timestep differs")
        report["camera_comparison"] = {}
        for cam in CAMERAS:
            expected = np.array(Image.open(TRAJECTORY / f"reset-{cam}.png"))
            actual = obs["pixels"][cam]
            diff = np.abs(actual.astype(float) - expected.astype(float))
            report["camera_comparison"][cam] = {"exact": bool(np.array_equal(actual, expected)),
                                                 "mean_abs_u8": float(diff.mean()), "max_abs_u8": float(diff.max())}
            Image.fromarray(actual).save(out / f"validated-reset-{cam}.png")
            # Permit sparse one-level rasterization rounding, not geometric/image drift.
            require(diff.max() <= 1 and diff.mean() <= 0.001,
                    f"Reset camera image differs materially: {cam}; stop for review")
        spec = geometry_spec(model, robot, low)
        data = mujoco.MjData(model)
        errors = {"base_m": 0.0, "eef_m": 0.0, "door_rad": 0.0}
        diagnostics = []
        # One independent forward-kinematics reconstruction per saved endpoint; no integration.
        for index, state in enumerate(states):
            apply_state(model, data, state)
            for label, site, key in [("base_m", spec["base_site"], "robot0_base_pos"),
                                     ("eef_m", spec["eef_site"], "robot0_eef_pos")]:
                errors[label] = max(errors[label], float(np.linalg.norm(data.site_xpos[site] - state[key])))
            expected_doors = reset["fridge"] if index == 0 else rows[index - 1]["fridge"]
            for name, door in spec["doors"].items():
                errors["door_rad"] = max(errors["door_rad"], abs(data.qpos[door["qpos_address"]] - expected_doors["qpos"][name]))
            if index % 50 == 0 or index in JUMPS:
                diagnostics.append({"step": index, "sim_time": data.time, "geometry": proximity(model, data, spec)})
        report["trajectory_max_errors"] = errors
        # Original robosuite observations can precede the final 2ms integration update.
        require(errors["base_m"] < 0.002 and errors["eef_m"] < 0.002 and errors["door_rad"] < 1e-10,
                f"Recorded trajectory inconsistent with reconstructed geometry: {errors}")
        model_path = out / "validated-scene.mjb"
        mujoco.mj_saveModel(model, str(model_path))
        restored = mujoco.MjModel.from_binary_path(str(model_path))
        require(plain(model_inventory(restored)) == plain(inv), "Model serialization changed structure")
        save(out / "model-inventory.json", inv)
        save(out / "geometry-spec.json", spec)
        save(out / "geometric-diagnostics.json", diagnostics)
        report.update(passed=True, model_sha256=digest(model_path),
                      inventory_sha256=digest(out / "model-inventory.json"),
                      geometry_sha256=digest(out / "geometry-spec.json"),
                      scene_metadata=low.get_ep_meta(), sampled_states_checked=len(states))
        print(json.dumps({"validation_passed": True, "reset": report["reset_errors"],
                          "cameras": report["camera_comparison"], "trajectory": errors}), flush=True)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        save(out / "validation.json", report)
        env.close()


def load_bundle(out):
    import mujoco
    report = read(out / "validation.json")
    require(report["passed"], "Scene validation did not pass; viewer disabled")
    require(report["evidence_hashes"] == evidence_hashes(), "Original evidence changed; revalidation needed")
    for filename, field in [("validated-scene.mjb", "model_sha256"),
                            ("model-inventory.json", "inventory_sha256"), ("geometry-spec.json", "geometry_sha256")]:
        require(digest(out / filename) == report[field], f"Validated bundle changed: {filename}")
    model = mujoco.MjModel.from_binary_path(str(out / "validated-scene.mjb"))
    require(plain(model_inventory(model)) == read(out / "model-inventory.json"), "Loaded model inventory differs")
    return model, read(out / "geometry-spec.json")


def view(out, states):
    import mujoco
    import mujoco.viewer
    model, spec = load_bundle(out)
    data = mujoco.MjData(model)
    commands = queue.Queue()

    def terminal():
        while True:
            try:
                command = input().strip().lower()
            except EOFError:
                return
            commands.put(command)
            if command == "q":
                return

    def key_callback(key):
        if key in (257, 335, 32):  # Enter/keypad Enter/space: next chunk or pause.
            commands.put("")
        elif 49 <= key <= 55:
            commands.put(str(JUMPS[key - 49]))

    print("RECORDED POSES ONLY. No policy, controls or physics integration.")
    print("Enter: next 50-action boundary / pause. Type step 0..600 + Enter to jump. q: close.")
    print("Viewer keys 1..7 jump to: " + ", ".join(map(str, JUMPS)))
    print("Mouse drag rotates; right-drag pans; wheel zooms. Input thread keeps paused viewer responsive.")
    threading.Thread(target=terminal, daemon=True).start()
    index, target = 0, 0
    next_tick = 0.0
    last_printed = None
    apply_state(model, data, states[0])
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = (data.site_xpos[spec["eef_site"]] +
                                    data.xpos[next(iter(spec["doors"].values()))["body_id"]]) / 2
            viewer.cam.distance = 1.5
            viewer.cam.azimuth, viewer.cam.elevation = 0, -20
            viewer.opt.geomgroup[0] = False  # Hide collision meshes; keep visual surfaces clear.
            viewer.opt.geomgroup[2] = False  # Unrecorded GUI target markers.
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        while viewer.is_running():
            while not commands.empty():
                command = commands.get_nowait()
                if command == "q":
                    return
                if command == "":
                    target = index if target > index else min(600, (index // 50 + 1) * 50)
                    next_tick = time.monotonic() + 0.05
                elif command.isdigit() and 0 <= int(command) <= 600:
                    index = target = int(command)
                else:
                    print("Use Enter, a step number 0..600, or q.", flush=True)
            now = time.monotonic()
            if index < target and now >= next_tick:
                index += 1
                next_tick = now + 0.05
            with viewer.lock():
                apply_state(model, data, states[index])
            viewer.sync()
            if last_printed != index:
                last_printed = index
                record = {"step": index, "chunk": (index - 1) // 50 + 1 if index else 0,
                          "sim_time": states[index]["time"],
                          "base_xyz": states[index]["robot0_base_pos"],
                          "base_quat_xyzw": states[index]["robot0_base_quat"],
                          "eef_xyz": states[index]["robot0_eef_pos"],
                          "eef_quat_xyzw": states[index]["robot0_eef_quat"],
                          "gripper_qpos": states[index]["robot0_gripper_qpos"]}
                # Detailed geometry at pauses/jumps; state and joint angles at every endpoint.
                record["door_degrees"] = {n: float(np.rad2deg(data.qpos[d["qpos_address"]])) for n, d in spec["doors"].items()}
                if index == target:
                    record["paused"] = True
                    record["proximity_not_contact_forces"] = proximity(model, data, spec)
                print(json.dumps(record), flush=True)
            time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Reconstruct and validate only; no viewer")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    require(Path(sys.prefix).name == "embodiedops-robocasa", "Use embodiedops-robocasa")
    require({k: md.version(k) for k in PINS} == PINS, "Pinned dependency versions differ")
    reset, rows, states = load_states()
    if args.validate:
        args.output.mkdir(parents=True, exist_ok=True)
        require(not (args.output / "validation.json").exists(), "Validation attempt already exists; preserve it")
        validate(args.output, reset, rows, states)
    else:
        view(args.output, states)


if __name__ == "__main__":
    main()
