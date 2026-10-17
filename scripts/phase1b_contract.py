"""Bounded simulator-only contract inspection. Never imports a learned policy."""
from __future__ import annotations

import json
import os
import sys
import traceback
import faulthandler
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path("outputs/phase1b")
CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
PROBES = [("eef_x_positive", 5, 0.05), ("base_x_positive", 0, 0.05),
          ("gripper_close", 11, 1.0), ("desired_mode", 4, 1.0)]


def neutral_action():
    action = np.zeros(12, dtype=np.float32)
    action[4] = -1  # achieved-pose arm goal updates
    action[11] = -1  # open/continue opening; no intermediate-hold code exists
    return action


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def save(name, value):
    (OUT / name).write_text(json.dumps(serial(value), indent=2), encoding="utf-8")


def sheet(obs, label):
    canvas = Image.new("RGB", (768, 288), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (cam, semantic) in enumerate(zip(CAMERAS, ["LEFT", "RIGHT", "WRIST"])):
        canvas.paste(Image.fromarray(obs["pixels"][cam]), (i * 256, 32))
        draw.text((i * 256 + 4, 4), f"{label}: {semantic} / camera{i+1}", fill="black")
    return canvas


def physical(low):
    raw = low._get_observations(force_update=True)
    keys = ["robot0_base_pos", "robot0_base_quat", "robot0_eef_pos", "robot0_eef_quat",
            "robot0_base_to_eef_pos", "robot0_base_to_eef_quat", "robot0_gripper_qpos"]
    return {"time": float(low.sim.data.time), "qpos": low.sim.data.qpos.copy(),
            "qvel": low.sim.data.qvel.copy(), **{k: raw[k].copy() for k in keys}}


def delta(before, after):
    return {k: after[k] - before[k] for k in before}


def measured_step(env, action):
    from lerobot.envs.robocasa import convert_action
    low = env._env.env
    robot = low.robots[0]
    cc = robot.composite_controller
    before = physical(low)
    trace = {"raw_action": action.copy(), "action_dict": deepcopy(convert_action(action)),
             "unmapped": deepcopy(env._env.key_converter.unmap_action(convert_action(action))),
             "before": before, "substeps": []}
    trace["base_forward_reversed"] = bool(cc.part_controllers["base"]._check_forward_joint_reversed())
    old_goal, old_pre = cc.set_goal, low._pre_action

    def goal(value):
        trace["controller_input"] = value.copy()
        result = old_goal(value)
        fields = ["goal_pos", "goal_ori", "goal_qpos", "goal_qvel", "_goal_update_mode",
                  "origin_pos", "origin_ori", "ref_pos", "ref_ori_mat", "init_ori"]
        trace["targets"] = {k: {f: deepcopy(getattr(c, f)) for f in fields if hasattr(c,f)}
                            for k,c in cc.part_controllers.items()}
        trace["gripper_accumulator"] = {k: g.current_action.copy() for k,g in robot.gripper.items()}
        return result

    def pre(value, policy_step=False):
        result = old_pre(value, policy_step)
        trace["substeps"].append({"time_before_physics": float(low.sim.data.time),
                                  "policy_step": policy_step, "ctrl": low.sim.data.ctrl.copy()})
        return result

    cc.set_goal, low._pre_action = goal, pre
    try:
        obs, reward, terminated, truncated, info = env.step(action.copy())
    finally:
        cc.set_goal, low._pre_action = old_goal, old_pre
    after = physical(low)
    trace.update(after=after, delta=delta(before, after), reward=reward, terminated=terminated,
                 truncated=truncated, info=info, state=obs["agent_pos"].copy(),
                 finite=bool(np.isfinite(after["qpos"]).all() and np.isfinite(after["qvel"]).all()
                             and np.isfinite(obs["agent_pos"]).all()
                             and all(np.isfinite(x).all() for x in obs["pixels"].values())))
    assert trace["finite"], "Nonfinite state: stop"
    assert not terminated and not truncated, "Unexpected termination: stop before further probes"
    assert len(trace["substeps"]) == round(low.control_timestep / low.model_timestep)
    assert np.isclose(after["time"] - before["time"], low.control_timestep)
    # Conservative diagnostic limits, not benchmark acceptance thresholds.
    for k in ["robot0_base_pos", "robot0_eef_pos"]:
        assert np.linalg.norm(trace["delta"][k]) < 0.05, f"Unexpected >5cm one-step motion: {k}"
    return obs, trace


def controls(env, initial_obs):
    neutral = neutral_action()
    result = {"neutral_action": neutral, "neutral_steps": [], "probes": []}
    frames = [sheet(initial_obs, "RESET")]
    low = env._env.env
    initial = physical(low)
    for i in range(5):
        obs, trace = measured_step(env, neutral)
        result["neutral_steps"].append(trace)
        frames.append(sheet(obs, f"NEUTRAL {i+1}"))
        save("controls.json", result)
    result["neutral_delta"] = delta(initial, physical(low))
    # One step per conservative probe, each from the identical seeded reset.
    for name, index, value in PROBES:
        np.random.seed(42)
        obs, _ = env.reset(seed=42)
        reset = physical(env._env.env)
        assert reset["qpos"].shape == initial["qpos"].shape
        assert np.allclose(reset["qpos"], initial["qpos"], atol=1e-9, rtol=0), "Seeded reset differs: stop"
        assert np.allclose(reset["qvel"], initial["qvel"], atol=1e-9, rtol=0), "Reset velocity differs: stop"
        action = neutral.copy()
        action[index] = value
        obs, trace = measured_step(env, action)
        trace["probe"] = name
        result["probes"].append(trace)
        save("controls.json", result)
        if name == "eef_x_positive":
            target = trace["targets"]["right"]
            base = trace["before"]["robot0_base_to_eef_pos"]
            assert np.isclose(target["goal_pos"][0] - base[0], 0.0025, atol=1e-5)
        elif name == "base_x_positive":
            axis = 1 if trace["base_forward_reversed"] else 0
            assert np.isclose(trace["targets"]["base"]["goal_qvel"][axis], 0.05, atol=1e-5)
        elif name == "desired_mode":
            assert trace["targets"]["right"]["_goal_update_mode"] == "desired"
        frames.append(sheet(obs, name))
    # Slow replay uses repeated rendered frames; it adds no simulation steps.
    import av
    with av.open(str(OUT / "controller-diagnostic.mp4"), mode="w") as video:
        stream = video.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 768, 288, "yuv420p"
        for frame in frames:
            for _ in range(5):
                for packet in stream.encode(av.VideoFrame.from_image(frame)):
                    video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    save("controls.json", result)
    print("Five neutral steps and four one-step probes completed", flush=True)


def interactive_replay(env):
    """Manual viewer for five neutral steps and the three requested probes."""
    import mujoco
    import mujoco.viewer

    faulthandler.cancel_dump_traceback_later()  # User-controlled pauses have no timeout.
    initial = physical(env._env.env)
    stages = [("five neutral steps (gripper opens)", None, None, 5)]
    stages += [(name, index, value, 1) for name, index, value in PROBES[:3]]
    print("Use Enter in this PowerShell prompt to advance; Ctrl+C to stop.", flush=True)
    for stage, index, value, count in stages:
        if index is not None:
            np.random.seed(42)
            env.reset(seed=42)
            reset = physical(env._env.env)
            for key in ("qpos", "qvel"):
                assert reset[key].shape == initial[key].shape
                assert np.allclose(reset[key], initial[key], atol=1e-9, rtol=0), "Seeded reset differs: stop"
        low = env._env.env
        # A display-only copy keeps viewer interactions out of simulator state.
        model = deepcopy(low.sim.model._model)
        data = mujoco.MjData(model)

        def refresh(viewer):
            with viewer.lock():
                data.qpos[:] = low.sim.data.qpos
                data.qvel[:] = low.sim.data.qvel
                data.ctrl[:] = low.sim.data.ctrl
                data.time = low.sim.data.time
                mujoco.mj_forward(model, data)
            viewer.sync()

        # Hard resets replace the MuJoCo model: close/reopen the viewer per stage.
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERAS[0])
            refresh(viewer)
            action = neutral_action()
            if index is not None:
                action[index] = value
            print(f"\nRESET seed 42 -> {stage}\nPublic action: {action.tolist()}", flush=True)
            input("Inspect reset in viewer. Press Enter here to execute the first step: ")
            for i in range(count):
                if not viewer.is_running():
                    return
                _, trace = measured_step(env, action)
                refresh(viewer)
                print(f"{stage}: step {i+1}/{count}; sim time {low.sim.data.time:.3f}s", flush=True)
                for key in ("robot0_base_pos", "robot0_eef_pos", "robot0_gripper_qpos"):
                    print(f"  delta {key} (mm): {np.asarray(trace['delta'][key]) * 1000}", flush=True)
                input("Inspect result. Press Enter here for the next step/stage (or finish): ")
    print("Replay complete: five neutral steps + EEF/base/gripper single-step probes.", flush=True)


def main():
    global OUT
    faulthandler.dump_traceback_later(60, repeat=True)
    print("Starting simulator contract inspection", flush=True)
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--controls", action="store_true", help="Run exactly five neutral steps and four one-step probes")
    mode.add_argument("--interactive", action="store_true", help="Manually replay five neutral steps and EEF/base/gripper probes in a viewer")
    args = parser.parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", str((OUT / "numba-cache").resolve()))
    if args.interactive:
        faulthandler.cancel_dump_traceback_later()
        OUT = OUT / "manual-replay"
    assert Path(sys.prefix).name == "embodiedops-robocasa", sys.prefix
    from lerobot.envs.robocasa import RoboCasaEnv
    print("LeRobot environment wrapper imported", flush=True)
    import importlib.metadata as md
    OUT.mkdir(parents=True, exist_ok=True)
    expected = {"lerobot": "0.6.1", "torch": "2.11.0+cu128", "numpy": "2.2.5",
                "robocasa": "1.0.0", "robosuite": "1.5.2", "mujoco": "3.3.1",
                "opencv-python-headless": "4.13.0.92"}
    for name, version in expected.items():
        assert md.version(name) == version, f"Dependency drift: {name}"
    assert not any(d.metadata["Name"].lower() == "opencv-python" for d in md.distributions())
    np.random.seed(42)
    env = RoboCasaEnv("CloseFridge", camera_name=CAMERAS)
    try:
        env._ensure_env()
        np.random.seed(42)
        obs, info = env.reset(seed=42)
        if args.interactive:
            interactive_replay(env)
            return
        low = env._env.env
        robot = low.robots[0]
        cc = robot.composite_controller
        raw = low._get_observations(force_update=True)
        controller_fields = ["input_min", "input_max", "output_min", "output_max",
                             "input_type", "input_ref_frame", "kp", "kd", "impedance_mode",
                             "actuator_min", "actuator_max", "goal_pos", "goal_ori",
                             "goal_qpos", "goal_qvel", "_goal_update_mode"]
        record = {
            "seed": 42, "python": sys.version, "versions": {
                k: md.version(k) for k in ["lerobot", "torch", "numpy", "opencv-python-headless",
                                          "robocasa", "robosuite", "mujoco", "numba", "scipy"]},
            "task_class": f"{type(low).__module__}.{type(low).__name__}",
            "wrapper_classes": [type(env).__name__, type(env._env).__name__],
            "observation_keys": list(obs), "raw_keys": list(raw),
            "images": {k: {"shape": v.shape, "dtype": str(v.dtype)} for k,v in obs["pixels"].items()},
            "state": obs["agent_pos"], "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "control_freq": low.control_freq, "model_timestep": low.model_timestep,
            "mujoco_timestep": low.sim.model.opt.timestep, "control_timestep": low.control_timestep,
            "horizon": low.horizon, "ignore_done": low.ignore_done,
            "lerobot_max_episode_steps": env._max_episode_steps,
            "task_instruction": env.task_description, "ep_meta": low.get_ep_meta(),
            "robot_class": type(robot).__name__, "robot_model": type(robot.robot_model).__name__,
            "controller_config": robot.composite_controller_config,
            "action_splits": cc._action_split_indexes,
            "controllers": {k: {"class": type(c).__name__, **{f: getattr(c,f) for f in controller_fields if hasattr(c,f)}} for k,c in cc.part_controllers.items()},
            "grippers": {k: {"class": type(g).__name__, "current_action": g.current_action, "init_qpos": g.init_qpos} for k,g in robot.gripper.items()},
            "fixture_class": type(low.fxtr).__name__, "fridge_joints": low.fxtr._fridge_door_joint_names,
            "fridge_joint_states": low.fxtr.get_joint_state(low, low.fxtr._fridge_door_joint_names),
            "base_forward_reversed": bool(cc.part_controllers["base"]._check_forward_joint_reversed()),
            "base_joint_axes": {n: low.sim.model.jnt_axis[low.sim.model.joint_name2id(n)] for n in cc.part_controllers["base"].joint_names},
            "actuator_names": low.sim.model.actuator_names,
            "success": bool(low._check_success()), "sim_time": low.sim.data.time,
        }
        save("reset.json", record)
        sheet(obs, "RESET").save(OUT / "reset-cameras.png")
        print(json.dumps(serial(record), indent=2), flush=True)
        if args.controls:
            controls(env, obs)
    finally:
        faulthandler.cancel_dump_traceback_later()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
