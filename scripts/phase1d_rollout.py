"""One guarded Phase 1D evaluation episode; inherited Phase 1C execution semantics.

Uses the installed LeRobot 0.6.1 evaluation processors and select_action queue.
Existing Phase 1A cache and Phase 1B simulator/cache are required. Run from repo root.
"""
from __future__ import annotations

import hashlib
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/phase1d/language-control"
os.environ["HF_HOME"] = str(ROOT / "outputs/phase1a/hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "outputs/phase1b/numba-cache")

import numpy as np
from PIL import Image
from phase1a_contract import MODEL, MODEL_REV, BACKBONE_REV, CAMERAS as CAMERA_MAP
from phase1b_contract import CAMERAS, serial, sheet, physical, delta


def save(name, value):
    (OUT / name).write_text(json.dumps(serial(value), indent=2), encoding="utf-8")


def emit(name, value):
    save(name + ".json", value)
    print(json.dumps({"stage": name, "value": serial(value)}), flush=True)


def tensor_info(value):
    import torch
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu()
        return {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(value.device),
                "min": x.min().item(), "max": x.max().item(), "finite": bool(torch.isfinite(x).all())}
    if isinstance(value, dict):
        return {k: tensor_info(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_info(v) for v in value]
    return value


def frames(obs, label):
    sheet(obs, label).save(OUT / f"{label}-cameras.png")
    for cam in CAMERAS:
        Image.fromarray(obs["pixels"][cam]).save(OUT / f"{label}-{cam}.png")


def fridge(low):
    names = low.fxtr._fridge_door_joint_names
    return {"normalized": low.fxtr.get_joint_state(low, names),
            "qpos": {n: float(low.sim.data.get_joint_qpos(n)) for n in names},
            "success": bool(low._check_success())}


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, (np.ndarray, float, np.floating)):
        return bool(np.isfinite(value).all())
    return True


def canonical_scene(value, key=None):
    """Only object category membership is unordered; poses and other lists retain order.

    RoboCasa kitchen_objects.py builds these groups by iterating all_types, a set.
    The recorded raw metadata is retained; this affects comparison only.
    """
    if isinstance(value, dict):
        return {k: canonical_scene(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if key == "groups_containing_sampled_obj":
            assert all(isinstance(v, str) for v in value)
            return sorted(value)
        return [canonical_scene(v) for v in value]
    return value


def action_gate(env, action):
    """No clipping here: retain the Phase 1B controller's actual semantics."""
    assert action.shape == (12,) and np.isfinite(action).all(), "Invalid public action"
    cc = env._env.env.robots[0].composite_controller
    assert type(cc).__name__ == "HybridMobileBase"
    assert not cc.part_controllers["base"]._check_forward_joint_reversed()
    outside = []
    for i in np.flatnonzero(np.abs(action) > 1):
        if i < 3:
            path = "raw base command -> initial/current yaw rotation -> actuator range mapping -> final actuator clipping"
        elif i == 3:
            path = "torso clips [-1,1], scales by 0.05 m, adds achieved joint position"
        elif i in (4, 11):
            path = "threshold at 0.5 to -1 or +1"
        else:
            path = "OSC clips [-1,1]; translation scales 0.05 m, rotation scales 0.5 rad"
        outside.append({"dimension": int(i), "value": float(action[i]), "validated_path": path})
    return {"finite": True, "outside_nominal_box": outside,
            "mode": "achieved" if action[4] < 0.5 else "desired",
            "gripper": "open" if action[11] < 0.5 else "close", "added_clipping": False}


def traced_step(env, action, index):
    """Phase 1B's set_goal/_pre_action instrumentation, retaining terminal traces."""
    from lerobot.envs.robocasa import convert_action
    low = env._env.env
    robot = low.robots[0]
    cc = robot.composite_controller
    before = physical(low)
    trace = {"step": index, "raw_action": action.copy(), "before": before,
             "fridge_before": fridge(low), "action_dict": deepcopy(convert_action(action)),
             "unmapped": deepcopy(env._env.key_converter.unmap_action(convert_action(action))), "substeps": []}
    old_goal, old_pre = cc.set_goal, low._pre_action

    def goal(value):
        trace["controller_input"] = value.copy()
        result = old_goal(value)
        fields = ["goal_pos", "goal_ori", "goal_qpos", "goal_qvel", "_goal_update_mode",
                  "origin_pos", "origin_ori", "ref_pos", "ref_ori_mat", "init_ori"]
        trace["targets"] = {k: {f: deepcopy(getattr(c, f)) for f in fields if hasattr(c, f)}
                            for k, c in cc.part_controllers.items()}
        trace["gripper_accumulator"] = {k: g.current_action.copy() for k, g in robot.gripper.items()}
        assert finite_tree(trace["targets"]), "Nonfinite controller target"
        return result

    def pre(value, policy_step=False):
        result = old_pre(value, policy_step)
        trace["substeps"].append({"time_before_physics": float(low.sim.data.time),
                                  "policy_step": policy_step, "ctrl": low.sim.data.ctrl.copy()})
        assert np.isfinite(low.sim.data.ctrl).all(), "Nonfinite MuJoCo ctrl"
        return result

    cc.set_goal, low._pre_action = goal, pre
    try:
        obs, reward, terminated, truncated, info = env.step(action.copy())
    except Exception:
        trace["exception"] = traceback.format_exc()
        save("failed-step-trace.json", trace)
        raise
    finally:
        cc.set_goal, low._pre_action = old_goal, old_pre
    after = physical(low)
    expected_input = np.concatenate((action[5:11], [1 if action[11] >= 0.5 else -1],
                                     action[:4], [1 if action[4] >= 0.5 else -1]))
    trace.update(after=after, delta=delta(before, after), reward=reward, terminated=terminated,
                 truncated=truncated, info=info, state=obs["agent_pos"].copy(), fridge_after=fridge(low),
                 finite=finite_tree(after) and finite_tree(obs))
    # Same conservative one-step translation limit as Phase 1B; add MuJoCo warnings.
    trace["warnings"] = np.asarray(low.sim.data._data.warning.number).copy()
    failures = []
    if not np.array_equal(trace["controller_input"], expected_input):
        failures.append("controller action reorder differs from Phase 1B")
    if not trace["finite"]:
        failures.append("nonfinite observation/physical state")
    if len(trace["substeps"]) != 25 or not np.isclose(after["time"] - before["time"], 0.05):
        failures.append("unexpected physics substep count/time")
    for k in ("robot0_base_pos", "robot0_eef_pos"):
        if np.linalg.norm(trace["delta"][k]) >= 0.05:
            failures.append(f"{k}: >=5cm one-step displacement")
    if np.any(trace["warnings"]):
        failures.append("MuJoCo warning counter nonzero")
    trace["guard_failures"] = failures
    if index == 1 or failures:
        save("first-action-trace.json" if index == 1 else "unstable-step-trace.json", trace)
    return obs, trace


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="Separate artifact directory; an existing run marker prevents reuse")
    parser.add_argument("--instruction", default="CloseFridge")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=MODEL, help="Pinned base model or saved local full checkpoint")
    parser.add_argument("--reference", type=Path, help="Paired baseline artifacts; verify reset before inference")
    args = parser.parse_args()
    OUT = args.output.resolve()
    checkpoint = args.checkpoint
    revision = MODEL_REV if checkpoint == MODEL else None
    assert Path(sys.prefix).name == "embodiedops-robocasa", sys.prefix
    OUT.mkdir(parents=True, exist_ok=True)
    # Exclusive marker prevents accidental launch of a second experiment over these results.
    with (OUT / "run-started.json").open("x", encoding="utf-8") as f:
        json.dump({"started_unix": time.time(), "seed": args.seed}, f)
    (OUT / "executed-script.py.txt").write_bytes(Path(__file__).read_bytes())
    import torch
    import av
    from huggingface_hub import hf_hub_download
    from lerobot.envs.robocasa import RoboCasaEnv, convert_action
    from lerobot.envs.configs import RoboCasaEnv as RoboCasaConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    class SingleEpisodeEnv(RoboCasaEnv):
        """LeRobot step conversion/formatting, with its terminal auto-reset omitted."""
        def reset(self, **kwargs):
            assert not getattr(self, "reset_done", False), "Second episode prohibited"
            result = super().reset(**kwargs)
            self.reset_done = True
            return result

        def step(self, action):
            assert action.ndim == 1
            raw, reward, done, truncated, info = self._env.step(convert_action(action))
            success = bool(info.get("success", False))
            info.update(task=self.task, done=done, is_success=success)
            if done or success:
                info["final_info"] = {"task": self.task, "done": bool(done), "is_success": success}
            return self._format_raw_obs(raw), reward, done or success, truncated, info

    baseline = json.loads((ROOT / "environment/phase1b-results.json").read_text(encoding="utf-8"))
    versions = {k: md.version(k) for k in baseline["versions"]}
    assert versions == baseline["versions"], "Dependency drift"
    manifest = {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "model": checkpoint, "model_revision": revision, "instruction": args.instruction, "backbone_revision": BACKBONE_REV,
                "simulator_sources": baseline["source_verification"], "versions": versions,
                "seed": args.seed, "python": sys.executable, "hf_cache": os.environ["HF_HOME"],
                "numba_cache": os.environ["NUMBA_CACHE_DIR"], "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    processor_hashes = {}
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json",
                 "policy_preprocessor_step_5_normalizer_processor.safetensors",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        path = (Path(hf_hub_download(MODEL, name, revision=MODEL_REV, local_files_only=True))
                if checkpoint == MODEL else Path(checkpoint) / name)
        processor_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name.endswith(".json") and (checkpoint == MODEL or name != "config.json"):
            assert json.loads(path.read_text()) == json.loads((ROOT / "outputs/phase1a" / name).read_text())
    manifest["checkpoint_file_hashes"] = processor_hashes
    emit("manifest", manifest)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, revision=revision)
    assert policy.model.vlm_with_expert.config._commit_hash == BACKBONE_REV
    cfg = policy.config
    for k, v in {"chunk_size": 50, "n_action_steps": 50, "num_steps": 10,
                 "max_state_dim": 32, "max_action_dim": 32, "use_amp": False}.items():
        assert getattr(cfg, k) == v, (k, getattr(cfg, k))
    assert tuple(cfg.input_features["observation.state"].shape) == (16,)
    assert tuple(cfg.output_features["action"].shape) == (12,)
    pre, post = make_pre_post_processors(cfg, pretrained_path=checkpoint, pretrained_revision=revision)
    assert pre.steps[0].get_config()["rename_map"] == {k: v[0] for k, v in CAMERA_MAP.items()}
    env_pre, env_post = make_env_pre_post_processors(RoboCasaConfig(), cfg)
    assert not env_pre.steps and not env_post.steps
    emit("policy", {"config": cfg.to_dict() if hasattr(cfg, "to_dict") else vars(cfg),
                    "parameter_dtypes": sorted({str(p.dtype) for p in policy.parameters()}),
                    "preprocessor": [{"class": type(s).__name__, "config": s.get_config()} for s in pre.steps],
                    "postprocessor": [{"class": type(s).__name__, "config": s.get_config()} for s in post.steps]})

    env = SingleEpisodeEnv("CloseFridge", camera_name=CAMERAS)
    video = None
    summary = {"inference_calls": 0, "simulator_steps": 0, "physics_steps": 0,
               "termination_reason": "initializing"}
    rollout_start = None
    initial = None
    obs = None
    latencies = []
    try:
        print("Initializing one CloseFridge environment; no timeout traceback timer", flush=True)
        np.random.seed(args.seed)
        env._ensure_env()
        np.random.seed(args.seed)
        obs, _ = env.reset(seed=args.seed)
        low = env._env.env
        initial = physical(low)
        assert env._max_episode_steps == 600 and low.control_freq == 20
        if args.seed == 42:
            assert np.allclose(obs["agent_pos"], baseline["reset_state"], atol=1e-7, rtol=0)
        old_reset = json.loads((ROOT / "outputs/phase1b/reset.json").read_text(encoding="utf-8"))
        robot = low.robots[0]
        assert serial(robot.composite_controller_config) == old_reset["controller_config"]
        assert serial(robot.composite_controller._action_split_indexes) == old_reset["action_splits"]
        assert type(robot.robot_model).__name__ == old_reset["robot_model"]
        for part, expected in old_reset["controllers"].items():
            controller = robot.composite_controller.part_controllers[part]
            assert type(controller).__name__ == expected["class"]
            for field in ("input_min", "input_max", "output_min", "output_max", "input_type",
                          "input_ref_frame", "kp", "kd", "impedance_mode", "actuator_min", "actuator_max"):
                if field in expected:
                    assert serial(getattr(controller, field)) == expected[field], (part, field)
        old_initial = json.loads((ROOT / "outputs/phase1b/controls.json").read_text(encoding="utf-8"))["neutral_steps"][0]["before"]
        for k in (("qpos", "qvel") if args.seed == 42 else ()):
            assert np.allclose(initial[k], old_initial[k], atol=1e-9, rtol=0), f"Phase 1B reset differs: {k}"
        environment_instruction = env.task_description
        env.task_description = environment_instruction if args.instruction == "environment" else args.instruction
        manifest["environment_instruction"] = environment_instruction
        manifest["resolved_instruction"] = env.task_description
        emit("manifest", manifest)
        emit("scene", {"ep_meta": low.get_ep_meta(), "robot": type(robot.robot_model).__name__,
                       "controller": robot.composite_controller_config})
        frames(obs, "reset")
        emit("reset", {"state": obs["agent_pos"], "task": env.task_description, "fridge": fridge(low),
                       "physical": initial, "hashes": {k: hashlib.sha256(initial[k].tobytes()).hexdigest() for k in ("qpos", "qvel")}})
        if args.reference:
            reference = json.loads((args.reference / "reset.json").read_text(encoding="utf-8"))
            assert reference["task"] == env.task_description, "Paired language mismatch"
            for k in ("qpos", "qvel"):
                assert np.array_equal(initial[k], reference["physical"][k]), f"Paired reset mismatch: {k}"
            assert np.array_equal(obs["agent_pos"], reference["state"]), "Paired observation mismatch"
            camera_errors = {}
            for cam in CAMERAS:
                difference = np.abs(np.asarray(Image.open(args.reference / f"reset-{cam}.png")).astype(float)
                                    - obs["pixels"][cam].astype(float))
                camera_errors[cam] = {"max_abs_u8": float(difference.max()), "mean_abs_u8": float(difference.mean()),
                                      "changed_channels": int(np.count_nonzero(difference))}
                # Same sparse one-level rasterization tolerance as phase1c_replay.py.
                assert difference.max() <= 1 and difference.mean() <= .001, f"Paired rendered camera mismatch: {cam}"
            if (args.reference / "scene.json").exists():
                assert canonical_scene(json.loads((OUT / "scene.json").read_text())) == canonical_scene(json.loads((args.reference / "scene.json").read_text()))
            else:
                assert args.seed == 42, "Paired scene metadata missing"
                assert canonical_scene(serial(low.get_ep_meta())) == canonical_scene(old_reset["ep_meta"]), "Original seed-42 metadata mismatch"
            emit("paired-reset-validation", {"passed": True, "reference": str(args.reference.resolve()),
                 "qpos_qvel_state_exact": True, "camera_errors": camera_errors,
                 "scene_metadata_compared": True,
                 "metadata_reference": "paired" if (args.reference / "scene.json").exists() else "Phase 1B seed42",
                 "unordered_metadata_field": "groups_containing_sampled_obj"})
        assert not low._check_success(), "Task already successful at reset"
        assert env.task_description == (environment_instruction if args.instruction == "environment" else args.instruction)

        def prepare(observation, first=False):
            assert set(observation["pixels"]) == set(CAMERAS)
            assert observation["agent_pos"].shape == (16,) and finite_tree(observation)
            bridged = preprocess_observation(observation)
            bridged["task"] = [env.task_description]
            before_processor = tensor_info(bridged)
            batch = pre(env_pre(bridged))
            # transition_to_batch always emits action=None for observation-only input.
            assert batch.get("action") is None, "Unexpected supplied action in inference input"
            assert tuple(batch["observation.state"].shape) == (1, 16)
            assert batch["observation.state"].dtype == torch.float32
            assert len(batch["task"]) == 1 and batch["task"][0].strip() == env.task_description.strip()
            tokens = batch["observation.language.tokens"]
            attention = batch["observation.language.attention_mask"]
            assert tuple(tokens.shape) == (1, 48) and tokens.dtype == torch.int64
            assert tuple(attention.shape) == (1, 48) and attention.dtype == torch.bool
            assert attention.any(), "Empty language attention mask"
            for key, _ in CAMERA_MAP.values():
                x = batch[key]
                assert tuple(x.shape) == (1, 3, 256, 256) and x.dtype == torch.float32
                assert x.min() >= 0 and x.max() <= 1
            assert all(torch.isfinite(v).all() for v in batch.values() if isinstance(v, torch.Tensor))
            if first:
                images, masks = policy.prepare_images(batch)
                state = policy.prepare_state(batch)
                assert tuple(state.shape) == (1, 32) and torch.isfinite(state).all()
                assert len(images) == 3
                assert all(tuple(x.shape) == (1, 3, 512, 512) and torch.isfinite(x).all() for x in images)
                emit("observation-contract", {"official_bridge": before_processor,
                     "saved_processor_output": tensor_info(batch), "model_images": tensor_info(images),
                     "model_image_masks": tensor_info(masks), "model_state": tensor_info(state),
                     "matches_phase1a": True})
            return batch

        # Instrument the official sampler/chunk method, returning their outputs unchanged.
        captured = {}
        old_sample, old_chunk = policy.model.sample_actions, policy._get_action_chunk

        def sample(*args, **kwargs):
            output = old_sample(*args, **kwargs)
            captured["internal"] = output.detach().cpu().clone()
            return output

        def chunk(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = old_chunk(*args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            summary["inference_calls"] += 1
            final = post(output.clone())
            assert tuple(captured["internal"].shape) == (1, 50, 32) and torch.isfinite(captured["internal"]).all()
            assert tuple(output.shape) == (1, 50, 12) and torch.isfinite(output).all()
            assert tuple(final.shape) == (1, 50, 12) and torch.isfinite(final).all()
            item = {"call": summary["inference_calls"], "before_step": summary["simulator_steps"] + 1,
                    "seconds": elapsed, "internal": tensor_info(captured["internal"]),
                    "normalized": tensor_info(output), "final": tensor_info(final),
                    "first_five": final[0, :5].tolist(), "per_dim_min": final.amin(dim=(0, 1)).tolist(),
                    "per_dim_max": final.amax(dim=(0, 1)).tolist(),
                    "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved": torch.cuda.max_memory_reserved()}
            emit(f"chunk-{summary['inference_calls']:02d}", item)
            if summary["inference_calls"] == 1:
                torch.save({"internal": captured["internal"], "normalized": output.detach().cpu(),
                            "final": final}, OUT / "first-chunk.pt")
            return output

        policy.model.sample_actions, policy._get_action_chunk = sample, chunk
        policy.reset()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        video = av.open(str(OUT / "rollout.mp4"), mode="w")
        stream = video.add_stream("libx264", rate=20)
        stream.width, stream.height, stream.pix_fmt = 768, 288, "yuv420p"

        def video_frame(observation, label):
            for packet in stream.encode(av.VideoFrame.from_image(sheet(observation, label))):
                video.mux(packet)

        video_frame(obs, "RESET")
        rollout_start = time.perf_counter()
        with (OUT / "steps.jsonl").open("w", encoding="utf-8") as log, torch.inference_mode():
            for index in range(1, env._max_episode_steps + 1):
                batch = prepare(obs, first=index == 1)
                queue_before = len(policy._queues["action"])
                calls_before = summary["inference_calls"]
                if index == 51:
                    frames(obs, "first-replan-input")
                selected = policy.select_action(batch)
                action = env_post({"action": post(selected)})["action"].cpu().numpy()[0]
                gate = action_gate(env, action)
                if index == 1:
                    emit("first-action-gate", {"action": action, **gate})
                obs, trace = traced_step(env, action, index)
                summary["simulator_steps"] = index
                summary["physics_steps"] += len(trace["substeps"])
                record = {"step": index, "time": trace["after"]["time"],
                          "inference": summary["inference_calls"] > calls_before,
                          "queue_before": queue_before, "queue_after": len(policy._queues["action"]),
                          "action": action, "action_min": float(action.min()), "action_max": float(action.max()),
                          "gate": gate, "physical": trace["after"], "returned_state": trace["state"],
                          "fridge": trace["fridge_after"], "reward": trace["reward"],
                          "terminated": trace["terminated"], "truncated": trace["truncated"],
                          "finite": trace["finite"], "guard_failures": trace["guard_failures"]}
                log.write(json.dumps(serial(record)) + "\n")
                log.flush()
                video_frame(obs, f"STEP {index}")
                if index == 1:
                    frames(obs, "first-action")
                    emit("first-action-result", {k: trace[k] for k in ("delta", "reward", "terminated", "truncated", "finite", "fridge_after", "guard_failures")})
                if index % 25 == 0:
                    print(f"Step {index}/600; inference calls {summary['inference_calls']}; fridge {trace['fridge_after']}", flush=True)
                if trace["guard_failures"]:
                    summary["termination_reason"] = "guard_failure: " + "; ".join(trace["guard_failures"])
                    break
                if trace["fridge_after"]["success"] or trace["terminated"] or trace["truncated"]:
                    summary["termination_reason"] = "success" if trace["fridge_after"]["success"] else "simulator_termination"
                    break
            else:
                summary["termination_reason"] = "registered_horizon_600"
    except Exception:
        summary["termination_reason"] = "exception"
        summary["exception"] = traceback.format_exc()
        print(summary["exception"], flush=True)
    finally:
        if rollout_start is not None:
            summary["wall_clock_rollout_seconds"] = time.perf_counter() - rollout_start
        if video is not None:
            for packet in stream.encode():
                video.mux(packet)
            video.close()
        if obs is not None:
            frames(obs, "final")
            summary["final_fridge"] = fridge(env._env.env)
            summary["final_physical"] = physical(env._env.env)
            summary["simulated_seconds"] = summary["final_physical"]["time"] - initial["time"]
        summary.update(inference_seconds=latencies, peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                       peak_cuda_reserved=torch.cuda.max_memory_reserved(),
                       gradients_present=any(p.grad is not None for p in policy.parameters()))
        emit("summary", summary)
        env.close()
    return 0 if summary["termination_reason"] in ("success", "registered_horizon_600") else 2


if __name__ == "__main__":
    raise SystemExit(main())
