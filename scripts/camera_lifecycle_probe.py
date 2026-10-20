"""Bounded saved-action camera-path probe; no policy inference or weight changes."""
import argparse
import ctypes
import gc
import json
import time
from pathlib import Path

import phase1d_rollout as rollout
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--bind", action="store_true")
    ap.add_argument("--safe-free", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rollout.OUT = args.output
    import glfw
    from robosuite.utils.binding_utils import MjRenderContext
    from lerobot.envs.robocasa import RoboCasaEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.factory import make_pre_post_processors

    def pointer(p):
        return ctypes.cast(p, ctypes.c_void_p).value if p else None

    phase = "construction"
    events = []
    low = None

    def record(stage, arrays=None, **extra):
        stats = {}
        for key, value in (arrays or {}).items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray) and value.ndim >= 3:
                stats[key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                  min=float(value.min()), max=float(value.max()),
                                  nonzero_channels=int(np.count_nonzero(value)),
                                  nonzero_pixels=int(np.count_nonzero(np.any(
                                      value != 0, axis=-1 if value.shape[-1] == 3 else -3))))
        events.append(dict(stage=stage, phase=phase, unix=time.time(),
                           simulation_time=(float(low.sim.data.time)
                                            if low is not None and hasattr(low.sim, "data") else None),
                           current_context=pointer(glfw.get_current_context()), images=stats, **extra))

    original_render = MjRenderContext.render
    original_read = MjRenderContext.read_pixels
    original_free = MjRenderContext.__del__

    def free(ctx):
        previous = glfw.get_current_context()
        target = pointer(ctx.gl_ctx._context)
        record("renderer_free_before", target_context=target)
        if args.safe_free:
            from renderer_lifecycle import dispose_in_owner_context
            dispose_in_owner_context(ctx, original_free, glfw)
        else:
            original_free(ctx)
        record("renderer_free_after", freed_context=target)

    def render(ctx, *a, **kw):
        record("renderer_before", target_context=pointer(ctx.gl_ctx._context),
               camera_id=kw.get("camera_id"))
        if args.bind:
            ctx.gl_ctx.make_current()
        return original_render(ctx, *a, **kw)

    def read(ctx, *a, **kw):
        result = original_read(ctx, *a, **kw)
        record("mujoco_read_pixels", {"rgb": result if isinstance(result, np.ndarray) else result[0]},
               target_context=pointer(ctx.gl_ctx._context))
        return result

    MjRenderContext.render, MjRenderContext.read_pixels = render, read
    MjRenderContext.__del__ = free
    pre, _ = make_pre_post_processors(
        SmolVLAConfig.from_pretrained(rollout.MODEL, revision=rollout.MODEL_REV),
        pretrained_path=rollout.MODEL, pretrained_revision=rollout.MODEL_REV)
    env = RoboCasaEnv("CloseFridge", camera_name=rollout.CAMERAS)
    try:
        np.random.seed(46)
        env._ensure_env()
        low = env._env.env
        raw_get = low._get_observations
        gym_get = env._env.get_observation
        format_get = env._format_raw_obs

        def raw(*a, **kw):
            value = raw_get(*a, **kw)
            record("robocasa_raw", value)
            return value

        def gym(value):
            result = gym_get(value)
            record("robocasa_gym", result)
            return result

        def formatted(value):
            result = format_get(value)
            record("lerobot_format", result["pixels"])
            return result

        low._get_observations = raw
        env._env.get_observation = gym
        env._format_raw_obs = formatted
        phase = "reset"
        np.random.seed(46)
        obs, _ = env.reset(seed=46)
        reference = rollout.ROOT / "outputs/agent-workflow-v1/learning-5000-v1/diagnostic-46-episode_balanced"
        reset = json.loads((reference / "reset.json").read_text())
        assert np.array_equal(low.sim.data.qpos, reset["physical"]["qpos"])
        for cam in rollout.CAMERAS:
            difference = np.abs(obs["pixels"][cam].astype(float) -
                                np.asarray(rollout.Image.open(reference / f"reset-{cam}.png")).astype(float))
            record("reset_comparison", camera=cam, max_difference=float(difference.max()),
                   mean_difference=float(difference.mean()))
            assert difference.max() <= 1 and difference.mean() <= .001

        def processed(obs):
            bridge = preprocess_observation(obs)
            bridge["task"] = ["CloseFridge"]
            record("lerobot_bridge", bridge)
            record("policy_preprocessing", pre(bridge))

        processed(obs)
        phase = "before_first_step"
        if args.collect:
            record("before_gc")
            collected = gc.collect()
            record("after_gc", collected=collected)
        action = np.asarray(json.loads((reference / "first-action-trace.json").read_text())["raw_action"], dtype=np.float32)
        result = rollout.traced_step(env, action, 1)
        # traced_step returns observation, trace and termination details; log already
        # captures every raw and formatted observation on the path.
        phase = "after_first_step"
        obs = env._format_raw_obs(env._env.get_observation(low._get_observations(force_update=True)))
        processed(obs)
        rollout.frames(obs, "after-step")
        record("complete", warnings=low.sim.data._data.warning.number.tolist(),
               qpos_finite=bool(np.isfinite(low.sim.data.qpos).all()),
               saved_first_step_qpos_exact=np.array_equal(
                   low.sim.data.qpos, json.loads((reference / "first-action-trace.json").read_text())["after"]["qpos"]),
               cameras={c: int(low.sim.model.camera_name2id(c)) for c in rollout.CAMERAS})
    finally:
        (args.output / "trace.json").write_text(json.dumps(events, indent=2))
        env.close()
        MjRenderContext.render, MjRenderContext.read_pixels = original_render, original_read
        MjRenderContext.__del__ = original_free
    print(json.dumps(dict(events=len(events), black_reads=sum(
        e["stage"] == "mujoco_read_pixels" and e["images"]["rgb"]["max"] == 0 for e in events))))


if __name__ == "__main__":
    main()
