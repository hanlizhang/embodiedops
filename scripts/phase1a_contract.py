"""Read real data and run forward-only SmolVLA checks; never create an environment.

Run in embodiedops-robocasa. Artifacts go to ignored outputs/phase1a/.
No simulator imports, optimizer, backward pass, or configuration/precision edits.
"""

import argparse
import copy
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

MODEL = "lerobot/smolvla_robocasa"
MODEL_REV = "f7af588d11e30b0069f200792fb285904b74c80b"
DATASET = "pepijn223/robocasa_CloseFridge"
DATASET_REV = "b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4"
BACKBONE_REV = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
CAMERAS = {
    "observation.images.robot0_agentview_left": ("observation.images.camera1", "LEFT"),
    "observation.images.robot0_agentview_right": ("observation.images.camera2", "RIGHT"),
    "observation.images.robot0_eye_in_hand": ("observation.images.camera3", "WRIST"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase1a"))
    parser.add_argument("--execute-forward", action="store_true",
                        help="Run only after inspecting the saved camera sheet from the initial invocation")
    args = parser.parse_args()
    if Path(sys.prefix).name != "embodiedops-robocasa":
        raise RuntimeError(f"Wrong environment: {sys.prefix}")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from huggingface_hub import hf_hub_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
    from PIL import Image, ImageDraw

    records = []

    def summarize(value):
        if isinstance(value, torch.Tensor):
            x = value.detach().cpu()
            result = {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(value.device)}
            if x.numel():
                result.update(min=x.min().item(), max=x.max().item(),
                              first=x.flatten()[:16].tolist(), finite=bool(torch.isfinite(x).all()))
            return result
        if isinstance(value, dict):
            return {str(k): summarize(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [summarize(v) for v in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def emit(name, value):
        record = {"stage": name, "value": summarize(value)}
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "trace.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    def memory():
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "free_total_bytes": list(torch.cuda.mem_get_info())}

    emit("manifest", {"python": sys.version, "prefix": sys.prefix, "platform": platform.platform(),
                      "model": MODEL, "model_revision": MODEL_REV,
                      "dataset": DATASET, "dataset_revision": DATASET_REV,
                      "backbone_revision": BACKBONE_REV, "seed": 42})
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    (args.output / "packages.txt").write_text("\n".join(f"{n}=={v}" for n, v in packages) + "\n")
    emit("packages", dict(packages))
    installed = {n.lower().replace("_", "-") for n, _ in packages}
    if installed & {"opencv-python", "robocasa", "robosuite"}:
        raise RuntimeError("Unexpected OpenCV provider or simulator package in isolated environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently substitute a CPU run")
    emit("gpu", {"name": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
                 "total_bytes": torch.cuda.get_device_properties(0).total_memory})

    saved = {}
    for name in ["config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        path = hf_hub_download(MODEL, name, revision=MODEL_REV)
        saved[name] = json.loads(Path(path).read_text())
        (args.output / name).write_text(json.dumps(saved[name], indent=2))
    emit("saved_checkpoint_json", saved)
    meta = LeRobotDatasetMetadata(DATASET, revision=DATASET_REV)
    emit("dataset_features", meta.features)
    emit("task_table", meta.tasks.reset_index().to_dict(orient="records"))
    emit("metadata", {"fps": meta.fps, "episodes": meta.total_episodes, "frames": meta.total_frames,
                      "root": str(meta.root), "revision": meta.revision})

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    phase = "checkpoint_loading"
    try:
        policy = SmolVLAPolicy.from_pretrained(MODEL, revision=MODEL_REV)
        emit("loaded_policy", {"input_features": policy.config.input_features,
             "output_features": policy.config.output_features,
             "state_feature_shape": policy.config.input_features["observation.state"].shape,
             "action_feature_shape": policy.config.output_features["action"].shape,
             **{k: getattr(policy.config, k) for k in ["chunk_size", "n_action_steps", "max_state_dim",
                "max_action_dim", "num_steps", "device", "use_amp", "load_vlm_weights"]}})
        emit("parameter_dtypes", sorted({str(p.dtype) for p in policy.parameters()}))
        backbone_commit = getattr(policy.model.vlm_with_expert.config, "_commit_hash", None)
        emit("backbone_commit", backbone_commit)
        if backbone_commit != BACKBONE_REV:
            raise RuntimeError(f"Backbone revision drift: {backbone_commit}; expected {BACKBONE_REV}")
        pre, post = make_pre_post_processors(policy.config, pretrained_path=MODEL, pretrained_revision=MODEL_REV)
        for label, pipeline in [("preprocessor", pre), ("postprocessor", post)]:
            emit(label, [{"class": type(step).__name__, "config": step.get_config(),
                 "state": step.state_dict()} for step in pipeline.steps])
        mismatches = []
        for key in ["observation.state", "action"]:
            features = policy.config.input_features if key.startswith("observation") else policy.config.output_features
            actual = tuple(features[key].shape)
            measured = tuple(meta.features[key]["shape"])
            if actual != measured:
                mismatches.append(f"policy {key}: {actual} versus dataset {measured}")
            for pipeline in [pre, post]:
                for step in pipeline.steps:
                    if key in getattr(step, "features", {}):
                        shape = tuple(step.features[key].shape)
                        if shape != measured:
                            mismatches.append(f"{type(step).__name__} {key}: {shape} versus dataset {measured}")
                    for stat_name, tensor in getattr(step, "_tensor_stats", {}).get(key, {}).items():
                        # Counts are scalar metadata, not one normalization value per feature.
                        if stat_name == "count":
                            continue
                        if tensor.numel() != measured[0]:
                            mismatches.append(f"{type(step).__name__} {key}.{stat_name}: {tuple(tensor.shape)}")
        candidate = {key: value[0] for key, value in CAMERAS.items()}
        actual_map = pre.steps[0].get_config().get("rename_map")
        if actual_map != candidate:
            mismatches.append(f"Saved camera mapping differs from candidate: {actual_map}")
        emit("contract_gate", {"mismatches": mismatches, "camera_mapping": actual_map})
        if mismatches:
            emit("STOP", "Contract mismatch: no preprocessing, model forward, or inference executed")
            return 2

        phase = "data_loading"
        dataset = LeRobotDataset(DATASET, episodes=[0], revision=DATASET_REV, video_backend="pyav",
            delta_timestamps={"action": [i / meta.fps for i in range(policy.config.chunk_size)]})
        sample = dataset[0]
        emit("sample_keys", sorted(sample))
        emit("raw_sample", sample)
        annotations = {}
        task_lookup = dict(zip(meta.tasks.task_index, meta.tasks.index))
        for key in ["task_index", "annotation.human.task_description", "annotation.human.task_name"]:
            if key in sample:
                index = int(sample[key].item())
                annotations[key] = {"index": index, "decoded": task_lookup.get(index)}
        emit("decoded_annotations", annotations)
        description = annotations.get("annotation.human.task_description", {}).get("decoded")
        if not isinstance(description, str):
            raise RuntimeError("Cannot decode the actual human task description")
        # Explicit language selection; preserve and report the loader's default task too.
        sample["task"] = description
        emit("selected_language", sample["task"])
        sheet = Image.new("RGB", (3 * 420, 330), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (raw, (processed, role)) in enumerate(CAMERAS.items()):
            pixels = (sample[raw].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8")
            image = Image.fromarray(pixels)
            sheet.paste(image, (column * 420 + 82, 66))
            draw.text((column * 420 + 8, 6), f"{role}\n{raw}\n-> {processed}", fill="black")
        sheet.save(args.output / "cameras.png")
        emit("camera_contact_sheet", str((args.output / "cameras.png").resolve()))
        # Use the real DataLoader collation rule: the saved single-observation
        # batching step does not add a batch axis to a [T,A] demonstration chunk.
        collated = torch.utils.data.default_collate([sample])
        emit("dataloader_batch_one", collated)
        for i, transition in enumerate(pre.step_through(copy.deepcopy(collated))):
            emit(f"preprocessing_boundary_{i}", transition)
        batch = pre(copy.deepcopy(collated))
        emit("processed_batch", batch)
        images, img_masks = policy.prepare_images(batch)
        for i, key in enumerate(policy.config.image_features):
            image = batch[key]
            resized = resize_with_pad(image, policy.config.resize_imgs_with_padding[1],
                                      policy.config.resize_imgs_with_padding[0], pad_value=0)
            emit(f"resize_boundary_{key}", resized)
            torch.testing.assert_close(images[i], resized * 2 - 1)
        emit("model_prepared_images", images)
        state = policy.prepare_state(batch)
        actions = policy.prepare_action(batch)
        emit("internal_state", state)
        emit("internal_demonstration_actions", actions)
        mask = ~batch["action_is_pad"]
        emit("valid_action_mask", mask)
        tail = dataset[len(dataset) - 1]
        emit("episode_tail", {k: tail[k] for k in ["episode_index", "frame_index", "action", "action_is_pad"]})
        if not args.execute_forward:
            emit("STOP", "Contract/data trace complete. Inspect cameras.png before --execute-forward.")
            return 0

        phase = "flow_matching_forward"
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)
        noise = policy.model.sample_noise(actions.shape, actions.device)
        t = policy.model.sample_time(actions.shape[0], actions.device)
        xt = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        ut = noise - actions
        emit("flow_inputs", {"demonstration_chunk": batch["action"], "noise": noise,
                            "t": t, "x_t": xt, "u_t": ut})
        captured = {}

        def capture_input(module, inputs):
            captured["actual_x_t"] = inputs[0].detach().clone()

        def capture_velocity(module, inputs, output):
            captured["predicted_velocity"] = output.detach().clone()

        handles = [policy.model.action_in_proj.register_forward_pre_hook(capture_input),
                   policy.model.action_out_proj.register_forward_hook(capture_velocity)]
        try:
            with torch.no_grad():
                loss, loss_info = policy(batch, noise=noise, time=t)
        finally:
            for handle in handles:
                handle.remove()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured["actual_x_t"], xt)
        expected = ((captured["predicted_velocity"][..., :batch["action"].shape[-1]] -
                     ut[..., :batch["action"].shape[-1]]) ** 2 * mask.unsqueeze(-1)).sum()
        expected /= mask.sum().clamp_min(1) * batch["action"].shape[-1]
        torch.testing.assert_close(loss, expected)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite forward loss")
        emit("flow_result", {**captured, "loss": loss, "loss_info": loss_info, "memory": memory(),
                             "gradients_present": any(p.grad is not None for p in policy.parameters())})
        torch.save({"noise": noise.cpu(), "t": t.cpu(), "x_t": xt.cpu(), "u_t": ut.cpu(),
                    **{k: v.cpu() for k, v in captured.items()}}, args.output / "flow_tensors.pt")
        del captured, xt, ut, expected, loss

        phase = "static_inference"
        policy.reset()
        observation = {k: v for k, v in batch.items() if k not in ["action", "action_is_pad"]}
        emit("inference_observation", observation)
        torch.manual_seed(42)
        inference_noise = policy.model.sample_noise(actions.shape, actions.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            public_normalized = policy.predict_action_chunk(observation, noise=inference_noise.clone())
            final = post(public_normalized)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not torch.isfinite(final).all():
            raise RuntimeError("Nonfinite generated action chunk")
        emit("static_inference", {"public_normalized": public_normalized, "final": final,
            "first_three_actions": final[0, :3].tolist(), "seconds": elapsed, "memory": memory()})
        # Independently expose the internal padded sampler output using identical fixed noise.
        with torch.no_grad():
            internal = policy.model.sample_actions(images, img_masks, batch[OBS_LANGUAGE_TOKENS],
                batch[OBS_LANGUAGE_ATTENTION_MASK], state, noise=inference_noise.clone())
        torch.testing.assert_close(internal[..., :public_normalized.shape[-1]], public_normalized)
        emit("internal_generated_actions", internal)
        torch.save({"internal": internal.cpu(), "public_normalized": public_normalized.cpu(),
                    "final": final.cpu()}, args.output / "inference_tensors.pt")
        if tuple(final.shape) != (1, policy.config.chunk_size, policy.config.output_features["action"].shape[0]):
            raise RuntimeError("Unexpected public action chunk shape")
        emit("STOP", "Phase 1A complete. No simulator, backward, optimizer, or training executed.")
        return 0
    except torch.cuda.OutOfMemoryError:
        emit("OOM_STOP", {"phase": phase, "memory": memory()})
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
