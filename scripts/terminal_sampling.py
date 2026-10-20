"""Frozen reward-window sampling. Recorded success is not a contact/recovery label."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/terminal-window-audit-v1/final"
ORDER_HASH = "94bba1a70cda870c30d9cea3404af24ae2a6ee199c78d26713e36c30c518c5e0"
VALIDATION_HASH = "97172a32b8f264436f06b8d997bed66f3ffcf53bd0b5a3d782289e54484c289a"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def anchor_pool(reward, chunk=50):
    reward = np.asarray(reward)
    if not np.isfinite(reward).all() or not set(np.unique(reward)) <= {0, 1}:
        raise ValueError("Invalid binary reward annotations")
    positive = np.flatnonzero(reward)
    if not len(positive):
        return np.array([], dtype=int)
    tau = int(positive[0])
    if tau == 0 or not np.all(reward[tau:] == 1):
        return np.array([], dtype=int)
    return np.arange(max(0, tau-chunk+1), min(tau-1, len(reward)-chunk)+1)


def replace_anchors(original, groups, pools, seed=242):
    rng = np.random.default_rng(seed)
    plan = []
    for k, row in enumerate(original):
        ep, idx = row["episode"], row["index"]
        component = "archived_uniform"
        if k % 2 == 0:
            if len(pools[ep]):
                idx = int(rng.choice(groups[ep][pools[ep]]))
                component = "pre_success_full_chunk"
            else:
                component = "ineligible_keep_archived_uniform"
        frame = int(np.flatnonzero(groups[ep] == idx).item())
        plan.append(dict(update=k+1, index=idx, episode=ep, frame=frame,
                         sampling_component=component))
    return plan


def dataset_arrays(dataset):
    table = dataset.hf_dataset.with_format(None)
    data = {key: np.asarray(table[key]) for key in
            ("episode_index", "frame_index", "index", "timestamp", "next.reward",
             "next.done", "action", "observation.state")}
    assert data["action"].shape == (len(dataset), 12)
    assert data["observation.state"].shape == (len(dataset), 16)
    assert np.isfinite(data["action"]).all() and np.isfinite(data["observation.state"]).all()
    groups, pools = {}, {}
    for ep in sorted(dataset.episodes):
        ix = np.flatnonzero(data["episode_index"] == ep)
        assert np.array_equal(data["frame_index"][ix], np.arange(len(ix)))
        assert np.all(np.diff(data["index"][ix]) == 1)
        assert np.allclose(data["timestamp"][ix], np.arange(len(ix))/20, atol=2e-6)
        assert np.flatnonzero(data["next.done"][ix]).tolist() == [len(ix)-1]
        groups[int(ep)] = ix
        pools[int(ep)] = anchor_pool(data["next.reward"][ix])
    return data, groups, pools


def prepare(train, val, baseline_split):
    """Rebuild from raw training rows, then require exact preregistered manifests."""
    assert sha(AUDIT/"proposed-order-NOT-TRAINED.json") == ORDER_HASH
    assert sha(AUDIT/"terminal-validation-anchors.json") == VALIDATION_HASH
    assert list(train.episodes) == baseline_split["train_episodes"]
    assert list(val.episodes) == baseline_split["validation_episodes"]
    assert not set(train.episodes) & set(val.episodes)
    data, groups, pools = dataset_arrays(train)
    vd, vg, vp = dataset_arrays(val)
    original = baseline_split["training_order"]
    assert all(data["episode_index"][r["index"]] == r["episode"] for r in original)
    plan = replace_anchors(original, groups, pools)
    assert plan == read(AUDIT/"proposed-order-NOT-TRAINED.json")
    terminal = []
    for ep in sorted(vg):
        assert len(vp[ep])
        for frame in (int(vp[ep][0]), int(vp[ep][-1])):
            terminal.append(dict(episode=ep, frame=frame, noise_seed=92000+len(terminal)))
    assert terminal == read(AUDIT/"terminal-validation-anchors.json")
    terminal_plan = [dict(r, index=int(vg[r["episode"]][r["frame"]])) for r in terminal]
    counts = Counter(r["sampling_component"] for r in plan)
    assert counts == {"pre_success_full_chunk": 2438, "archived_uniform": 2500,
                      "ineligible_keep_archived_uniform": 62}

    def coverage(order):
        padded = valid = pool_hits = 0
        for r in order:
            ep, idx = r["episode"], r["index"]
            frame = int(data["frame_index"][idx])
            n = min(50, len(groups[ep])-frame)
            valid += n
            padded += int(n < 50)
            pool_hits += int(frame in pools[ep])
        return dict(updates=len(order), unique_anchors=len({r["index"] for r in order}),
                    padded_chunks=padded, valid_action_rows=valid, anchors_in_pool=pool_hits)

    evidence = dict(passed=True, episode_order_equal=True, train_episodes=len(groups),
                    validation_episodes=len(vg), disjoint=True,
                    eligible_train_episodes=sum(bool(len(p)) for p in pools.values()),
                    eligible_train_anchors=sum(len(p) for p in pools.values()),
                    components=dict(counts), baseline=coverage(original), candidate=coverage(plan),
                    episode_visits=dict(Counter(r["episode"] for r in plan)),
                    targeted_episode_visits=dict(Counter(r["episode"] for r in plan
                        if r["sampling_component"] == "pre_success_full_chunk")),
                    unique_targeted_anchors=len({r["index"] for r in plan
                        if r["sampling_component"] == "pre_success_full_chunk"}),
                    order_sha256=ORDER_HASH, terminal_validation_sha256=VALIDATION_HASH,
                    label_sources=["pinned demonstration next.reward/action/state/frame/timestamp",
                                   "archived training split and episode order"],
                    rollout_labels_used=False, contact_labels=False, recovery_labels=False,
                    timing_limit="Reader row alignment verified; converter's physical reward offset is not independently known")
    assert evidence["candidate"]["anchors_in_pool"] == 2566
    assert evidence["candidate"]["padded_chunks"] == 469
    return plan, terminal_plan, evidence


def verify_reader_alignment(train, val, plan, terminal_plan):
    """Check every chosen query/pad mask, plus real decoded boundary samples."""
    import torch
    checks, decoded = 0, []
    for name, dataset, order in (("train", train, plan), ("validation", val, terminal_plan)):
        data, groups, pools = dataset_arrays(dataset)
        assert dataset.reader.delta_indices["action"] == list(range(50))
        indices = sorted({r["index"] for r in order})
        examples = []
        for idx in indices:
            ep = int(data["episode_index"][idx])
            frame, absolute = int(data["frame_index"][idx]), int(data["index"][idx])
            end = int(data["index"][groups[ep][-1]])
            expected = np.minimum(np.arange(absolute, absolute+50), end)
            query, padding = dataset.reader._get_query_indices(absolute, ep)
            assert query["action"] == expected.tolist()
            assert padding["action_is_pad"].tolist() == (np.arange(absolute, absolute+50) > end).tolist()
            if frame in pools[ep]:
                tau = int(np.flatnonzero(data["next.reward"][groups[ep]])[0])
                assert frame < tau <= frame+49 and not padding["action_is_pad"].any()
                assert data["next.reward"][idx] == 0
            if len(examples) < 4 or (frame in pools[ep] and frame in (pools[ep][0], pools[ep][-1])):
                examples.append(idx)
            checks += 1
        # Cover full, padded and both terminal endpoints with actual __getitem__.
        chosen = list(dict.fromkeys(examples[:4] + examples[-4:]))
        padded = [i for i in indices if data["frame_index"][i]+50 > len(groups[int(data["episode_index"][i])])]
        if padded:
            chosen.append(padded[-1])
        for idx in dict.fromkeys(chosen):
            raw = dataset[idx]
            ep, frame = int(raw["episode_index"]), int(raw["frame_index"])
            relative = groups[ep][np.minimum(np.arange(frame, frame+50), len(groups[ep])-1)]
            assert torch.equal(raw["action"], torch.as_tensor(data["action"][relative], dtype=raw["action"].dtype))
            assert raw["action_is_pad"].tolist() == (np.arange(frame, frame+50) >= len(groups[ep])).tolist()
            assert np.isclose(float(raw["timestamp"]), frame/20, atol=2e-6)
            cameras = {}
            for cam in dataset.meta.camera_keys:
                im = raw[cam]
                assert torch.isfinite(im).all() and torch.count_nonzero(im) > 0
                cameras[cam] = dict(shape=list(im.shape), dtype=str(im.dtype),
                                    nonzero=int(torch.count_nonzero(im)))
            decoded.append(dict(split=name, index=idx, episode=ep, frame=frame,
                                valid_actions=int((~raw["action_is_pad"]).sum()), cameras=cameras))
    return dict(passed=True, selected_query_masks_checked=checks, decoded_samples=decoded,
                action_offsets=list(range(50)), reward_is_offline_index_only=True)
