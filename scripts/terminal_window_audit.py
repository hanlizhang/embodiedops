"""Read-only reward/action/state/video audit. No policy, optimizer or simulator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/phase1a/hf-cache/lerobot/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots/b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4"
BASELINE = ROOT / "outputs/agent-workflow-v1/learning-5000-v1/episode_balanced"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def quantiles(values):
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    np.quantile(values, [0, .25, .5, .75, 1]).tolist()))


def motion(a, s, lo, hi):
    """Inclusive action interval; last action omitted from achieved-motion pair."""
    action = a[lo:hi+1]
    end = min(hi+1, len(s)-1)
    dp = np.diff(s[lo:end+1, 7:10], axis=0)
    q = s[lo:end+1, 10:14].copy()
    q /= np.linalg.norm(q, axis=1)[:, None]
    da = 2*np.arccos(np.clip(np.abs((q[1:]*q[:-1]).sum(axis=1)), 0, 1))
    quiet = (np.linalg.norm(action[:, 5:8], axis=1) < .05) & (
        np.linalg.norm(action[:, 8:11], axis=1) < .05)
    return dict(frames=[lo, hi], actions=len(action),
                mean_requested_translation_mm=float(np.linalg.norm(
                    np.clip(action[:, 5:8], -1, 1)*.05, axis=1).mean()*1000),
                achieved_eef_base_relative_path_m=float(np.linalg.norm(dp, axis=1).sum()),
                mean_achieved_translation_mm=float(np.linalg.norm(dp, axis=1).mean()*1000) if len(dp) else None,
                eef_base_relative_orientation_path_deg=float(np.rad2deg(da).sum()),
                quiet_action_fraction=float(quiet.mean()),
                close_gripper_commands=int((action[:, 11] >= .5).sum()),
                desired_pose_mode_commands=int((action[:, 4] >= .5).sum()),
                gripper_qpos_range=[s[lo:hi+1, 14:16].min(axis=0).tolist(),
                                   s[lo:hi+1, 14:16].max(axis=0).tolist()])


def extract_frame(container, stream, timestamp):
    container.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
    closest = None
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        delta = abs(t - timestamp)
        if closest is None or delta < closest[0]:
            closest = (delta, frame, t)
        if t >= timestamp:
            break
    assert closest is not None and closest[0] < .002, (timestamp, closest[0] if closest else None)
    return closest[1].to_image(), closest[2], closest[0]


def video_evidence(out, meta, episode_stats, selected):
    import av
    from PIL import Image, ImageDraw, ImageFont
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    evidence = []
    for ep in selected:
        stats = episode_stats[ep]
        tau = stats["first_positive_reward_frame"]
        # Fixed landmarks before viewing. Nonmonotone cases use reward boundaries.
        if stats["stable_success_tail"]:
            frames = sorted(set([0, tau//3, 2*tau//3, max(0, tau-100),
                                 max(0, tau-50), max(0, tau-25), tau, stats["length"]-1]))
        else:
            frames = sorted(set([max(0, tau-25)] +
                [r["first"] for r in stats["reward_runs"][1:]] +
                [r["last"] for r in stats["reward_runs"][1:]]))
        canvas = Image.new("RGB", (len(frames)*256, 3*256+58), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f"Demo {ep}, {stats['task']}; first next.reward=1 at frame {tau}. RGB only; no contact labels.", fill="black", font=font)
        m = meta.loc[ep]
        for row, cam in enumerate(cameras):
            key = f"videos/observation.images.{cam}"
            path = DATA / f"videos/observation.images.{cam}/chunk-{m[key+'/chunk_index']:03d}/file-{m[key+'/file_index']:03d}.mp4"
            start = float(m[key+"/from_timestamp"])
            with av.open(str(path)) as c:
                stream = c.streams.video[0]
                for col, frame in enumerate(frames):
                    image, pts, error = extract_frame(c, stream, start+frame/20)
                    pixels = np.asarray(image)
                    assert pixels.shape == (256, 256, 3) and pixels.max() > 0
                    canvas.paste(image, (col*256, 58+row*256))
                    if row == 0:
                        reward = next(r["reward"] for r in stats["reward_runs"] if r["first"] <= frame <= r["last"])
                        draw.text((col*256+5, 29), f"f{frame}, {frame/20:.2f}s, r={reward}", fill="black", font=font)
                    evidence.append(dict(episode=ep, camera=cam, frame=frame,
                                         file=str(path.relative_to(ROOT)), video_pts_seconds=pts,
                                         requested_seconds=start+frame/20, timestamp_error_seconds=error,
                                         nonzero_channels=int(np.count_nonzero(pixels))))
        canvas.save(out/f"demo-{ep}-reward-windows.jpg", quality=90)
    return evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT/"outputs")
    out.mkdir(parents=True, exist_ok=False)
    parquet = DATA/"data/chunk-000/file-000.parquet"
    metadata = DATA/"meta/episodes/chunk-000/file-000.parquet"
    df = pq.read_table(parquet).to_pandas()
    meta = pq.read_table(metadata).to_pandas().set_index("episode_index")
    split = read(BASELINE/"split.json")
    train, val = set(split["train_episodes"]), set(split["validation_episodes"])
    assert not train & val and train | val == set(range(513))
    assert len(df) == 155443 and read(DATA/"meta/info.json")["fps"] == 20
    eps = {}
    for ep, f in df.groupby("episode_index", sort=True):
        ep = int(ep)
        assert np.array_equal(f.frame_index, np.arange(len(f)))
        assert np.allclose(f.timestamp, np.arange(len(f))/20, atol=2e-6)
        a, s = np.stack(f.action), np.stack(f["observation.state"])
        assert a.shape == (len(f), 12) and s.shape == (len(f), 16)
        assert np.isfinite(a).all() and np.isfinite(s).all()
        reward = np.asarray(f["next.reward"])
        done = np.flatnonzero(np.asarray(f["next.done"]))
        assert set(np.unique(reward)) <= {0, 1} and done.tolist() == [len(f)-1]
        positive = np.flatnonzero(reward > 0)
        tau = int(positive[0]) if len(positive) else None
        monotone = tau is not None and bool((reward[tau:] == 1).all())
        cuts = np.r_[0, np.flatnonzero(np.diff(reward)) + 1, len(f)]
        reward_runs = [dict(first=int(x), last=int(y-1), reward=int(reward[x]))
                       for x, y in zip(cuts[:-1], cuts[1:])]
        # Only full 50-action chunks which start before and contain first reward.
        lo = max(0, tau-49) if tau is not None else None
        hi = min(tau-1, len(f)-50) if tau is not None else None
        eligible = bool(monotone and tau > 0 and hi >= lo)
        eps[ep] = dict(episode=ep, split="train" if ep in train else "validation",
            task=str(meta.loc[ep, "tasks"][0]), length=len(f),
            first_positive_reward_frame=tau, positive_frames=len(positive),
            reward_rises=int(np.count_nonzero(np.diff(reward, prepend=0) > 0)),
            reward_falls=int(np.count_nonzero(np.diff(reward) < 0)),
            reward_runs=reward_runs,
            stable_success_tail=monotone, terminal_pool_eligible=eligible,
            terminal_anchor_frames=[lo, hi] if eligible else None,
            terminal_anchor_count=hi-lo+1 if eligible else 0,
            pre_success=motion(a, s, max(0, tau-100), tau-1) if tau and tau > 0 else None,
            final_50_pre_success=motion(a, s, max(0, tau-50), tau-1) if tau and tau > 0 else None,
            post_success=motion(a, s, tau, len(f)-1) if tau is not None else None)
    tr = df[df.episode_index.isin(train)].reset_index(drop=True)
    assert len(tr) == split["train_frames"]
    order = split["training_order"]
    assert all(int(tr.iloc[r["index"]].episode_index) == r["episode"] for r in order)
    coverage = Counter()
    for r in order:
        ep, t = r["episode"], int(tr.iloc[r["index"]].frame_index)
        e = eps[ep]
        tau, n = e["first_positive_reward_frame"], e["length"]
        if tau is not None:
            coverage["anchor_last_100_pre_reward"] += int(max(0, tau-100) <= t < tau)
            coverage["anchor_post_success"] += int(t >= tau)
            coverage["chunk_contains_first_reward_transition"] += int(t <= tau <= t+49)
        coverage["chunk_requires_padding"] += int(t+49 >= n)
        if e["terminal_pool_eligible"]:
            lo, hi = e["terminal_anchor_frames"]
            coverage["anchor_in_proposed_pool"] += int(lo <= t <= hi)
    # Proposal-only indices: no training interface is changed or invoked.
    groups = {int(ep): np.flatnonzero(tr.episode_index.to_numpy() == ep) for ep in train}
    rng = np.random.default_rng(242)
    proposed, proposed_counts = [], Counter()
    for k, r in enumerate(order):
        ep, index, component = r["episode"], r["index"], "archived_uniform"
        if k % 2 == 0:
            if eps[ep]["terminal_pool_eligible"]:
                lo, hi = eps[ep]["terminal_anchor_frames"]
                index, component = int(rng.choice(groups[ep][lo:hi+1])), "pre_success_full_chunk"
            else:
                component = "ineligible_keep_archived_uniform"
        t = int(tr.iloc[index].frame_index)
        e = eps[ep]
        if e["terminal_pool_eligible"]:
            lo, hi = e["terminal_anchor_frames"]
            proposed_counts["anchor_in_proposed_pool"] += int(lo <= t <= hi)
        proposed_counts["chunk_requires_padding"] += int(t+49 >= e["length"])
        proposed_counts[component] += 1
        proposed.append(dict(update=k+1, index=index, episode=ep, frame=t, sampling_component=component))
    assert [r["episode"] for r in proposed] == [r["episode"] for r in order]
    terminal_validation = []
    for ep in sorted(val):
        assert eps[ep]["terminal_pool_eligible"]
        for t in eps[ep]["terminal_anchor_frames"]:
            terminal_validation.append(dict(episode=ep, frame=t, noise_seed=92000+len(terminal_validation)))
    # Visual sample uses TRAIN split only: short, median and long eligible episode
    # within each instruction stratum, ties resolved by episode ID, before viewing.
    selected = []
    selection = []
    for task in sorted({eps[e]["task"] for e in train}):
        eligible = sorted((eps[e] for e in train if eps[e]["task"] == task and eps[e]["terminal_pool_eligible"]),
                          key=lambda e: (e["length"], e["episode"]))
        for q in (.25, .5, .75):
            e = eligible[round((len(eligible)-1)*q)]
            selected.append(e["episode"])
            selection.append(dict(task=task, length_quantile=q, episode=e["episode"], length=e["length"]))
    # Previously reviewed examples retained as a separate, explicitly nonrandom set.
    anomalies = sorted((r for r in eps.values() if not r["stable_success_tail"]),
                       key=lambda r: (-max(z["last"]-z["first"]+1 for z in r["reward_runs"][1:] if z["reward"] == 0),
                                      r["episode"]))[:2]
    visual_ids = list(dict.fromkeys(selected+[0, 437, 5]+[r["episode"] for r in anomalies]))
    video = video_evidence(out, meta, eps, visual_ids)
    summary = dict(dataset_revision=DATA.name, frames=len(df), episodes=len(eps),
        train_episodes=len(train), validation_episodes=len(val),
        reward_counts={str(k): int(v) for k, v in Counter(df["next.reward"]).items()},
        no_positive_reward=[e for e, r in eps.items() if r["first_positive_reward_frame"] is None],
        nonmonotone_reward=[e for e, r in eps.items() if not r["stable_success_tail"]],
        positive_tail_lengths=dict(Counter(r["positive_frames"] for r in eps.values())),
        train_eligible=sum(eps[e]["terminal_pool_eligible"] for e in train),
        validation_eligible=sum(eps[e]["terminal_pool_eligible"] for e in val),
        eligible_train_anchor_count=sum(eps[e]["terminal_anchor_count"] for e in train),
        train_pool_sizes=quantiles([eps[e]["terminal_anchor_count"] for e in train]),
        train_episode_lengths=quantiles([eps[e]["length"] for e in train]),
        original_5000_order_coverage=dict(coverage),
        proposed_order_status="NOT_TRAINED; audit-only exact sampling proposal",
        proposed_order_rule="same archived episode sequence; zero-based even updates choose uniformly from terminal pool with RNG242; others keep archived row; ineligible fallback unchanged",
        proposed_order_coverage=dict(proposed_counts),
        eligible_anchor_rule="max(0,first_reward-49) <= frame <= min(first_reward-1,length-50); reward must stay positive through episode end",
        pre_success_motion={k: quantiles([eps[e]["pre_success"][k] for e in train if eps[e]["pre_success"]])
            for k in ["mean_requested_translation_mm", "mean_achieved_translation_mm",
                      "achieved_eef_base_relative_path_m", "quiet_action_fraction"]},
        visual_selection_rule="training-only per instruction, length quantiles .25/.5/.75; ties episode ID",
        visual_selection=selection, prior_examples=[0, 437, 5],
        reward_anomaly_visual_selection=[r["episode"] for r in anomalies],
        annotations_are_contacts=False, recovery_demonstrations_verified=False,
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [
            parquet, metadata, DATA/"meta/info.json", BASELINE/"split.json", Path(__file__)]})
    for name, obj in [("summary.json", summary), ("episodes.json", eps), ("video-frames.json", video),
                      ("proposed-order-NOT-TRAINED.json", proposed),
                      ("terminal-validation-anchors.json", terminal_validation)]:
        (out/name).write_text(json.dumps(obj, indent=2)+"\n", encoding="utf-8")
    (out/"executed-audit.py.txt").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
