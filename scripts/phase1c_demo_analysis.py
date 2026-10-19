"""Offline inspection of cached demonstrations and recorded policy outputs.

No simulator, policy, Torch, training, or network imports. Requires existing
NumPy, SciPy, PyArrow, PyAV and Pillow. All derived artifacts use a separate folder.
"""
from pathlib import Path
import argparse
import hashlib
import json

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from safetensors.numpy import load_file
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
REV = "b4b32587d3fd89f7c7121e62ff55c1c8f06f5aa4"
DATA = ROOT / "outputs/phase1a/hf-cache/lerobot/hub/datasets--pepijn223--robocasa_CloseFridge/snapshots" / REV
RUN = ROOT / "outputs/phase1c/retry-20261018"
OUT = ROOT / "outputs/phase1c/demo-analysis"
CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
STATE_NAMES = ["base_x", "base_y", "base_z", "base_qx", "base_qy", "base_qz", "base_qw",
               "eef_rel_x", "eef_rel_y", "eef_rel_z", "eef_rel_qx", "eef_rel_qy", "eef_rel_qz", "eef_rel_qw", "finger1", "finger2"]


def plain(x):
    if isinstance(x, (np.ndarray, np.generic)):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def save(name, obj):
    (OUT / name).write_text(json.dumps(plain(obj), indent=2), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runs(mask):
    """Inclusive start/end indices for True runs."""
    ix = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [[int(a), int(b - 1)] for a, b in ix.reshape(-1, 2)]


def stats(a):
    return {"min": a.min(0), "p05": np.quantile(a, .05, axis=0), "median": np.median(a, axis=0),
            "p95": np.quantile(a, .95, axis=0), "max": a.max(0), "mean": a.mean(0), "std": a.std(0)}


def motion(a, s):
    # Relative EEF is a base-frame pose. World reconstruction is diagnostic only.
    base_rot = Rotation.from_quat(s[:, 3:7])
    world_eef = s[:, :3] + base_rot.apply(s[:, 7:10])
    local_base = Rotation.from_quat(s[0, 3:7]).inv().apply(s[:, :3] - s[0, :3])
    return {"actions": stats(a), "state": stats(s), "base_net_m": np.linalg.norm(s[-1, :3] - s[0, :3]),
            "base_path_m": np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1).sum(),
            "base_delta_initial_base_frame_m": local_base[-1],
            "eef_world_net_m": np.linalg.norm(world_eef[-1] - world_eef[0]),
            "eef_world_path_m": np.linalg.norm(np.diff(world_eef, axis=0), axis=1).sum(),
            "eef_relative_range_m": np.ptp(s[:, 7:10], axis=0),
            "eef_translation_command_norm_mean": np.linalg.norm(a[:, 5:8], axis=1).mean(),
            "eef_rotation_command_norm_mean": np.linalg.norm(a[:, 8:11], axis=1).mean(),
            "base_command_active_fraction_gt_001": (np.linalg.norm(a[:, :3], axis=1) > .01).mean(),
            "desired_mode_fraction": (a[:, 4] >= .5).mean(), "desired_mode_runs": runs(a[:, 4] >= .5),
            "close_command_fraction": (a[:, 11] >= .5).mean(), "close_command_runs": runs(a[:, 11] >= .5)}


def episode_frames(meta, cam):
    key = f"videos/observation.images.{cam}"
    path = DATA / f"videos/observation.images.{cam}/chunk-{meta[key + '/chunk_index']:03d}/file-{meta[key + '/file_index']:03d}.mp4"
    start = float(meta[key + "/from_timestamp"])
    n = int(meta["length"])
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        c.seek(int(start / stream.time_base), stream=stream, backward=True)
        count = 0
        for f in c.decode(stream):
            t = float(f.pts * stream.time_base)
            if t < start - .0001:
                continue
            if count == n:
                break
            assert abs(t - start - count / 20) < .0002, (cam, count, t, start)
            yield f.to_ndarray(format="rgb24")
            count += 1
        assert count == n, (cam, count, n)


def encode(path, frames, size):
    with av.open(str(path), "w") as c:
        stream = c.add_stream("libx264", rate=20)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "fast"}
        n = 0
        for array in frames:
            f = av.VideoFrame.from_ndarray(np.asarray(array), format="rgb24")
            for packet in stream.encode(f):
                c.mux(packet)
            n += 1
        for packet in stream.encode():
            c.mux(packet)
    with av.open(str(path)) as c:
        decoded = sum(1 for _ in c.decode(video=0))
    assert n == decoded
    return {"path": str(path.relative_to(ROOT)), "frames": n, "fps": 20, "sha256": sha(path)}


def export(ep, meta, df):
    arrays = [list(episode_frames(meta, cam)) for cam in CAMS]
    n = len(df)
    marks = sorted(set([0, n - 1, *np.linspace(0, n - 1, 9, dtype=int).tolist(),
                        int(np.flatnonzero(df["next.reward"].values > 0)[0])]))
    sheet = Image.new("RGB", (768, 300 * len(marks)), "white")
    export_rows = []
    def frames():
        for i in range(n):
            canvas = Image.new("RGB", (768, 300), "#101820")
            draw = ImageDraw.Draw(canvas)
            task = "Close the fridge doors." if int(df.task_index.iloc[0]) == 0 else "Close the fridge door."
            draw.text((8, 2), f"DEMO {ep} | t={i/20:.2f}s | frame {i}/{n-1} | {task}", font=FONT, fill="white")
            draw.text((8, 22), f"LEFT                           RIGHT                           WRIST       next.reward={df['next.reward'].iloc[i]:.0f}", font=FONT, fill="white")
            for j in range(3):
                canvas.paste(Image.fromarray(arrays[j][i]), (j * 256, 44))
            if i in marks:
                row = marks.index(i)
                sheet.paste(canvas, (0, row * 300))
            export_rows.append({"frame": i, "episode_timestamp": float(df.timestamp.iloc[i]),
                                "source_pts_seconds": float(meta[f"videos/observation.images.{CAMS[0]}/from_timestamp"]) + i/20,
                                "next_reward": float(df['next.reward'].iloc[i]), "next_done": bool(df['next.done'].iloc[i])})
            yield np.asarray(canvas)
    result = encode(OUT / f"demo-{ep:03d}-three-camera.mp4", frames(), (768, 300))
    sheet.save(OUT / f"demo-{ep:03d}-contact-sheet.png")
    save(f"demo-{ep:03d}-timestamps.json", export_rows)
    # Arrays are retained only for the comparison composition, not model input.
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 437, 5])
    args = parser.parse_args()
    assert len(args.episodes) == 3 and len(set(args.episodes)) == 3, "Select three distinct demonstration episodes"
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = DATA / "data/chunk-000/file-000.parquet"
    meta_path = DATA / "meta/episodes/chunk-000/file-000.parquet"
    df = pq.read_table(data_path).to_pandas()
    eps = pq.read_table(meta_path).to_pandas()
    task_table = {x["task_index"]: x["task"] for x in pq.read_table(DATA / "meta/tasks.parquet").to_pylist()}
    assert len(df) == 155443 and len(eps) == 513
    assert np.array_equal(df['index'], np.arange(len(df)))
    reset = json.loads((RUN / "reset.json").read_text())
    steps = [json.loads(x) for x in (RUN / "steps.jsonl").read_text().splitlines()]
    ps = np.array([reset["state"]] + [r["returned_state"] for r in steps])
    pa = np.array([r["action"] for r in steps])
    all_s, all_a = np.stack(df['observation.state']), np.stack(df.action)
    assert all_s.shape == (155443, 16) and all_a.shape == (155443, 12)
    assert np.isfinite(all_s).all() and np.isfinite(all_a).all()
    checks, metrics = [], {}
    for _, meta in eps.iterrows():
        e = int(meta.episode_index)
        q = df.iloc[int(meta.dataset_from_index):int(meta.dataset_to_index)]
        n = len(q)
        assert len(q) == meta.length and np.all(q.episode_index == e)
        assert np.array_equal(q.frame_index, np.arange(n))
        assert np.allclose(q.timestamp, np.arange(n)/20, atol=2e-6, rtol=0)
        assert q.task_index.nunique() == 1 and q['annotation.human.task_description'].equals(q.task_index)
        assert np.all(q['annotation.human.task_name'] == 2) and task_table[2] == 'CloseFridge'
        for cam in CAMS:
            key = f'videos/observation.images.{cam}'
            assert abs(float(meta[key+'/to_timestamp'] - meta[key+'/from_timestamp']) - n/20) < 1e-8
        checks.append({'episode': e, 'length': n, 'task': task_table[int(q.task_index.iloc[0])],
                       'terminal_reward': float(q['next.reward'].iloc[-1]), 'max_reward': float(q['next.reward'].max()),
                       'done_count': int(q['next.done'].sum()), 'last_done': bool(q['next.done'].iloc[-1]),
                       'first_positive_reward_frame': int(np.flatnonzero(q['next.reward'].values > 0)[0]),
                       'positive_reward_frames': int((q['next.reward'] > 0).sum())})
        a, s = np.stack(q.action), np.stack(q['observation.state'])
        metrics[e] = motion(a, s)
    save('episode-audit.json', checks)
    save('episode-metrics.json', metrics)
    initial = all_s[eps.dataset_from_index.astype(int)]
    pos_distance = np.linalg.norm(initial[:, 7:10] - ps[0, 7:10], axis=1)
    quat = initial[:, 10:14] / np.linalg.norm(initial[:, 10:14], axis=1)[:, None]
    ref = ps[0, 10:14] / np.linalg.norm(ps[0, 10:14])
    angles = np.degrees(2*np.arccos(np.clip(np.abs(quat @ ref), 0, 1)))
    state_comparison = []
    for i, name in enumerate(STATE_NAMES):
        state_comparison.append({'dimension': i, 'name': name, 'reset': ps[0, i],
                                 'initial': stats(initial[:, i]), 'all_frames': stats(all_s[:, i]),
                                 'reset_percentile_among_initial': 100*np.mean(initial[:, i] <= ps[0, i]),
                                 'outside_initial_range': bool(ps[0,i] < initial[:,i].min() or ps[0,i] > initial[:,i].max())})
    summary = {'dataset_revision': REV, 'total_episodes': len(eps), 'total_frames': len(df),
               'task_table': task_table, 'episode_instruction_counts': df[df.frame_index==0].task_index.value_counts().to_dict(),
               'reward_counts': df['next.reward'].value_counts().to_dict(),
               'all_terminal_rewards_one': all(c['terminal_reward'] == 1 for c in checks),
               'all_done_once_at_end': all(c['last_done'] and c['done_count'] == 1 for c in checks),
               'positive_reward_tail_lengths': sorted(set(c['positive_reward_frames'] for c in checks)),
               'all_action_stats': stats(all_a), 'all_state_stats': stats(all_s),
               'all_translation_command_norm_mean': np.linalg.norm(all_a[:,5:8],axis=1).mean(),
               'all_rotation_command_norm_mean': np.linalg.norm(all_a[:,8:11],axis=1).mean(),
               'all_desired_mode_fraction': (all_a[:,4]>=.5).mean(), 'all_close_fraction': (all_a[:,11]>=.5).mean(),
               'episodes_with_any_close': [e for e,m in metrics.items() if m['close_command_fraction'] > 0],
               'episodes_with_base_commands_gt_001': sum(m['base_command_active_fraction_gt_001']>0 for m in metrics.values()),
               'state_comparison': state_comparison, 'reset_nearest_eef_relative_position':
               [{'episode': int(i), 'distance_m': pos_distance[i], 'orientation_difference_deg': angles[i]} for i in np.argsort(pos_distance)[:10]],
               'policy': motion(pa, ps), 'selected': {e: metrics[e] for e in args.episodes}}
    # Endpoints 0..50, 50..100 etc. Action comparisons use commands 1..50 etc.
    chunks = []
    for start in range(0, 600, 50):
        item = motion(pa[start:start+50], ps[start:start+51])
        item.update(action_start=start+1, action_end=start+50,
                    door_start=reset['fridge'] if start == 0 else steps[start-1]['fridge'],
                    door_end=steps[start+49]['fridge'])
        chunks.append(item)
    save('policy-chunks.json', chunks)
    save('analysis.json', summary)
    demo_chunks = []
    for e in args.episodes:
        q = df[df.episode_index == e]
        a, s = np.stack(q.action), np.stack(q['observation.state'])
        for start in range(0, len(q), 50):
            stop = min(len(q), start + 50)
            demo_chunks.append({'episode': e, 'frame_start': start, 'frame_end_exclusive': stop,
                                'time_start': start/20, 'metrics': motion(a[start:stop], s[start:stop])})
    save('demo-chunks.json', demo_chunks)
    normalizer = ROOT / ('outputs/phase1a/hf-cache/hub/models--lerobot--smolvla_robocasa/snapshots/'
                         'f7af588d11e30b0069f200792fb285904b74c80b/'
                         'policy_preprocessor_step_5_normalizer_processor.safetensors')
    normalizer_stats = load_file(str(normalizer))
    mean, std = normalizer_stats['observation.state.mean'], normalizer_stats['observation.state.std']
    normalized = (ps[0] - mean) / (std + 1e-8)
    save('reset-normalization.json', {'state': ps[0], 'checkpoint_mean': mean, 'checkpoint_std': std,
                                      'normalized': normalized, 'range': [normalized.min(), normalized.max()],
                                      'stats_sha256': sha(normalizer)})
    videos, arrays = [], {}
    for e in args.episodes:
        meta = eps[eps.episode_index == e].iloc[0]
        q = df.iloc[int(meta.dataset_from_index):int(meta.dataset_to_index)]
        v, arrays[e] = export(e, meta, q)
        videos.append(v)
        print(f'Exported demonstration {e}: {len(q)} frames', flush=True)
    # Same elapsed-time comparison; finished demonstrations are explicitly frozen.
    # Each column has three original camera views, with the policy as final column.
    with av.open(str(RUN/'rollout.mp4')) as c:
        policy_frames = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
    assert len(policy_frames) == 601
    def comparison_frames():
        for i in range(601):
            im = Image.new('RGB', (1024, 842), '#101820'); draw = ImageDraw.Draw(im)
            for col, e in enumerate(args.episodes):
                n = len(arrays[e][0]); j = min(i, n-1)
                draw.text((col*256+4, 3), f'DEMO {e} | t={j/20:.2f}s', font=FONT, fill='white')
                draw.text((col*256+4, 24), 'END / FROZEN' if i>=n else 'recorded demonstration', font=FONT, fill='#8ee8a5')
                for camera in range(3):
                    im.paste(Image.fromarray(arrays[e][camera][j]), (col*256, 74+camera*256))
            draw.text((772,3), f'POLICY seed42 | t={i/20:.2f}s', font=FONT, fill='white')
            draw.text((772,24), f'action {i} | success=false', font=FONT, fill='#ffb3aa')
            for col in range(4):
                draw.text((col*256+4,49), 'Top: LEFT / middle: RIGHT / bottom: WRIST', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',11), fill='white')
            # Original policy montage has a 32 px header and three 256px cameras.
            for camera in range(3):
                im.paste(Image.fromarray(policy_frames[i][32:288, camera*256:(camera+1)*256]), (768,74+camera*256))
            if i in [0, 150, 285, 450, 600]:
                im.save(OUT/f'comparison-{i:03d}.png')
            yield np.asarray(im)
    videos.append(encode(OUT/'demonstrations-vs-policy.mp4', comparison_frames(), (1024,842)))
    sources = [data_path, meta_path, DATA/'meta/info.json', DATA/'meta/tasks.parquet',
               RUN/'steps.jsonl', RUN/'reset.json', RUN/'manifest.json', RUN/'rollout.mp4',
               ROOT/'outputs/phase1a/train_config.json', normalizer]
    sources += [DATA/f'videos/observation.images.{cam}/chunk-000/file-000.mp4' for cam in CAMS]
    save('manifest.json', {'script_sha256': sha(Path(__file__)), 'episodes': args.episodes,
                           'inputs_sha256': {str(p.relative_to(ROOT)):sha(p) for p in sources},
                           'videos': videos, 'no_simulator_or_policy_execution': True,
                           'alignment': 'same elapsed time; demo end held and labeled; not matched task progress'})
    print(json.dumps(plain({'audit_passed': True, 'episodes':args.episodes,'videos':videos}), indent=2))


if __name__ == '__main__':
    main()
