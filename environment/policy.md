# Policy / Robot Learning Environment

Verified locally on Windows 11 with an NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM).

## Conda environment

```powershell
conda create -n embodiedops python=3.12 -y
conda activate embodiedops
```

## PyTorch

```powershell
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Verified:

- PyTorch: `2.11.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`

## LeRobot

```powershell
conda install ffmpeg -c conda-forge -y
pip install "lerobot[training,diffusion,pusht]"
```

Verified:

- LeRobot: `0.6.1`
- NumPy: `2.2.6`
- `opencv-python-headless`: `4.13.0.92`
- PushT environment loads and renders
- Diffusion Policy forward/backward training runs on CUDA
- Checkpoint saving succeeds with Windows Developer Mode enabled
- PushT evaluation and rollout-video generation succeed

## OpenCV note

`gym-pusht` declares `opencv-python`, while LeRobot uses `opencv-python-headless`.
Both packages provide the same `cv2` Python namespace and should not be installed together.

The current runtime has been verified with `opencv-python-headless`, including PushT rendering.

As a result, `pip check` may report a metadata-level warning that `gym-pusht` requires `opencv-python`. Treat this as a known dependency metadata mismatch as long as the verified PushT runtime smoke test continues to pass.

## Runtime smoke test

```powershell
python -c "import gymnasium as gym, gym_pusht; env=gym.make('gym_pusht/PushT-v0', render_mode='rgb_array'); obs,info=env.reset(seed=0); frame=env.render(); print('PushT OK'); print(frame.shape, frame.dtype); env.close()"
```
