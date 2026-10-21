# EmbodiedOps interactive demonstration

The presentation covers VLA training, robot control, physical diagnostics and the
diagnosis/repair tool loop. Successful and failed rollouts come first; the four
sustained learning experiments share one table. Terminal-window is an expandable
case study, not the entire presentation.

## Watch the public recordings

**[Open the interactive Demo](https://hanlizhang.github.io/embodiedops/demo/)**

The three original videos are now connected in `docs/demo/index.html`. No local
video directory is required for online playback. Download that HTML and open it
in a browser; use the clip selectors to switch recordings. Each clip also has an
**Open video** link.

- [Completed task: episode-balanced, seed 44](https://drive.google.com/file/d/1v1GlguQ2h86hdpv7uq4vtzRITG5HGdgZ/view?usp=drivesdk)
- [Incomplete closure: episode-balanced, seed 43](https://drive.google.com/file/d/1HovTLk54HdhBUJH_UcA2cx4Ky4ebHMIJ/view?usp=drivesdk)
- [Terminal-window candidate: seed 43](https://drive.google.com/file/d/1N2Vh7-ny13ELoIvPfFjYbKI4KfzzLTq8/view?usp=drivesdk)

## Play locally with the existing recordings

On the Windows evidence machine, update the checkout and choose a fresh output:

```powershell
git pull --ff-only
python -B scripts/build_portfolio_demo.py --output outputs/portfolio-demo-v3
ii .\outputs\portfolio-demo-v3\index.html
```

If `portfolio-demo-v3` already exists, choose another fresh name. The builder
never overwrites historical outputs. It verifies original results, summaries,
manifests and MP4 containers, uses `docs/demo/index.html`, and inserts correct
relative video paths. Google Drive is not needed for local playback. No model,
training, inference or simulation is started, and no new dependency is required.

A source-only clone does not contain the videos. Missing original artifacts stop
the local build with exact paths. The resulting `provenance.json` retains input
hashes, template, builder and generated-HTML provenance.

## Three original recordings

All paths below are relative to the repository root.

| Slot | Original recording | Recorded outcome |
|---|---|---|
| `success` | `outputs/agent-workflow-v1/learning-5000-v1/diagnostic-44-episode_balanced/rollout.mp4` | Task success at action 220; final opening 0.31° |
| `incomplete` | `outputs/agent-workflow-v1/learning-5000-v1/diagnostic-43-episode_balanced/rollout.mp4` | 600 actions; 13.11° remains open |
| `candidate` | `outputs/agent-workflow-v1/terminal-window-v1/diagnostic-43/rollout.mp4` | 600 actions; 90.00° remains open |

These are selected illustrations, not success-rate estimates. The page includes
the complete four-scene comparison. Playback follows recorded simulation time,
not wall-clock inference speed. Original recording overlays remain unchanged.

## Hosting and future video updates

The public recording links are already configured. To replace a recording:

1. Upload the corresponding original file. For Drive, allow anyone with the link to
   view each video and wait until processing completes.
2. Check playback while signed out.
3. Insert the three file-sharing URLs into the `video-sources` JSON block in
   `docs/demo/index.html`, using the slot names above. Do not use a folder link.
4. The page converts Drive URLs to iframe previews and preserves `resourcekey`.
   HTTPS MP4 URLs use the native video player.
5. Publish the HTML through GitHub Pages or another static host and check all
   three clips before adding the hosted URL to a CV. A `github.com/.../blob/...`
   address opens the source file and should not be labelled as a playable demo.

GitHub Pages publishes **main → /docs**. The presentation is served at the
`/demo/` path; future pushes to `main` update the hosted page.
`docs/.nojekyll` keeps the HTML as a static file. Never replace a missing recording
with generated imagery.

## Evidence

The presentation embeds the measured plots from `docs/assets/terminal-window-loss.svg`
and `docs/assets/terminal-window-doors.svg`. Its results come from the compact
[record](../environment/terminal-window-experiment-v1-results.json) and
[technical overview](TECHNICAL_OVERVIEW.md). Experimental limitations and the
runtime-cap deviation remain in the expandable case study and original reports.
