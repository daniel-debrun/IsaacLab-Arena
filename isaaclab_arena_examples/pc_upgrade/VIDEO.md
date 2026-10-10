# Multi-angle PC upgrade video

From the feature worktree inside the initialized Arena container (see `README.md`
for the container/worktree setup), record the complete demonstration:

```bash
PYTHONPATH="$PWD" ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1 \
  /isaac-sim/python.sh isaaclab_arena_examples/pc_upgrade/demo.py \
  --video --output outputs/pc_upgrade_video
```

This creates `pc_upgrade_1x.mp4`, `video_manifest.json`, `result.json` and
`episodes.jsonl`. The video uses 1440 × 1080 H.264 at 30 fps. It cuts between the
original overview, a motherboard/socket close-up during insertion, and a zoomed
top-down view showing all three card names. Brief still holds allow time to
inspect the details. Captions describe each stage and identify the demonstration
as force-assisted. Physics remains at 120 Hz and policy control at 15 Hz;
recording samples every fourth physics step, regardless of rendering speed.

Only one environment can be recorded. `--video` renders headlessly by default;
no SSH display is required. `--smoke --video` checks the introductory views and
encoder without executing the task. When combined with `--capture`, video takes
precedence over the separate still-image camera sequence.

The container needs `ffmpeg` with the `libx264` encoder and DejaVu Sans fonts.
The MP4 initially has a `.partial.mp4` suffix and is renamed when the run passes
its checks. The manifest records frame count, duration and camera-cut timestamps.
Check `result.json` for actual task success; smoke mode only checks setup/reset.

Create faster versions from the finished recording on the host or in the
container, from the same feature-worktree root:

```bash
for speed in 2 5; do
  ffmpeg -y -i outputs/pc_upgrade_video/pc_upgrade_1x.mp4 \
    -vf "setpts=PTS/${speed}" -r 30 -an \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
    -movflags +faststart "outputs/pc_upgrade_video/pc_upgrade_${speed}x.mp4"
done
```

Videos and runtime outputs are ignored by Git under `outputs/`. Keep the
recording code and this recipe for regeneration; copy the MP4 files separately
when sharing the results.
