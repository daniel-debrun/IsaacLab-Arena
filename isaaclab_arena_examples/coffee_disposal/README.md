# Dispose of spent coffee grounds

`coffee_disposal` recreates the complete machine-service sequence observed in
`dispose_coffee.mp4`: raise the dispenser, extract the tray, open the waste
drawer, remove its organic-bin lid, lift/carry/empty the grounds container,
reseat it, reinstall the tray, replace the lid, and close the drawer.

The source was analyzed across the full clip and dense samples around short
actions. [Reference analysis](../../COFFEE_REFERENCE_ANALYSIS.md) records 27
observable actions and transitions, mapped into eleven ordered simulation
stages. The video does not show washing or liquid disposal, and the dispenser
remains raised at the end. Small decorative residue is retained in the cup;
the eight dynamic clumps represent the loose grounds that must be disposed of.

## Run in the existing Arena runtime

From the host, discover the container mounting the parent checkout:

```bash
ARENA_CONTAINER=$(docker ps \
  --filter volume=/home/dtingdahl/develop/IsaacLab-Arena \
  --format '{{.Names}}' | head -1)
docker exec -it "$ARENA_CONTAINER" su dtingdahl
```

Inside that container:

```bash
cd /workspaces/isaaclab_arena/.worktrees/coffee-disposal
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Initialize, settle, inspect/reset the scene.
/isaac-sim/python.sh isaaclab_arena_examples/coffee_disposal/demo.py \
  --smoke --capture --output outputs/coffee_smoke

# Complete the physical assisted demonstration and verify reset.
/isaac-sim/python.sh isaaclab_arena_examples/coffee_disposal/demo.py \
  --output outputs/coffee_disposal

# Open an interactive Kit window (requires display access).
/isaac-sim/python.sh isaaclab_arena_examples/coffee_disposal/demo.py \
  --viz kit --real-time --output outputs/coffee_gui

# Capture normal-speed, 2x and 5x MP4s, without requiring a display.
ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1 \
  /isaac-sim/python.sh isaaclab_arena_examples/coffee_disposal/demo.py \
  --num_envs 2 --video --output outputs/coffee_disposal_verified

# Ordinary environment with the standard Franka action interface.
/isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
  --viz kit --device cpu --no_solve_relations coffee_disposal
```

A fresh worktree does not inherit populated submodules. Reusing the parent
container and setting `PYTHONPATH` selects this worktree's code while keeping the
installed Isaac Lab runtime. A standalone checkout needs normal Arena setup.

## Demonstration and video

The demonstration is explicitly **force-assisted**: smooth virtual-grasp forces
move the cup/lid, while position drives move the dispenser, tray and drawer.
Collisions remain enabled, released objects settle physically, and the grounds
fall under gravity when the cup is inverted. It demonstrates scene dynamics,
ordered task completion and reset; it is not a trained robot policy or verified
physical finger-grasp controller. The ordinary environment retains the normal
seven-dimensional Franka action interface and passive sliding joints. The bare
arm is mounted on the front of the countertop, with the bins elevated on a
physical steel support inside the drawer. The dispenser uses static friction to
hold its raised setting without an active servo. Sampled joint-limited reach
checks cover the work area; continuous collision-free robot execution remains
unverified.

The recording starts with an overview of the kitchen and a readable task plan
before any manipulation. Camera cuts then cover the machine, extracted tray,
waste drawer, falling grounds, the deposited grounds inside the bin, and the
final restored scene. Output is 1440 × 1080,
H.264 at 30 fps, sampled from 120 Hz physics with 15 Hz control. Brief inspection
holds do not advance physics. With multiple environments, `--video` records
environment zero while task validation covers every environment. Recording stops
on environment zero's success before its reset. Video takes precedence over
`--capture`'s separate PNG sequence.

Files are `coffee_disposal_1x.mp4`, `coffee_disposal_2x.mp4`,
`coffee_disposal_5x.mp4`, `video_manifest.json`, `result.json` and
`episodes.jsonl`. Videos remain outside Git under `outputs/`. `ffmpeg` with
libx264 and DejaVu Sans fonts is required. `--smoke --video` records only setup
views; inspect `result.json` to distinguish it from a successful full task.

The validated two-environment recording is in `outputs/coffee_disposal_verified/`.
Its 1x/2x/5x durations are 102.767/51.400/20.567 seconds. Both episodes completed
all eleven stages and reset; all video frames decoded with no blank scene frames.
The output includes a preview contact sheet and `artifact_validation.json`.
The recorder rejects blank captures and saves camera/scene diagnostics if that
failure recurs. Each episode keeps direct terminal states as well as Arena's
progress history.

## Asset construction and limits

The machine, kitchen, waste drawer, grounds container, removable lid and coffee
clumps are custom USD geometry authored in
[`build_assets.py`](../../isaaclab_arena_environments/coffee_disposal/build_assets.py).
PNG labels/wood grain are generated with Pillow. The Franka robot is reused from
Isaac Lab. [Asset provenance](../../isaaclab_arena_environments/coffee_disposal/assets/ATTRIBUTION.md)
records the sources and rebuilding procedure. No source-video frames are used
as runtime textures or committed to Git.

Metric dimensions and friction are engineering approximations from an
uncalibrated handheld video. Grounds are rigid clumps, not a fine-grained wet
powder model. The source obscures the drawer-opening contact and most falling
grounds, so the simulated pull and clump shapes/count are implementation choices.
The machine uses fictional AROMA branding and does not model brewing, electronics
or water flow.

## Task checks

Eleven ordered stages require observed changes, aligned placement, low speed and
release. Grounds settling uses centre and collision-surface speeds, accounting
for the clumps' small size, plus an eight-control-step dwell. Grounds must be
inside the organic bin, not the neighboring recycling bin. Final success rechecks tray, cup, lid, drawer, raised dispenser and grounds
retention. Floor drops fail; grounds left elsewhere prevent success. Partial
resets clear only the selected environment's history and assistance.

Focused tests (inside the initialized container):

```bash
/isaac-sim/python.sh -m pytest -q \
  isaaclab_arena/tests/test_coffee_disposal_task.py \
  isaaclab_arena/tests/test_coffee_disposal_motion.py

# Camera cuts and terminal holds (separate RTX-enabled process).
/isaac-sim/python.sh -m pytest -q -m with_cameras \
  isaaclab_arena/tests/test_coffee_disposal_video.py
```

[Findings and validation log](../../COFFEE_DISPOSAL_FINDINGS.md) records measured
results and implementation limits. Generated videos, reference media and runtime
outputs are kept separate from source assets.
