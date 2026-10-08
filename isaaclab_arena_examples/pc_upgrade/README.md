# PC upgrade service bench

`pc_upgrade` is a registered Arena environment with a Franka arm, an articulated
PC service panel, a branded NVIDIA GeForce RTX 5090, fictional VORTEX VX 880 and
CINDER C700 cards, and an electronics recycling bin. Assets are shipped locally.
The existing Franka robot asset is reused from Isaac Lab.

Run these commands inside an initialized Arena container as the host user.
No policy server or model checkpoint is required. A fresh Git worktree does not
inherit populated submodules: launching a new container on it can produce
`No module named 'isaaclab'`.

For this worktree, reuse the parent checkout's existing runtime. From a **host
terminal**, discover and enter the container mounting the parent checkout:

```bash
ARENA_CONTAINER=$(docker ps \
  --filter volume=/home/dtingdahl/develop/IsaacLab-Arena \
  --format '{{.Names}}')
docker exec -it "$ARENA_CONTAINER" su dtingdahl
```

Then, inside that container:

```bash
cd /workspaces/isaaclab_arena/.worktrees/pc-upgrade
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The commands below now use the feature worktree with the parent runtime.
A standalone checkout needs its dependencies initialized using the repository's
setup instructions first.

```bash
# Validate initial scene stability and resets.
/isaac-sim/python.sh isaaclab_arena_examples/pc_upgrade/demo.py --smoke

# Exercise the complete task and write result.json.
/isaac-sim/python.sh isaaclab_arena_examples/pc_upgrade/demo.py \
  --output /tmp/pc-upgrade-result

# Also save overview images of the stages.
/isaac-sim/python.sh isaaclab_arena_examples/pc_upgrade/demo.py \
  --capture --output /tmp/pc-upgrade-result

# Run two independent environments.
/isaac-sim/python.sh isaaclab_arena_examples/pc_upgrade/demo.py --num_envs 2

# Inspect and manipulate the ordinary scene with the Kit GUI (Shift + drag).
/isaac-sim/python.sh isaaclab_arena/scripts/environment_runner.py \
  --viz kit --device cpu pc_upgrade

# Smoke-test through Arena's standard policy runner.
/isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action --num_steps 120 pc_upgrade
```

The demo is an **assisted state demonstration**: it moves rigid-body poses and the
panel joint along scripted trajectories, then releases cards to fall and settle
under physics. It verifies the environment, task progression, terminal conditions,
and reset. It does **not** demonstrate an autonomous robot policy or physical
finger grasps. The ordinary environment has no pose-writing controller: policies
receive the normal seven-dimensional Franka relative-pose/gripper action space.
No trained policy is included.

## Task contract

The episode budget is 300 seconds, with six ordered stages:

1. Open the panel beyond 80 degrees for five control steps.
2. Lift the RTX 5090 above the bench and outside the PC footprint with the panel open.
3. Seat the RTX 5090 in its horizontal riser carrier. Position tolerance is
   12 mm in X/Y and 8 mm in Z, angular tolerance 10 degrees; the hand must be
   at least 8 cm outside the card bounds and the card must settle for ten steps.
4. Close the panel within three degrees, retaining the seated card, for ten steps.
5. Lift and dispose of the VORTEX card after closing the panel; settle fully inside the bin.
6. Lift and dispose of the CINDER card, then retain the installed RTX 5090, closed
   panel, and both contained cards for fifteen steps.

Each pickup requires an observed settled card at bench height before its lift.
Disposal requires a fresh lift outside the bin after the preceding stage, so
pre-positioned cards cannot skip the sequence. Full oriented card bounds must be
inside the bin and below its rim. Linear speed must be below 3 cm/s and angular
speed below 0.15 rad/s. A GPU falling 15 cm below the tabletop ends the episode
as a failure. Arena reports per-stage completion and overall success.

The PC lies on its back with a horizontal PCIe riser carrier. Card geometry is
representative hardware (304 × 137 mm footprint), not manufacturing CAD. Electrical
connections, screws, power cables and latch forces are outside this task. The
panel and cards have real collision and rigid-body dynamics; no installation snap
or hidden success flag is used. Mimic data generation is not configured.

## Integration and assets

- Typed config: `PCUpgradeEnvironmentCfg` in `pc_upgrade_environment.py`.
- State observation: `task` contains three GPU poses, the chassis pose, and the hinge angle;
  positions are relative to the environment origin and quaternions are `xyzw`.
- Set `enable_cameras=True` for wrist and overview RGB observations.
- Asset source: `isaaclab_arena_environments/pc_upgrade/build_assets.py`.
- Asset provenance and logo attribution: `pc_upgrade/assets/ATTRIBUTION.md`.
- Implementation findings and validation record: `PC_UPGRADE_FINDINGS.md`.

Regenerate assets using the Arena interpreter (Pillow, USD and DejaVu Sans are
required). The committed assets do not require generation at runtime:

```bash
/isaac-sim/python.sh isaaclab_arena_environments/pc_upgrade/build_assets.py
/isaac-sim/python.sh -m pytest -q isaaclab_arena/tests/test_pc_upgrade_geometry.py \
  isaaclab_arena/tests/test_pc_upgrade_task.py
```

The full simulation regression starts its own Isaac Sim child process:

```bash
/isaac-sim/python.sh -m pytest -q -m with_subprocess \
  isaaclab_arena/tests/test_pc_upgrade_simulation.py
```

`result.json` identifies the assisted mode and reports per-environment success;
`episodes.jsonl` contains Arena's terminal progress and ordered predicate events.
Images are captured before the successful episode automatically resets.

## Saved preview and validation

![Service bench](previews/scene.png)

![Distinct graphics cards](previews/cards.png)

`validation/result.json` and `validation/episodes.jsonl` preserve the passing
two-environment assisted regression and its ordered terminal events.
