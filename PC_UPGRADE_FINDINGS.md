# PC upgrade environment: findings and work log

Current worktree: `/home/dtingdahl/develop/IsaacLab-Arena/.worktrees/pc-upgrade`.
Branch: `dtingdahl/feature/pc-upgrade-task`.

The current refinement uses an upright tower and motherboard PCIe insertion,
distinct dual-fan/triple-fan/blower GPUs, a rounded recycling bin, and
physics-rate assisted motion. The original version is preserved in commit
`588f1ca241`; the early sections below are its historical findings. See the
2026-10-08 refinement section for the new design and validation.

## Asset creation and provenance

The PC, all three graphics cards, workbench and recycling bin are custom
procedural assets authored in Python and exported as USD by
[`build_assets.py`](isaaclab_arena_environments/pc_upgrade/build_assets.py).
Their meshes, materials, collision shapes and chassis hinge are defined in
code. Pillow generates the labels, recycling emblem and oak-grain texture.
These props are original code-generated models, not downloaded CAD or stock
3D models; no generative-image service was used to create their textures.

The exceptions are the **NVIDIA logo**, extracted from NVIDIA's official site
and saved as SVG/PNG, and the **Franka robot**, reused through Arena from Isaac
Lab's existing asset library. The RTX model uses published dimensions as a
reference, while its geometry is illustrative. VORTEX and CINDER branding is
fictional. Floor and lighting use standard Isaac Lab spawners.

The [asset provenance document](isaaclab_arena_environments/pc_upgrade/assets/ATTRIBUTION.md)
provides a per-asset breakdown, logo attribution and rebuilding instructions.

## Objective and design

Build a runnable Arena scene with a service bench, hinged PC chassis, visibly branded
NVIDIA GeForce RTX 5090, fictional VORTEX VX 880 and CINDER C700 cards, and an
open electronics recycling bin. Use original procedural USD geometry with PBR
materials, modeled cooling fans, heatsinks, PCB components, fasteners, and readable
labels. The PC lies on its back for top access; its service panel rotates on a
physical hinge. GPU dimensions are representative, not manufacturing CAD.

The ordered task is: open panel, lift RTX 5090, seat and release it in the keyed
slot, close panel, recycle both fictional cards. Success must retain the installed
card, closed panel, and both recycled cards. Use a Franka arm, local assets and
no remote policy dependency. Provide a deterministic assisted demonstration as
well as an ordinary robot-action environment; clearly distinguish assisted scene
validation from autonomous manipulation performance.

## Plan

1. Write tests for pose/containment predicates, ordering and per-environment reset.
2. Generate local USD assets and integrate a typed registered environment and task.
3. Add the runnable demonstration, visual capture, and usage documentation.
4. Run focused tests, real simulation, inspect render, and independently review.

## Decisions and evidence

- 2026-10-07: Created `/tmp/arena-pc-upgrade`, branch
  `dtingdahl/feature/pc-upgrade-task`, from `1d3dfdbb46` (current user branch).
  Existing checkout and untracked files are left untouched.
- Runtime discovered using dev-container workflow: `isaaclab_arena-latest`,
  image `isaaclab_arena:latest`; NVIDIA RTX A6000 available.
- The running container mounts `/tmp`. Use the isolated worktree through that
  mount with an explicit PYTHONPATH; reuse installed Isaac Lab dependencies.
  No runtime restart, image rebuild or global editable-install changes needed.
- Arena supports ordered CompletionCriteria, temporal settling predicates,
  articulated USD assets, typed environment registration and partial resets.
- Choose recycling rather than an incinerator: it supplies a physical containment
  target and avoids adding an unrelated fire/fluid simulation.
- User requested independent execution, so design and routine implementation
  decisions proceed without intermediate approval gates.

## Validation

Completed verification:

- 7 focused task/geometry tests passed.
- 1 end-to-end subprocess regression passed in two environments (181.84 s).
- 2 existing environment-registry tests passed.
- Standard Arena policy runner completed a 20-step zero-action smoke run.
- Full assisted sequence succeeded in one rendered environment and two parallel environments.
- Final clean rendered rollout passed from the persisted worktree path: success `[true]`,
  reset verified, exactly one successful terminal episode, and no renderer errors.
  Its snapshots replace all earlier preview images.
- Final image-only smoke run passed; inspected all three distinct labels and NVIDIA branding.
- Wheel built; verified six USD models, seven PNG textures, SVG, and attribution included.
- Black 24.3.0, isort 5.13.2, flake8 7.0.0 passed for new Python files.

This is focused validation, not a run of Arena's entire three-phase suite.
The demonstration is assisted state control, not a verified autonomous Franka policy.
Saved previews and two-environment result evidence are in
`isaaclab_arena_examples/pc_upgrade/previews/` and `validation/`.

### Implementation progress

- Original USD models and decals generated successfully (about 436 KB total).
- Official logo artwork extracted from https://www.nvidia.com/en-us/ on 2026-10-07;
  the logo is preserved with a white wordmark field on the card label.
- Card reference: https://www.nvidia.com/en-au/geforce/graphics-cards/50-series/rtx-5090/
  (304 × 137 mm footprint). Geometry is an illustrative model, not a certified replica.
- Added typed `pc_upgrade` environment, six ordered stages, settling checks,
  whole-card bin containment, drop failures, state observations, and reset events.
- Geometric tests: initial RED (missing implementation), GREEN (2 tests).
- Integration finding: installed Isaac Lab uses xyzw quaternions. Added regression
  input in that convention, observed failure, corrected quaternion interpretation.
- Assisted demo explicitly writes scene states along trajectories and releases cards
  for gravity settling. It is an integration reference, not a trained-policy score.
- First simulation smoke passed: stable initial poses, zero initial task progress,
  successful reset. Output: `/tmp/pc-smoke/result.json`.
- Rendering integration: this Isaac Lab version deliberately returns None from
  `env.render(rgb_array)`. Added a proper overview RGB camera instead; wrist and
  overview images are available as policy observations with cameras enabled.
- Negative regression exposed early disposal counting later. Tightened disposal
  stages to require a fresh lift outside the bin after the previous stage.
- Whole-card containment now includes the fan/label height and asymmetric visual
  bounds, rather than only the rigid collision box.
- Full assisted physics run passed, including all six ordered stages and reset:
  `/tmp/pc-upgrade-physics/result.json`, `success: [true]`. The installed card
  settled at z=0.8635 m, and remained there through closure and recycling.
- The task/geometry suite passed: 6 tests. Extended it with explicit retention,
  temporal completion, and completed-environment partial-reset assertions.
- Added terminal recording before automatic reset, preserving successful scene
  images and Arena's actual progress/event log in `episodes.jsonl`.
- Independent review found two material predicate gaps. Both now have regression
  coverage: an elevated card before stage activation cannot count as a fresh lift;
  an end effector at the long GPU tip cannot count as release. The reviewer found
  no further important packaging, offset, bounds, or retention issues.
- Review regressions were observed failing, then fixed. Focused suite now passes
  7 tests (Isaac Lab emits existing deprecation/OmniHub warnings).
- Two-environment assisted camera run succeeded in both environments. Found and
  removed a redundant explicit reset which duplicated terminal records and images;
  the successful run now verifies Arena's automatic reset directly.
- Host Black 24.3.0, isort 5.13.2 and flake8 7.0.0 checks pass on new Python files.
- Final simulation regression and final preview checks passed after review fixes.
- Visual inspection found RTX shader texture aliasing across initially identical
  material names. Unique per-card material paths produced the correct RTX 5090,
  VORTEX VX 880 and CINDER C700 labels, verified in the all-card close-up.
- One in-progress capture logged transient texture errors because assets were
  regenerated while it was rendering. A fresh capture after generation completed
  rendered without those errors; the shipped runtime does not regenerate assets.
- Fixed the guide's initial policy-runner path typo and verified the corrected
  `isaaclab_arena/evaluation/policy_runner.py` command. The short zero-action run
  reports zero completed episodes, as expected; it is startup validation only.

## Using this isolated worktree in the existing container

The parent checkout is mounted by the existing Arena container. Run as the host user:

```bash
cd /workspaces/isaaclab_arena/.worktrees/pc-upgrade
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/isaac-sim/python.sh isaaclab_arena_examples/pc_upgrade/demo.py --capture
```

The worktree was moved out of `/tmp` after validation. Its parent directory is
ignored via Git's local `info/exclude`; no tracked files in the original checkout
were changed. No Docker restart, image change, hooks change, or global package
installation was needed.

## Time and token accounting

Task started: 2026-10-07T12:40:38.750000+00:00. Snapshot: 2026-10-07T13:25:50.113327+00:00.
Elapsed wall time: 45.2 minutes (includes simulation, tools, and review).

| Counter | Primary | Reviewer | Combined |
|---|---:|---:|---:|
| input_tokens | 13,028,590 | 459,338 | 13,487,928 |
| cached_input_tokens | 12,820,480 | 410,112 | 13,230,592 |
| uncached_input_tokens | 208,110 | 49,226 | 257,336 |
| output_tokens | 58,984 | 2,131 | 61,115 |
| reasoning_output_tokens | 17,568 | 776 | 18,344 |
| total_tokens | 13,087,574 | 461,469 | 13,549,043 |

Source: local session telemetry, latest per-agent thread usage records.
Input counts include repeatedly processed context. Cached input is a subset of input;
reasoning output is a subset of output. These are not unique-content or billing totals.
Machine-readable snapshot: `PC_UPGRADE_USAGE.json`. Final reporting may add tokens after this snapshot.

## Follow-up: worktree container import error

The new `isaaclab_arena-latest-pc-upgrade` container mounts the feature worktree
at `/workspaces/isaaclab_arena`. Its IsaacLab submodule sources are unpopulated,
so the image's editable install cannot resolve `isaaclab`. Confirmed with
`find_spec("isaaclab") == None` and a missing source `__init__.py`. Verified that
the existing parent container imports `AppLauncher` correctly, while PYTHONPATH
selects Arena from the nested feature worktree. Updated the README with explicit
host/container steps. This route needs no rebuild or dependency installation.

## Refinement pass — 2026-10-08

Baseline preserved in signed-off commit `588f1ca241`, with all staged host
pre-commit hooks passing. This commit reproduces the original service-carrier
scene and its saved validation. The refinement remains in the same isolated
worktree and will be a separate commit.

Design: upright tower, vertical motherboard, real side-facing PCIe socket,
horizontal card insertion, rear-hinged side panel with visible hardware.
Rotate the selected card 180 degrees about its long axis before insertion so
its connector faces the motherboard and its cooling fans face down. Preserve
the six ordered stages and require an aligned approach outside the socket
before crediting insertion. Distinct cooler designs differentiate the three
GPUs. A tapered hollow bin with rim, handles and recycling graphics replaces
the plain rectangular container. Move bench instructions to a clear front strip.

Jitter investigation: baseline writes poses and zero velocity at 15 Hz while
physics runs at 120 Hz. Gravity acts during the eight intervening substeps.
Measure this sawtooth motion, then validate a demo-only physics-rate controller
with continuous forces and smooth trajectories. The ordinary robot-action
environment must not acquire hidden assistance.

Validation planned: geometry/order/partial-reset regressions, quantitative
motion checks, full two-environment task and reset, rendered asset/scene
inspection, host commit hooks. Runtime is the existing parent-checkout
container; no shared dependency or Docker configuration changes.

### Refinement investigation and checks

- Preserved the original version with a signed-off commit after all host hooks
  passed; no changes to the parent checkout were included.
- Measured the original controller's gravity sag: 21.885 mm every 66.67 ms,
  followed by a pose reset. Replaced state writes with a demo-only action term
  applying bounded gravity-compensated forces/torques every 1/120 s. Trajectories
  start and stop with zero velocity and acceleration. Card collisions stay active.
- Added a demo-only smooth position drive for the hinged panel; ordinary policy
  environments retain the passive damped hinge and seven robot action dimensions.
- Seven task/geometry tests passed. New regression was observed failing before
  the socket orientation and approach-history checks were implemented.
- Independent review found partial-reset assistance leaking across clones.
  Fixed with per-environment active masks and indexed wrench clearing. Two
  focused motion tests passed, including an actual airborne grasp and panel hold
  retained in clone one while clone zero resets, followed by physical release.
  Reviewer checked the fix and found no remaining issue within that scope.
- First insertion attempt tilted at coplanar support contact. A USD collision
  audit exposed zero lead-in clearance. Lowered bracket tops 2 mm and widened
  the simplified socket collision channel to 7 mm around its 3 mm tongue.
  The card now slides freely then settles 2 mm onto the socket/supports.
- Force-based control exposed two paths crossing obstacles that pose writes had
  obscured. Move/rotate the RTX in front of the open door before approaching the
  socket. Carry discarded cards above the robot/tower and lower over the bin.
- Initial renders verified unobstructed table instructions. Corrected heatsink
  fins crossing fan faces. Added a single-blower CINDER cooler, subtle oak grain,
  board traces and ICs, magnetic-catch visuals, and per-card material names.

- Full two-environment regression passed in 192.46 s: both environments completed
  all six stages, retained the installed GPU and closed panel, recycled both mock
  cards, and reset. Exactly two terminal episode records. Saved under the example's
  `validation/` directory. Worst free-space tracking error 0.08037 mm; maximum
  angular tracking error 0.000866 rad. Largest physics-step displacement 3.855 mm
  corresponds to commanded motion (peak speed 0.463 m/s), not residual jitter.
- Built wheel and inspected its contents: all 16 local asset files (six USD,
  eight PNG, SVG and attribution) and the new controller are included.

- Final polished rendered rollout passed with strict 2 mm free-space tracking and
  0.01 rad orientation bounds, success `[true]`, reset verified, exactly one
  terminal episode, and no renderer errors in its log. Fresh previews include
  closed/open tower, all three coolers, empty socket and installed GPU close-ups.
  Headless capture now renders snapshots explicitly; GUI runs retain normal
  viewport rendering. Added soft frontal fill lighting for interior detail.
- All required host pre-commit hooks pass. Ten focused pytest tests passed across
  geometry/order (7), motion/partial reset (2), and full two-environment
  subprocess integration (1). This is not the entire Arena test suite.
- The demo remains explicitly assisted by virtual forces, not a verified
  autonomous robot policy. Manufacturing tolerances, electrical connections,
  cable attachment and mechanical connector latching are outside this model.

### Refinement time and token accounting

Started 2026-10-08T12:25:50.831Z; snapshot 2026-10-08T13:05:45.840063+00:00.
Elapsed wall time: 39.9 minutes, including simulation, tools and review.
Combined output: 81,744 tokens; uncached input: 370,633;
cached input: 20,453,760. Input counts include repeated context.
This pass excludes the overnight gap and previous work. The original usage
snapshot remains in `PC_UPGRADE_USAGE.json`; this pass is recorded separately in
`PC_UPGRADE_REFINEMENT_USAGE.json`. Counters are telemetry snapshots, not billing
estimates, and exclude final bookkeeping after the snapshot.

### Multi-angle video recording

- Added `demo.py --video` and a demo-only recorder sampling every fourth physics
  substep: 30 fps output with unchanged 120 Hz dynamics and 15 Hz control.
- Camera edit: original overview, near-vertical card-name close-up, open
  motherboard/socket, continuous insertion close-up, then overview for panel
  closure and both recycling operations. Short still holds let viewers inspect
  details. Captions explicitly identify the force-assisted demonstration.
- Smoke recording validated H.264 encoding and top-down framing: all three
  cards and their names fit, with the front table instructions visible.
- Reproduction commands are in `isaaclab_arena_examples/pc_upgrade/VIDEO.md`.
  Videos are stored separately in the ignored `outputs/pc_upgrade_video/`
  directory. The main example README was left untouched while its editor lock
  file was present.
- Full recorded rollout completed successfully with all six stages, one terminal
  episode, verified reset, and motion bounds passing. Maximum free-space tracking
  error was 0.08025 mm. The encoder exited successfully with an empty error log.
- Inspected overview, card labels, insertion, installed GPU and completed frames.
  Verified all three MP4 files by full decoding with FFmpeg and metadata checks:
  1440 × 1080, H.264, 30 fps. Normal speed is 91.33 s (2,740 frames), 2× is
  45.73 s, and 5× is 18.33 s; frame rounding adds at most two output frames.
  `video_manifest.json` contains cut timestamps, and `video_validation.json`
  contains the final file sizes, durations and decode-check results.
- Video pass time snapshot: 24.8 minutes from
  2026-10-08T13:14:18.842Z, including rendering and encoding. Output tokens:
  24,226; uncached input: 161,452;
  cached input: 3,096,832. Detailed counters are in
  `PC_UPGRADE_VIDEO_USAGE.json`. These are telemetry deltas, not billing estimates,
  and exclude final bookkeeping after the snapshot.
