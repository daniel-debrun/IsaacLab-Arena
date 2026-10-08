# PC upgrade environment: findings and work log

Current worktree: `/home/dtingdahl/develop/IsaacLab-Arena/.worktrees/pc-upgrade`.
Branch: `dtingdahl/feature/pc-upgrade-task`.

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
