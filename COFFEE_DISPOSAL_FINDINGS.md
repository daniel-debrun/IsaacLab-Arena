# Coffee disposal: design, findings and work log

Worktree: `.worktrees/coffee-disposal`; branch:
`dtingdahl/feature/coffee-disposal-task`. Base: `71c0cbf8c4` (validated PC task
and recording tools). Parent checkout and PC worktree remain separate.

## Current status

Completed and validated in the isolated worktree. The final force-assisted
rollout passed all eleven stages in both environments, preserved separate
terminal records, and verified reset. Nine focused regressions and repository
hooks passed across the validation runs. No trained finger-grasp policy is claimed.

Final artifacts are in `outputs/coffee_disposal_verified/`:

- `coffee_disposal_1x.mp4`: 102.767 seconds, 3,083 frames.
- `coffee_disposal_2x.mp4`: 51.400 seconds, 1,542 frames.
- `coffee_disposal_5x.mp4`: 20.567 seconds, 617 frames.
- `preview.jpg`, `frames/`, `result.json`, `episodes.jsonl`,
  `video_manifest.json` and `artifact_validation.json` (SHA-256 and checks).

All three videos are 1440 × 1080 H.264 at 30 fps. Every frame was decoded and
scanned outside the caption ribbon: zero blank frames. Nine representative
frames were visually inspected, including the pre-task overview, all six
camera views, deposited grounds and final restoration. The chronological log
below also records earlier rejected diagnostic runs; those are not deliverables.

Run instructions: [coffee example README](isaaclab_arena_examples/coffee_disposal/README.md).
Reference decomposition: [27 observed actions](COFFEE_REFERENCE_ANALYSIS.md).
Time/token accounting: [usage snapshot](COFFEE_DISPOSAL_USAGE.json).

## Reference and intended task

Reference: `/home/dtingdahl/Downloads/dispose_coffee.mp4`, 77.504 s, portrait
presentation (1920 × 1080 HEVC with -90° display rotation), 30 fps, audio track.
Reference footage and extracted personal-video frames remain outside Git.
Detailed timestamp evidence is in `COFFEE_REFERENCE_ANALYSIS.md`.

The reference empties spent coffee grounds from an automatic machine:
1. Raise the height-adjustable dispenser.
2. Pull out the complete drip-tray and grounds-container assembly.
3. Open the under-counter waste drawer.
4. Lift off the organic-bin lid and park it on the neighboring bin.
5. Lift the grounds container out of the drip tray.
6. Carry the container above the organic bin.
7. Invert and rock it to release the grounds into the bin.
8. Return the upright container to its drip-tray recess.
9. Slide the complete tray back into the machine.
10. Retrieve and reseat the organic-bin lid.
11. Close the waste drawer, retaining all final results.

No rinsing, liquid pouring, beverage preparation, or dispenser lowering is
observed. Small adhered grounds residue remains after tipping in the reference.
The simulation represents loose grounds as rigid clumps, plus decorative
residue; it does not model particle-fluid or electrical-machine behavior.

## Design decisions

- Custom USD geometry based on the footage: silver/black automatic machine,
  lit display, twin adjustable nozzles, slotted stainless drip grate, removable
  molded grounds tub, white counter/cabinet, inset sink and faucet, sliding
  two-bin waste drawer with a removable organic lid. Generic AROMA branding.
- Articulated prismatic dispenser, tray and waste drawer; dynamic grounds tub,
  lid and coffee clumps. Simple collision shapes separate from detailed meshes.
- Eleven sequential, physically observed stages with dwell checks, history
  requirements, containment, retention and indexed reset coverage. A configured
  scene is not success just because objects are pre-positioned at final poses.
- Reuse Arena's Franka embodiment. As in the PC task, the integration demo uses
  smooth, explicitly labelled force assistance and joint drives. Ordinary
  policy environments retain normal robot actions; no trained policy is claimed.
- Full task overview before action, then machine, waste-drawer and overhead
  detail views. Export H.264 at 30 fps and derive 2×/5× versions from the master.
- Follow the source order, including replacement of the waste-bin lid after
  the machine tray is reinstalled. Validate real fall/containment of grounds.

## Plan and validation

Implementation plan: `docs/superpowers/plans/2026-10-10-coffee-disposal.md`.
Run all Isaac/Arena code inside the existing parent-checkout container as the
host user, with worktree PYTHONPATH. The container was discovered as
`isaaclab_arena-latest`; imports resolve to the new worktree and parent Isaac Lab.
Host lint/pre-commit only. Do not change Docker configuration or dependencies.

Planned checks: asset articulation/collision audit; meaningful predicate/order
and partial-reset regressions; live smoke and full multi-environment rollout;
rendered visual inspection; full MP4 decoding and speed/duration validation;
packaging and host lint. Time/token counters will be snapshots of this request,
excluding the idle gap since the preceding PC work.

## Implementation evidence

- Independent visual analysis documented 27 timestamped micro-actions and
  transitions. Eleven stateful task stages group those observations; reaching
  and walking between locations are not separate success predicates.
- Original asset builder exported six USD props and five generated textures;
  numeric checks cover joints, collision topology, cradle clearance and lid
  support. Source footage is not copied into runtime assets.
- An early task/framework regression batch passed 21 focused tests, including existing progress-tracker tests.
  An additional actual-Isaac-ProxyArray regression failed at the TorchScript
  quaternion boundary, then passed after converting asset reads to tensors.
- Initial live smoke passed settling and reset and produced overview/machine
  renders. It is not a full-task success. Visual review found an intersecting
  static cabinet rear panel in the sink and incomplete floor boundaries; both
  were corrected and verified in the refined smoke renders.
- Demo assistance partial-reset regression was first run with the implementation
  absent and failed on the expected missing module. The live two-environment green run passed after implementation.

## Integration review and refinements

- The first physical rollout completed raising, extraction, drawer opening, lid
  parking and cup carry with roughly 3 mm peak translation tracking error.
  Released grounds fell into the organic bin. A conservative spherical
  containment approximation rejected two flattened clumps resting against its
  walls; containment now uses the actual authored collision shape instead.
- Review identified missing passive dispenser retention. A live regression failed before physical slider friction was added and passed afterward; ordinary environments
  must not depend on the demonstration's position servo.
- The initial rear-mounted Panda could not reach the open drawer. The revised
  countertop mount is `(0.05, -0.45, 0.90)` with both bins raised 0.13 m on a
  drawer-mounted steel support. Joint-limited CPU inverse-kinematics checks find
  solutions with downward lid/tray approaches, frontal dispenser access and
  changing wrist approach along the drawer extension.
  This is endpoint feasibility, not a claim of collision-free policy execution.
  Geometry review verifies support and sink/counter clearances; lid lift is now
  0.12 m to retain 38 mm clearance below the countertop.
- An additional negative task test rejects grounds deposited before a ceremonial
  empty-cup tip. Valid early discharge before full inversion remains allowed.
- The bare Franka USD is mounted directly on the counter; its inherited floor
  stand is removed. The parked arm turns away from the machine transfer path.
  A CPU URDF audit saved 18 endpoint/intermediate drawer pose solutions under
  `/tmp/coffee-reach/`. It checks joint limits and a coarse countertop envelope,
  not full self/object collisions or a continuous robot controller. Frontal
  dispenser access and changing wrist approach during drawer extension matter.
- Ground containment transforms the actual collider vertices into the moving bin
  frame, with a uniform 0.2 mm solver-contact allowance on floor, walls and rim.
  A diagnostic recording exposed PhysX simplifying the original 360-vertex
  visual hull: rendered vertices extended up to 1.21 mm below the physical floor.
  The temporary floor allowance was removed. Grounds now have a separate,
  invisible 24-vertex/26-face convex collider enclosing the detailed visual mesh;
  it fits the GPU cooking limits without vertex decimation. The live regression
  drops eight differently oriented clumps near every wall and corner, reads the
  actual USD collision points and checks settling/containment. It failed with the
  old hull and passed with the new collider (four task tests, 54.11 s).
- Ordinary dispenser retention uses 3.5 N static joint friction for its 0.30 kg
  slider (gravity is approximately 2.94 N), plus 2 N kinetic friction and damping.
  It remains movable; ordinary scenes have zero position-drive stiffness.
- Time and API token snapshots are maintained in `COFFEE_DISPOSAL_USAGE.json`.
  Cached input is included within total input, and reasoning tokens within output;
  concurrent agent durations are not added to wall-clock elapsed time.

- Before the final collider refinement, a combined regression run passed **5 tests in 113.78 s**: causal task history,
  actual ProxyArray handling, hull boundaries, two-environment partial assistance
  reset, and unassisted dispenser retention. Host hooks passed. Read-only code
  review found no remaining material issue after the three earlier fixes.

- Refined smoke video completed successfully. Its manifest correctly reports
  `recording_valid: true` and `simulation_success: false` (setup-only capture).
  Intro and machine frames were inspected: captions fit, source-like appliance
  details are visible, the sink is unobstructed, and the floor/wall scene is complete.
  A diagnostic full recording exposed the collider mismatch described above;
  it is retained separately and is not presented as a successful result.

- Final collider review independently verified the exported mesh is closed,
  consistently wound and convex, with all primary render vertices enclosed by
  at least 0.05 mm. No new material code issue was found. Package-data globs cover
  all twelve coffee asset files (six USDA, five PNG and attribution).

- The first two-environment captured rollout reached disposal in both scenes,
  but only one passed its settled-ground check at the fixed inspection time.
  Offline transforms of all sixteen exact collision meshes confirmed every
  clump was contained (worst boundary discrepancy 0.0036 mm). One clump resting
  on the other pile still rotated at 0.406 rad/s, above the 0.35 rad/s threshold.
  The demo now waits up to ten additional simulation seconds for observed stage
  completion. Physical containment and settling thresholds remain unchanged.

- A second two-environment run showed that a bounded wait alone does not cure
  persistent tiny-clump contact jitter in one pile. That unsuccessful run is
  retained in `outputs/coffee_full_two_env_validated` (no success report was
  written). A saved-state physical regression is being used to test grounds-only
  damping before another complete rollout; task thresholds remain unchanged.

- Follow-up saved-pile replays passed the original settling test in passive and
  stationary driven drawers, but retained nonzero reported velocities despite
  only micrometre-scale pose changes. Thus replaying pose/velocity alone does
  not recreate the full run's contact history. The initial sleep-state hypothesis
  was not established: these scenes report a zero sleep threshold. Targeted
  grounds-only damping/sleep trials include a moving-drawer check to ensure
  resting clumps still respond to later drawer motion.

- Source inspection explained why initial sleep trials had no effect:
  Isaac Lab's `activate_contact_sensors()` forces sleep thresholds to zero after
  spawn properties are applied. Disabling unused reporting made a positive
  threshold effective, but did not eliminate residual velocities against the
  awake driven drawer. Angular damping 2 and sixteen velocity iterations also
  had negligible benefit. Those experimental physics changes were reverted.
- The measured problem is the grounds' settling metric: applying the same raw
  angular threshold to a 15–18 mm clump and a much larger cup ignores their
  different physical size. Grounds now require centre speed below 0.01 m/s
  (stricter than the previous 0.06 m/s), and every actual collision vertex must
  move below 0.02 m/s using `v + omega × R vertex`. Eight consecutive control
  steps and exact bin containment are still required. Cup/lid criteria remain
  unchanged. The saved problematic pile has stationary poses and surface speeds
  around 0.009 m/s; fast spinning, drifting or falling clumps must still fail.

- Independent review verified the surface-velocity calculation and that testing
  the vertices bounds the full convex hull. This is a bounded-motion criterion,
  not proof of perfectly stationary poses: slow physical rotation is allowed.
  Final containment and retained assembly checks still apply through drawer
  closure. Review also strengthened the combined translation/rotation regression
  so each component passes alone and only their sum fails.

- Final focused validation passed seven task/motion tests in 109.08 seconds,
  including exact hull boundaries, causal order/reset, physical clump drops,
  saved-pile settling in two environments, subsequent drawer transport/retention,
  partial assistance reset, and passive dispenser retention. The strengthened
  combined-motion case passed separately. Host hooks passed.
- Final recording also runs the full task in two environments. Cameras are
  positioned per environment, but the encoder uses environment zero. Its success
  callback captures the restored scene and stops recording before auto-reset;
  the task still requires success and reset from every environment. This path
  received a separate read-only review before the final run.

- The final two-environment recording passed stages 1–10 and reached task success,
  but failed during the pre-reset recording callback: Arena deep-copied a bound
  `Demo.terminal` method while `Demo` was still being initialized, so the copied
  owner lacked its later `base`/video state. The unsuccessful recording is kept
  separately in `outputs/coffee_diagnostic_callback`. The recorder now stores a
  module-level function and resolves the live demo through the environment at
  callback time. A dedicated copy/ownership regression and a new complete
  recording validate this fix; the failed file is not presented as final output.

- The terminal-ownership regression passed in 29.38 seconds. It reproduces the
  old copied-method failure, constructs a real demo, and exercises the real
  pre-reset recorder boundary. Evidence and a video spy verify final completion
  and poses are captured before reset, recording stops, and exactly one episode
  is written. All hooks passed. The complete two-environment recording is being
  repeated with this tested callback.

- The repeat completed **all eleven stages in both environments**, wrote two
  successful episode records, and verified reset. Maximum assisted tracking
  error was 3.10 mm and orientation error 0.0175 rad. The physics outcome is
  valid, but visual inspection rejected this recording: 749 of 3,083 frames
  (77.8 seconds through the end) were uniform gray despite successful encoding.
  Full decoding and metadata checks alone do not establish video validity.
- Camera capture now warms the actual sensor render product and rejects blank
  images before captions are added. Warm-up alone is not a confirmed fix:
  a stationary saved-state regression reproduced blank output immediately after
  synchronizing articulation kinematics with `sim.forward()`. Without that
  synchronization the frames appeared valid but showed stale drawer geometry.
  Rendering diagnosis and a visually validated replacement remain outstanding.

- The camera regression passed in 62.93 seconds after publishing each restored
  fixture with one ordinary setup control step. All six views of both states and
  the terminal hold contained nonblank scene pixels, and visual inspection
  confirmed the open bin with eight clumps and the restored final cabinet.
  Camera cuts/holds themselves changed neither physics step counters nor rigid
  states. These fixture renders are diagnostics, not replacement footage; the
  original long-run blackout cause is still unconfirmed. No renderer settings
  or upstream runtime files were changed.
- Final review found video mode had replaced Arena's progress recorder, leaving
  episode records without detailed stage histories. The demo is being corrected
  to preserve progress recording and capture terminal state independently for
  each environment, including staggered completions. Rendering remains limited
  to environment zero.

- The per-environment terminal regression passed in 28.03 seconds. Video mode
  now retains `ProgressTrackingRecorderManagerCfg`, and each episode includes
  direct pre-reset stages, rigid poses/velocities and articulation coordinates.
  The test completes two environments at different reset boundaries and removes
  cached extras before the second reset, verifying its terminal evidence still
  survives. Host hooks passed.
- Blank-frame failures now save `blank_camera.png` and `blank_camera.json`
  (camera pose, current shot, frame/physics counters, rigid states and joints)
  before rejecting the recording. The guarded complete rollout is being repeated
  in `outputs/coffee_disposal_verified`; rejected earlier outputs remain separate.

- **Final guarded recording passed end-to-end** (process exit 0). Both
  environments completed all eleven stages and reset; each episode contains
  twenty-four predicate transition events and its own final physical state.
  All six views rendered through the previously failing interval and the
  pre-reset final overview. The original intermittent blackout was not isolated
  to a proven upstream cause; actual sensor warm-up, blank-frame rejection and
  diagnostic capture are retained. The delivered video is one complete rollout,
  not a splice of diagnostic footage.
- Final export verification decoded all three MP4s, checked resolution/frame
  rate/duration ratios, and scanned every scene frame for uniform blank output.
  All checks passed. Visual inspection confirmed the overview plan, machinery,
  open tray and drawer, tipping, return, retained grounds and restored scene.
  Maximum assisted translation tracking error was **3.10 mm**, maximum angular
  tracking error **0.01745 rad**. This measures tracking, not robot grasp skill.
- Final `pre-commit run --all-files` passed with all new files staged. Earlier
  independent task/collider reviews and the final recorder review are resolved;
  source footage, diagnostic media, generated MP4s and logs remain outside Git.
