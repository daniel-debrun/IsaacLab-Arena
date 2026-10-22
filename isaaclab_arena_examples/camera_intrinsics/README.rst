Independent camera intrinsics across environments
================================================

Enabling a camera intrinsics variation disables tiling for every camera in its
rig. ``ArenaCameraCfg.get_cfg()`` selects ``SeparateCameraRenderer`` automatically
when ``use_tiled_camera`` is false. No manual renderer assignment is required.

On the tested Isaac Lab revision, ``Camera`` and the deprecated ``TiledCamera``
both use the tiled RTX backend. Selecting ``CameraCfg`` alone therefore still
creates one tiled render product for all environments. Different reported camera
intrinsics do not produce independent rendered projections.

``SeparateCameraRenderer`` creates an ordinary render product and RGB annotator
for each environment, writing into slices of the existing batched output buffers.
It reuses Isaac Lab's stage setup, output conversion, and cleanup. The selection
applies to the entire rig, including cameras without an intrinsics variation.

Enable the variation
--------------------

For a DROID experiment, add the existing variation override:

.. code-block:: yaml

   variations:
     droid_abs_joint_pos.camera_intrinsics_wrist_camera.enabled: true

The variation's build-time hook calls ``camera_config.set_use_tiled_camera(False)``.
Direct callers can also use that method to disable tiling for the whole rig.
The copied rig preserves the renderer's quality and scene-partition settings;
the source camera configurations remain unchanged.

Untiled rendering currently supports the Isaac RTX backend and RGB/RGBA outputs.
Configuration rejects depth, segmentation, ISP, custom background colors, and
custom renderer implementations instead of silently dropping those settings.

Reproduce the captures
----------------------

Run these commands from the repository root inside the clone's development
container, as the host user. Choose fresh output directories. The force-exit
setting avoids the known Kit shutdown hang after the scripts finish.

.. code-block:: bash

   export ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1
   /isaac-sim/python.sh isaaclab_arena_examples/camera_intrinsics/capture_intrinsics.py \
     --visualizer none --enable_cameras --num_envs 1 \
     --capture-dir output/camera_intrinsics_review/single_captures
   /isaac-sim/python.sh isaaclab_arena_examples/camera_intrinsics/capture_intrinsics.py \
     --visualizer none --enable_cameras --num_envs 4 \
     --capture-dir output/camera_intrinsics_review/multi_captures
   /isaac-sim/python.sh isaaclab_arena_examples/camera_intrinsics/analyze_captures.py \
     --capture-root output/camera_intrinsics_review

The scripts capture a fixed scene while applying Arena's existing intrinsics
variation event. They also exercise ordinary resets with randomized intrinsics.
Image analysis estimates zoom using SIFT matches and an affine fit with RANSAC,
and checks that the camera poses stayed fixed.

The original wrist-only prototype measured scales for four environments as
``[0.6000, 1.0000, 1.2500, 1.5001]`` for requested scales
``[0.6, 1.0, 1.25, 1.5]``. Changing only environment 0 produced
``[1.5002, 1.0000, 1.2499, 1.5001]``. See
`image_analysis.json <results/image_analysis.json>`_. A separate normal rollout
completed eight zero-action episodes and produced 24 camera videos.
The generated images, videos, logs, and HDF5 datasets remain outside this snapshot.

Measure whole-rig rendering performance
--------------------------------------

Use a running OpenPI pi05 DROID joint-position server on ``127.0.0.1:8001``.
``benchmark.yaml`` declares that endpoint. The script uses normal Arena experiment
execution and wraps the rollout calls to collect timings. ``separate`` uses the
automatic selection for all cameras; ``tiled`` explicitly restores stock rendering
for all cameras to provide a comparison. It does not stop the
policy server.

.. code-block:: bash

   export ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1
   /isaac-sim/python.sh isaaclab_arena_examples/camera_intrinsics/benchmark.py \
     --visualizer none --enable_cameras --num_envs 4 --renderer tiled \
     --output-dir output/camera_intrinsics_review/performance_n4_tiled
   /isaac-sim/python.sh isaaclab_arena_examples/camera_intrinsics/benchmark.py \
     --visualizer none --enable_cameras --num_envs 4 --renderer separate \
     --output-dir output/camera_intrinsics_review/performance_n4_separate

Each process discards 60 warm-up steps and measures 180 vector steps. All three
cameras render RGB at 1280x720, with no viewport or recording. Intrinsics variation
is enabled with zero deltas in both cases for comparable projections. Simulation
runs at 120 Hz, rendering at 60 Hz, and control at 15 Hz. The six-second episode
timeout exercises resets during measurement. CUDA synchronization separates policy
and environment timing. Initialization is excluded; resets are included.

Whole-rig performance check
---------------------------

A fresh four-environment comparison with all three cameras untiled completed
successfully: 17.90 aggregate environment steps/s tiled versus 16.74 independent,
a 6.4% throughput reduction. Simulation time increased from 203.3 to 218.9 ms per
vector step; policy time remained about 19.7 ms. The independent run used four
render products per camera (12 total). Each mode ran once, with 60 warm-up steps
and 180 measured steps, so this is a bounded check rather than a repeated sweep.
See `whole_rig_performance.json <results/whole_rig_performance.json>`_.

Regression coverage also checks automatic rig-wide selection, preservation of
source configs and renderer settings, toggling back to tiled rendering,
unsupported settings, RGB/RGBA buffers, independent rendered zoom in two
environments, ordinary reset reuse, and cleanup.

Historical wrist-only performance
--------------------------------

The results below predate the rig-wide integration: only the wrist camera was
untiled, and both external cameras retained stock rendering. They do not measure
the cost of untiling every camera.

The recorded sweep ran counts 1, 2, 4, 8, tiled then separate, followed by counts
8, 4, 2, 1, separate then tiled, each in a fresh process. All 16 runs completed,
with 120 completed episodes and 2,880 measured vector steps. OpenPI requests are
sequential across environments and use a 15-step action horizon.

.. list-table:: Aggregate environment steps per second
   :header-rows: 1

   * - Environments
     - Tiled
     - Independent wrist
     - Throughput loss
   * - 1
     - 5.79
     - 5.74
     - 0.9%
   * - 2
     - 10.65
     - 10.37
     - 2.6%
   * - 4
     - 18.22
     - 17.30
     - 5.0%
   * - 8
     - 32.76
     - 31.62
     - 3.5%

.. image:: results/performance.png
   :alt: Throughput and simulation versus policy time for tiled and independent wrist rendering

See `performance.json <results/performance.json>`_ for both individual runs and
aggregates, and `performance.csv <results/performance.csv>`_ for the aggregate
metrics. Figure error bars span the two repeats, not confidence intervals.
Simulation timings include physics, all cameras, observations, and resets.
The short episodes benchmark throughput, not task success.

Measurement provenance
----------------------

* Date: 2026-10-22.
* Arena: ``dd94853f9a2ac31987c773e35c87d469c9e86ee8``.
* Isaac Lab: ``91a0906f93802eb859192dcb0bca879f891e92d6``.
* Isaac Sim: ``6.1.0-rc.26+release.49347.2d230af4.gl``.
* GPU: RTX 5090, 32 GB, driver 580.65.06; OpenPI shares the GPU.
* Another pre-existing process held approximately 612 MiB of GPU memory.
* Physics: PhysX; policy: OpenPI pi05 DROID joint positions.
* Seed and placement seed: 42; placement solving on reset disabled.

The recorded results above come from the original wrist-only experiment runs.
The current scripts exercise the rig-wide integration.

Remaining validation
--------------------

The original tiled failure was reproduced on the upgraded Isaac Lab / Isaac Sim
runtime, but the independent renderer has only been exercised on Sim 6.1.
Depth, segmentation, ISP, and other rendering backends remain unsupported.
