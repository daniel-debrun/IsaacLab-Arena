Independent wrist-camera rendering prototype
============================================

This draft preserves the working experiment for independent camera intrinsics in
parallel environments. It is not connected to Arena's camera-tiling flag yet.
The scripts explicitly select ``SeparateCameraRenderer`` for the wrist camera.

On the tested Isaac Lab revision, ``Camera`` and the deprecated ``TiledCamera``
both use the tiled RTX backend. Disabling Arena's tiling flag therefore still
creates one tiled render product for all environments. Different reported camera
intrinsics do not produce independent rendered projections.

``SeparateCameraRenderer`` creates an ordinary render product and RGB annotator
for each environment, writing into slices of the existing batched output buffers.
It reuses Isaac Lab's stage setup, output conversion, and cleanup.
Only the wrist camera uses this renderer; the external cameras keep stock rendering.

Manual opt-in
-------------

After constructing an ``ArenaEnvBuilder`` inside a running SimulationApp:

.. code-block:: python

   from isaaclab_arena_examples.camera_intrinsics.separate_camera_renderer import SeparateCameraRenderer

   env_cfg, env_kwargs = builder.build_env_cfg()
   env_cfg.scene.wrist_camera.renderer_cfg.class_type = SeparateCameraRenderer
   env = builder.make_registered(env_cfg, env_kwargs)

The prototype accepts RGB/RGBA only and rejects ISP and custom background colors.
RGB has been exercised; RGBA has not been separately validated.

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

Previously measured scales for four environments were
``[0.6000, 1.0000, 1.2500, 1.5001]`` for requested scales
``[0.6, 1.0, 1.25, 1.5]``. Changing only environment 0 produced
``[1.5002, 1.0000, 1.2499, 1.5001]``. See
`image_analysis.json <results/image_analysis.json>`_. A separate normal rollout
completed eight zero-action episodes and produced 24 camera videos.
The generated images, videos, logs, and HDF5 datasets remain outside this snapshot.

Reproduce a performance comparison
----------------------------------

Use a running OpenPI pi05 DROID joint-position server on ``127.0.0.1:8001``.
``benchmark.yaml`` declares that endpoint. The script uses normal Arena experiment
execution and wraps the rollout calls to collect timings. It does not stop the
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

The scripts were relocated from the local experiment output folder for this
snapshot. Imports and configuration paths were adjusted; the renderer's behavior
is unchanged. The recorded results above come from the original experiment runs.

Remaining integration work
--------------------------

* Wire the existing tiling flag to the independent renderer.
* Add regression coverage for independent rendered intrinsics across environments.
* Decide the supported camera-output and backend scope, including explicit handling
  of unsupported configurations.
* Validate against the upgraded Isaac Lab / Isaac Sim runtime. The original tiled
  failure was reproduced there, but this renderer has only been tested on Sim 6.1.
