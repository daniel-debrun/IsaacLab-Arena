# Coffee Disposal Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the isolated asset task and review; parent integrates runtime/task/demo. Steps below are checkable deliverables.

**Goal:** Reproduce the reference's complete spent-grounds disposal and restoration sequence in Arena, with realistic local assets and multi-angle videos.
**Architecture:** Local USD props and articulations, registered Arena environment, sequential state predicates, optional force-assisted demo, explicit renderer/FFmpeg recording.
**Tech stack:** Python, USD, Isaac Lab/Arena, PhysX, Torch, Pillow, FFmpeg.
**Spec:** `COFFEE_DISPOSAL_FINDINGS.md`; reference evidence in `COFFEE_REFERENCE_ANALYSIS.md`.

## Global constraints

- Work only in `.worktrees/coffee-disposal`; no changes to the PC task or parent checkout.
- User authorized independent decisions; no questions or approval pauses for routine work.
- Eleven observed stages, including spout raising, removable bin lid and final restoration.
- Ordinary policy actions stay unassisted; demo assistance is labelled.
- Physics 120 Hz, control 15 Hz, video 30 fps. Use xyzw quaternions.
- Do not commit source footage, extracted reference frames, or generated videos.
- Keep findings, asset provenance, reproducible commands and usage counters.

## Review focus

- Pre-positioned final objects cannot bypass extraction, lifting, tipping or replacement.
- Grounds must actually enter the organic bin; spilling elsewhere is failure.
- Returned tub/lid must be aligned and supported; final retention is rechecked.
- Partial reset must clear only selected task histories and demo forces/drives.
- Camera cuts cannot advance physics or reset the scene shown at completion.

## Tasks

- [x] Analyze all source-video stages and dense samples around brief actions. Save timestamps and uncertainties.
- [x] Author `coffee_disposal/layout.py` as the shared geometry/pose contract.
- [x] Build `coffee_disposal/build_assets.py` and `assets/`: custom machine, kitchen/waste drawer, grounds tub, lid and grounds. Audit clearances, joint axes, colliders, masses and exported USDA. Asset author owns these files only.
- [x] Write failing task tests in `isaaclab_arena/tests/test_coffee_disposal_task.py`: complete ordered trace, wrong bin/spilled clumps, fake final poses, missing tip, retention loss and partial reset. Implement `coffee_disposal/task.py` against those behaviors.
- [x] Implement registered `coffee_disposal_environment.py` and `coffee_disposal/cameras.py`, local assets, passive ordinary joints, observation state and reset events. Add packaged asset globs in `pyproject.toml`.
- [x] Implement `isaaclab_arena_examples/coffee_disposal/motion.py` and `demo.py`: smooth forces, independent named joint drives, real tipping/release, telemetry and indexed reset. Reuse the validated minimum-jerk approach without changing PC code.
- [x] Run initial smoke/render, fix measured geometry/physics issues, then complete the ordered task in two environments. Add meaningful live-reset coverage for assistance.
- [x] Implement `video.py`: readable pre-task overview, contextual camera cuts, 30 fps capture and 1×/2×/5× output. Inspect key frames and decode every exported MP4.
- [x] Document provenance/run commands, save validation evidence and usage snapshots. Run host hooks, package asset audit and independent whole-change review; address concrete findings.
