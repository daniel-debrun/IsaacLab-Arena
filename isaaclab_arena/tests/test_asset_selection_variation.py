# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AssetSelectionVariation, through ArenaEnvBuilder.build_env_cfg()."""

import json

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True

TABLE_TOP_Z = 0.1
ON_CLEARANCE_M = 0.01
NUM_ENVS = 2
DEFAULT_ASSET_NAME = "cracker_box"
SELECTION_KEY = "pickup.asset_selection"


def get_registered_usd_path(asset_name):
    """Return the USD path of the registered object ``asset_name``."""
    from isaaclab_arena.assets.registries import AssetRegistry

    return AssetRegistry().get_asset_by_name(asset_name).usd_path


def make_table():
    """Return a fixed 2 m by 2 m table, 0.1 m thick, with its top at TABLE_TOP_Z."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.relations.relations import IsAnchor
    from isaaclab_arena.utils.pose import Pose

    return Object(
        name="table",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(size=(2.0, 2.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg()),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        relations=[IsAnchor()],
    )


def make_pickup():
    """Return the registered cracker box, named ``pickup``."""
    from isaaclab_arena.assets.registries import AssetRegistry

    return AssetRegistry().get_asset_by_name(DEFAULT_ASSET_NAME)(instance_name="pickup")


def compose_pick_and_place_with_selection(
    *,
    selection_cfg=None,
    seed=42,
    hydra_overrides=None,
    recorded_variation_samples_path=None,
):
    """Compose a PickAndPlaceTask environment whose pickup, a cracker box on a table, has an asset selection.

    Args:
        selection_cfg: Settings applied to the pickup's asset selection. None keeps the disabled default.
        seed: Build seed.
        hydra_overrides: Variation overrides for the build.
        recorded_variation_samples_path: Record to replay, if any.

    Returns:
        The pickup Object, the composed env_cfg and the env_kwargs.
    """
    from types import SimpleNamespace

    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
    from isaaclab_arena.utils.pose import Pose

    table = make_table()
    pickup = make_pickup()
    pickup.add_relation(On(table, clearance_m=ON_CLEARANCE_M))
    if selection_cfg is not None:
        pickup.get_variation("asset_selection").apply_cfg(selection_cfg)
    plate_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    plate_pose = Pose(position_xyz=(0.7, 0.7, 0.11), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    plate = Object(name="plate", object_type=ObjectType.RIGID, spawner_cfg=plate_spawn_cfg, initial_pose=plate_pose)
    # The namespace stands in for a background; the task reads only its object_min_z.
    task = PickAndPlaceTask(pickup, plate, SimpleNamespace(object_min_z=-1.0))
    scene = Scene(assets=[table, pickup, plate])
    arena_env = IsaacLabArenaEnvironment(name="test_asset_selection_variation", scene=scene, task=task)
    # resolve_on_reset=False stores the solved pose per environment on the object. Only the config is composed.
    builder_cfg = ArenaEnvBuilderCfg(
        num_envs=NUM_ENVS,
        seed=seed,
        placement_seed=0,
        resolve_on_reset=False,
        recorded_variation_samples_path=recorded_variation_samples_path,
    )
    env_cfg, env_kwargs = ArenaEnvBuilder(arena_env, builder_cfg, hydra_overrides=hydra_overrides).build_env_cfg()
    return pickup, env_cfg, env_kwargs


def _test_selected_asset_is_spawned_placed_and_recorded(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.variations_hydra import overrides_from_dict

    hydra_overrides = overrides_from_dict(
        {f"{SELECTION_KEY}.enabled": True, f"{SELECTION_KEY}.asset_names": ["sugar_box"]}
    )

    pickup, env_cfg, env_kwargs = compose_pick_and_place_with_selection(hydra_overrides=hydra_overrides)

    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path("sugar_box")
    # The center of the sugar box rests 13 mm lower than the center of the cracker box would.
    sugar_box_bottom_z = AssetRegistry().get_asset_by_name("sugar_box")().get_bounding_box().min_point[0, 2].item()
    expected_z = TABLE_TOP_Z + ON_CLEARANCE_M - sugar_box_bottom_z
    placed_z = [pose.position_xyz[2] for pose in pickup.get_initial_pose().poses]
    assert placed_z == pytest.approx([expected_z] * NUM_ENVS, abs=1e-3)
    selection_record = env_kwargs["variation_recorder"][SELECTION_KEY]
    recorded_names = [selection_record.sample_for_episode(env_id, 0) for env_id in range(NUM_ENVS)]
    assert recorded_names == ["sugar_box"] * NUM_ENVS
    return True


def _test_disabled_selection_keeps_default_asset(simulation_app):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    _, env_cfg, _ = compose_pick_and_place_with_selection(
        selection_cfg=AssetSelectionVariationCfg(enabled=False, asset_names=["sugar_box", "mustard_bottle"])
    )

    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path(DEFAULT_ASSET_NAME)
    return True


def _test_in_order_selection_follows_build_seed(simulation_app):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    asset_names = ["mustard_bottle", "sugar_box", "tomato_soup_can"]
    spawned_usd_paths = []
    for seed in (42, 43, 44):
        _, env_cfg, _ = compose_pick_and_place_with_selection(
            selection_cfg=AssetSelectionVariationCfg(enabled=True, asset_names=asset_names), seed=seed
        )
        spawned_usd_paths.append(env_cfg.scene.pickup.spawn.usd_path)

    # 42 % 3 == 0, so seed 42 takes the first of asset_names and each next seed the next one.
    assert spawned_usd_paths == [get_registered_usd_path(asset_name) for asset_name in asset_names]
    return True


def _test_random_selection_draws_from_seeded_generator(simulation_app):
    import torch

    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    def compose_random_selection(seed):
        selection_cfg = AssetSelectionVariationCfg(
            enabled=True, random_choice=True, asset_names=["mustard_bottle", "sugar_box", "tomato_soup_can"]
        )
        _, env_cfg, _ = compose_pick_and_place_with_selection(selection_cfg=selection_cfg, seed=seed)
        return env_cfg.scene.pickup.spawn.usd_path

    # A draw from torch's global random state would differ between these two builds.
    torch.manual_seed(0)
    first_usd_path = compose_random_selection(seed=7)
    torch.manual_seed(2)
    second_usd_path = compose_random_selection(seed=7)
    assert first_usd_path == second_usd_path

    # These seeds share seed % 3, so in-order selection would take one name for all of them.
    usd_paths_over_seeds = {compose_random_selection(seed) for seed in range(0, 15, 3)}
    assert len(usd_paths_over_seeds) > 1
    return True


def _test_recorded_asset_name_is_replayed(simulation_app, tmp_path):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    recorded_samples_path = tmp_path / "asset_selection_replay.jsonl"
    recorded_row = json.dumps({"variations": {SELECTION_KEY: "mustard_bottle"}})
    recorded_samples_path.write_text(f"{recorded_row}\n" * NUM_ENVS)
    # In order with seed 42, a live draw takes "sugar_box" (42 % 2 == 0).
    _, env_cfg, _ = compose_pick_and_place_with_selection(
        selection_cfg=AssetSelectionVariationCfg(enabled=True, asset_names=["sugar_box", "mustard_bottle"]),
        seed=42,
        recorded_variation_samples_path=str(recorded_samples_path),
    )

    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path("mustard_bottle")
    return True


def _test_asset_selection_rejects_invalid_configuration(simulation_app, case_id):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    def build_with_asset_names(asset_names):
        compose_pick_and_place_with_selection(
            selection_cfg=AssetSelectionVariationCfg(enabled=True, asset_names=asset_names)
        )

    def repeat_name():
        make_pickup().get_variation("asset_selection").apply_cfg(
            AssetSelectionVariationCfg(asset_names=["sugar_box", "sugar_box"])
        )

    def build_with_bounding_box_override():
        pickup = make_pickup()
        pickup.get_variation("asset_selection").apply_cfg(
            AssetSelectionVariationCfg(enabled=True, asset_names=["sugar_box"])
        )
        pickup.bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
        arena_env = IsaacLabArenaEnvironment(name="bounding_box_override", scene=Scene(assets=[pickup]))
        ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1)).build_env_cfg()

    invalid_configurations = {
        "empty_asset_names": (lambda: build_with_asset_names([]), "'pickup.asset_selection' is enabled but its"),
        "unknown_name": (lambda: build_with_asset_names(["sugar_box", "banana_split"]), "'banana_split' is not a"),
        "non_rigid_name": (lambda: build_with_asset_names(["ground_plane"]), "'ground_plane' is a base Object"),
        "repeated_name": (repeat_name, r"'pickup\.asset_selection\.asset_names' repeats a name"),
        "bounding_box_override": (build_with_bounding_box_override, "'pickup' sets a bounding_box override"),
    }
    build_invalid_configuration, message_pattern = invalid_configurations[case_id]
    with pytest.raises(AssertionError, match=message_pattern):
        build_invalid_configuration()
    return True


def _test_placement_replay_needs_recorded_asset_selection(simulation_app, tmp_path):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    pickup = make_pickup()
    pickup.get_variation("asset_selection").apply_cfg(
        AssetSelectionVariationCfg(enabled=True, asset_names=["sugar_box"])
    )
    # One layout row as write_placement_samples writes it: a pose for pickup and no variation values.
    pickup_pose = {"position_xyz": [0.0, 0.0, 0.05], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    layout_row = {"placement": {"layout_id": "layout_000000", "poses": {"pickup": pickup_pose}}}
    layouts_path = tmp_path / "layouts.jsonl"
    layouts_path.write_text(json.dumps(layout_row) + "\n")
    arena_env = IsaacLabArenaEnvironment(name="placement_replay_without_selection", scene=Scene(assets=[pickup]))
    builder_cfg = ArenaEnvBuilderCfg(num_envs=1, recorded_variation_samples_path=str(layouts_path))

    with pytest.raises(AssertionError, match=r"no 'pickup\.asset_selection' value"):
        ArenaEnvBuilder(arena_env, builder_cfg).build_env_cfg()
    return True


def _test_rigid_objects_outside_sets_have_disabled_asset_selection(simulation_app):
    from isaaclab_arena.assets.object_set import RigidObjectSet

    pickup = make_pickup()
    assert not pickup.get_variation("asset_selection").enabled
    assert "asset_selection" not in make_table().variations
    pickup_set = RigidObjectSet(name="pickup_set", objects=[pickup])
    assert "asset_selection" not in pickup_set.variations
    return True


def test_selected_asset_is_spawned_placed_and_recorded():
    assert run_function_with_persistent_simulation_app(
        _test_selected_asset_is_spawned_placed_and_recorded, headless=HEADLESS
    )


def test_disabled_selection_keeps_default_asset():
    assert run_function_with_persistent_simulation_app(_test_disabled_selection_keeps_default_asset, headless=HEADLESS)


def test_in_order_selection_follows_build_seed():
    assert run_function_with_persistent_simulation_app(_test_in_order_selection_follows_build_seed, headless=HEADLESS)


def test_random_selection_draws_from_seeded_generator():
    assert run_function_with_persistent_simulation_app(
        _test_random_selection_draws_from_seeded_generator, headless=HEADLESS
    )


def test_recorded_asset_name_is_replayed(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_recorded_asset_name_is_replayed, headless=HEADLESS, tmp_path=tmp_path
    )


@pytest.mark.parametrize(
    "case_id", ["empty_asset_names", "unknown_name", "non_rigid_name", "repeated_name", "bounding_box_override"]
)
def test_asset_selection_rejects_invalid_configuration(case_id):
    assert run_function_with_persistent_simulation_app(
        _test_asset_selection_rejects_invalid_configuration, headless=HEADLESS, case_id=case_id
    )


def test_placement_replay_needs_recorded_asset_selection(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_placement_replay_needs_recorded_asset_selection, headless=HEADLESS, tmp_path=tmp_path
    )


def test_rigid_objects_outside_sets_have_disabled_asset_selection():
    assert run_function_with_persistent_simulation_app(
        _test_rigid_objects_outside_sets_have_disabled_asset_selection, headless=HEADLESS
    )
