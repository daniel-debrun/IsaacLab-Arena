# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AssetSelectionVariation, on its own and through ArenaEnvBuilder.build_env_cfg()."""

import json

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True

TABLE_TOP_Z = 0.1
ON_CLEARANCE_M = 0.01
NUM_ENVS = 2
DEFAULT_REGISTRY_NAME = "cracker_box"
SELECTION_KEY = "pickup.asset_selection"


def get_registered_usd_path(registry_name):
    """Return the USD path of the registered object ``registry_name``."""
    from isaaclab_arena.assets.registries import AssetRegistry

    return AssetRegistry().get_asset_by_name(registry_name).usd_path


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

    return AssetRegistry().get_asset_by_name(DEFAULT_REGISTRY_NAME)(name="pickup")


def compose_pick_and_place_with_selection(
    *,
    selection_cfg=None,
    seed=42,
    rebuild_index=0,
    hydra_overrides=None,
    recorded_variation_samples_path=None,
    solve_relations=False,
    task_description=None,
):
    """Compose a PickAndPlaceTask environment whose pickup, a cracker box on a table, has an asset selection.

    Args:
        selection_cfg: Settings applied to the pickup's asset selection. None keeps the disabled default.
        seed: Build seed.
        rebuild_index: Index of the build among the rebuilds of an Experiment Run.
        hydra_overrides: Variation overrides for the build.
        recorded_variation_samples_path: Record to replay, if any.
        solve_relations: Whether to solve the pickup's placement on the table.
        task_description: Description passed to the task. None keeps its default description.

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
    task = PickAndPlaceTask(pickup, plate, SimpleNamespace(object_min_z=-1.0), task_description=task_description)
    scene = Scene(assets=[table, pickup, plate])
    arena_env = IsaacLabArenaEnvironment(name="test_asset_selection_variation", scene=scene, task=task)
    # resolve_on_reset=False stores the solved pose per environment on the object. Only the config is composed.
    builder_cfg = ArenaEnvBuilderCfg(
        num_envs=NUM_ENVS,
        seed=seed,
        solve_relations=solve_relations,
        placement_seed=0,
        resolve_on_reset=False,
        recorded_variation_samples_path=recorded_variation_samples_path,
    )
    arena_builder = ArenaEnvBuilder(
        arena_env, builder_cfg, hydra_overrides=hydra_overrides, rebuild_index=rebuild_index
    )
    env_cfg, env_kwargs = arena_builder.build_env_cfg()
    return pickup, env_cfg, env_kwargs


def _test_selected_asset_is_spawned_placed_recorded_and_described(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.variations_hydra import overrides_from_dict

    hydra_overrides = overrides_from_dict(
        {f"{SELECTION_KEY}.enabled": True, f"{SELECTION_KEY}.registry_names": ["sugar_box"]}
    )

    pickup, env_cfg, env_kwargs = compose_pick_and_place_with_selection(
        hydra_overrides=hydra_overrides, solve_relations=True
    )

    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path("sugar_box")
    # The center of the sugar box rests 13 mm lower than the center of the cracker box would.
    sugar_box_bottom_z = AssetRegistry().get_asset_by_name("sugar_box")().get_bounding_box().min_point[0, 2].item()
    expected_z = TABLE_TOP_Z + ON_CLEARANCE_M - sugar_box_bottom_z
    placed_z = [pose.position_xyz[2] for pose in pickup.get_initial_pose().poses]
    assert placed_z == pytest.approx([expected_z] * NUM_ENVS, abs=1e-3)
    selection_record = env_kwargs["variation_recorder"][SELECTION_KEY]
    recorded_names = [selection_record.sample_for_episode(env_id, 0) for env_id in range(NUM_ENVS)]
    assert recorded_names == ["sugar_box"] * NUM_ENVS
    # The pickup keeps the name of its role, and its registry name and the description name the selected asset.
    assert pickup.name == "pickup"
    assert pickup.registry_name == "sugar_box"
    # The selection sets registry_name on the object only, so its class keeps the declared registry name.
    assert type(pickup).registry_name == DEFAULT_REGISTRY_NAME
    assert env_cfg.task_description == "Pick up the sugar_box, and place it into the plate"
    return True


def _test_disabled_selection_keeps_default_asset(simulation_app):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    _, env_cfg, _ = compose_pick_and_place_with_selection(
        selection_cfg=AssetSelectionVariationCfg(enabled=False, registry_names=["sugar_box", "mustard_bottle"])
    )

    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path(DEFAULT_REGISTRY_NAME)
    assert env_cfg.task_description == "Pick up the cracker_box, and place it into the plate"
    return True


def _test_pick_and_place_keeps_passed_task_description(simulation_app):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    _, env_cfg, _ = compose_pick_and_place_with_selection(
        selection_cfg=AssetSelectionVariationCfg(enabled=True, registry_names=["sugar_box"]),
        task_description="Put the box on the plate",
    )

    assert env_cfg.task_description == "Put the box on the plate"
    return True


def _test_in_order_selection_follows_rebuild_index(simulation_app):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    registry_names = ["mustard_bottle", "sugar_box", "tomato_soup_can"]

    def compose_in_order_selection(seed, rebuild_index):
        _, env_cfg, _ = compose_pick_and_place_with_selection(
            selection_cfg=AssetSelectionVariationCfg(enabled=True, registry_names=registry_names),
            seed=seed,
            rebuild_index=rebuild_index,
        )
        return env_cfg.scene.pickup.spawn.usd_path

    # Rebuild k takes name k % 3, so the fourth rebuild starts over at the first name.
    usd_paths_over_rebuilds = [compose_in_order_selection(seed=42, rebuild_index=k) for k in range(4)]
    expected_registry_names = registry_names + registry_names[:1]
    assert usd_paths_over_rebuilds == [
        get_registered_usd_path(registry_name) for registry_name in expected_registry_names
    ]
    # Another seed takes the same name at the same rebuild index.
    assert compose_in_order_selection(seed=44, rebuild_index=1) == usd_paths_over_rebuilds[1]
    return True


def _test_random_selection_draws_from_seeded_generator(simulation_app):
    import torch

    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    def compose_random_selection(seed):
        selection_cfg = AssetSelectionVariationCfg(
            enabled=True, random_choice=True, registry_names=["mustard_bottle", "sugar_box", "tomato_soup_can"]
        )
        _, env_cfg, _ = compose_pick_and_place_with_selection(selection_cfg=selection_cfg, seed=seed)
        return env_cfg.scene.pickup.spawn.usd_path

    # A draw from torch's global random state would differ between these two builds.
    torch.manual_seed(0)
    first_usd_path = compose_random_selection(seed=7)
    torch.manual_seed(2)
    second_usd_path = compose_random_selection(seed=7)
    assert first_usd_path == second_usd_path

    usd_paths_over_seeds = {compose_random_selection(seed) for seed in range(5)}
    assert len(usd_paths_over_seeds) > 1
    return True


def _test_recorded_registry_name_is_replayed(simulation_app, tmp_path):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    recorded_samples_path = tmp_path / "asset_selection_replay.jsonl"
    recorded_row = json.dumps({"variations": {SELECTION_KEY: "mustard_bottle"}})
    recorded_samples_path.write_text(f"{recorded_row}\n" * NUM_ENVS)
    # In order, a live draw in the first rebuild takes "sugar_box".
    _, env_cfg, _ = compose_pick_and_place_with_selection(
        selection_cfg=AssetSelectionVariationCfg(enabled=True, registry_names=["sugar_box", "mustard_bottle"]),
        recorded_variation_samples_path=str(recorded_samples_path),
    )

    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path("mustard_bottle")
    return True


def _test_asset_registered_by_factory_is_selected(simulation_app):
    from unittest.mock import patch

    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.register import register_asset_factory
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    factory_registry_name = "sugar_box_from_factory"
    asset_registry = AssetRegistry()
    # The lookup registers every library asset first, so the registry patched below already holds them.
    sugar_box_class = asset_registry.get_asset_by_name("sugar_box")
    # patch.dict removes the factory afterwards, so later tests see the registry unchanged.
    with patch.dict(asset_registry._components):

        @register_asset_factory(registry_name=factory_registry_name, object_type=ObjectType.RIGID)
        def make_sugar_box():
            return sugar_box_class()

        pickup, env_cfg, _ = compose_pick_and_place_with_selection(
            selection_cfg=AssetSelectionVariationCfg(enabled=True, registry_names=[factory_registry_name])
        )

    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path("sugar_box")
    assert pickup.registry_name == factory_registry_name
    return True


def _test_asset_selection_rejects_invalid_configuration(simulation_app, case_id):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariationCfg

    def build_with_registry_names(registry_names):
        compose_pick_and_place_with_selection(
            selection_cfg=AssetSelectionVariationCfg(enabled=True, registry_names=registry_names)
        )

    def build_with_bounding_box_override():
        pickup = make_pickup()
        pickup.get_variation("asset_selection").apply_cfg(
            AssetSelectionVariationCfg(enabled=True, registry_names=["sugar_box"])
        )
        pickup.bounding_box = AxisAlignedBoundingBox(min_point=(0.0, 0.0, 0.0), max_point=(0.1, 0.1, 0.1))
        arena_env = IsaacLabArenaEnvironment(name="bounding_box_override", scene=Scene(assets=[pickup]))
        ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1)).build_env_cfg()

    invalid_configurations = {
        "empty_registry_names": (lambda: build_with_registry_names([]), "'pickup.asset_selection' is enabled but its"),
        "unknown_name": (lambda: build_with_registry_names(["sugar_box", "banana_split"]), "'banana_split' is not a"),
        "non_rigid_name": (
            lambda: build_with_registry_names(["ground_plane"]),
            r"'ground_plane' is not a rigid object \(object type: base\)",
        ),
        "non_object_name": (
            lambda: build_with_registry_names(["droid_differential_ik"]),
            r"'droid_differential_ik' is not a rigid object \(object type: None\)",
        ),
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

    def build_replaying(record_row, record_name):
        """Return the env_cfg of a scene holding only the pickup, with its selection on, that replays ``record_row``."""
        pickup = make_pickup()
        pickup.get_variation("asset_selection").apply_cfg(
            AssetSelectionVariationCfg(enabled=True, registry_names=["mustard_bottle", "sugar_box"])
        )
        record_path = tmp_path / f"{record_name}.jsonl"
        record_path.write_text(json.dumps(record_row) + "\n")
        arena_env = IsaacLabArenaEnvironment(name=record_name, scene=Scene(assets=[pickup]))
        builder_cfg = ArenaEnvBuilderCfg(num_envs=1, recorded_variation_samples_path=str(record_path))
        env_cfg, _ = ArenaEnvBuilder(arena_env, builder_cfg).build_env_cfg()
        return env_cfg

    # One layout row as write_placement_samples writes it: a pose for pickup and no variation values.
    pickup_pose = {"position_xyz": [0.0, 0.0, 0.05], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    layout_row = {"placement": {"layout_id": "layout_000000", "poses": {"pickup": pickup_pose}}}
    with pytest.raises(AssertionError, match=r"no 'pickup\.asset_selection' value"):
        build_replaying(layout_row, "layout_without_selection")

    # In order, a live draw in the first rebuild would take "mustard_bottle".
    layout_row_with_selection = {**layout_row, "variations": {SELECTION_KEY: "sugar_box"}}
    env_cfg = build_replaying(layout_row_with_selection, "layout_with_selection")
    assert env_cfg.scene.pickup.spawn.usd_path == get_registered_usd_path("sugar_box")
    return True


def _test_registered_rigid_objects_have_disabled_asset_selection(simulation_app):
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry

    pickup = make_pickup()
    assert not pickup.get_variation("asset_selection").enabled
    # A registered library object that is not rigid has no selection.
    ground_plane = AssetRegistry().get_asset_by_name("ground_plane")()
    assert ground_plane.object_type == ObjectType.BASE
    assert "asset_selection" not in ground_plane.variations
    # Objects that are not library objects have no selection either.
    cube = Object(name="cube", object_type=ObjectType.RIGID, spawner_cfg=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)))
    assert "asset_selection" not in cube.variations
    pickup_set = RigidObjectSet(name="pickup_set", objects=[pickup])
    assert "asset_selection" not in pickup_set.variations
    return True


def test_selected_asset_is_spawned_placed_recorded_and_described():
    assert run_function_with_persistent_simulation_app(
        _test_selected_asset_is_spawned_placed_recorded_and_described, headless=HEADLESS
    )


def test_disabled_selection_keeps_default_asset():
    assert run_function_with_persistent_simulation_app(_test_disabled_selection_keeps_default_asset, headless=HEADLESS)


def test_pick_and_place_keeps_passed_task_description():
    assert run_function_with_persistent_simulation_app(
        _test_pick_and_place_keeps_passed_task_description, headless=HEADLESS
    )


def test_in_order_selection_follows_rebuild_index():
    assert run_function_with_persistent_simulation_app(
        _test_in_order_selection_follows_rebuild_index, headless=HEADLESS
    )


def test_random_selection_draws_from_seeded_generator():
    assert run_function_with_persistent_simulation_app(
        _test_random_selection_draws_from_seeded_generator, headless=HEADLESS
    )


def test_recorded_registry_name_is_replayed(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_recorded_registry_name_is_replayed, headless=HEADLESS, tmp_path=tmp_path
    )


def test_asset_registered_by_factory_is_selected():
    assert run_function_with_persistent_simulation_app(_test_asset_registered_by_factory_is_selected, headless=HEADLESS)


@pytest.mark.parametrize(
    "case_id",
    [
        "empty_registry_names",
        "unknown_name",
        "non_rigid_name",
        "non_object_name",
        "bounding_box_override",
    ],
)
def test_asset_selection_rejects_invalid_configuration(case_id):
    assert run_function_with_persistent_simulation_app(
        _test_asset_selection_rejects_invalid_configuration, headless=HEADLESS, case_id=case_id
    )


def test_placement_replay_needs_recorded_asset_selection(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_placement_replay_needs_recorded_asset_selection, headless=HEADLESS, tmp_path=tmp_path
    )


def test_registered_rigid_objects_have_disabled_asset_selection():
    assert run_function_with_persistent_simulation_app(
        _test_registered_rigid_objects_have_disabled_asset_selection, headless=HEADLESS
    )


class _RecordingRigidObject:
    """Stands in for the Object whose asset a selection sets, and records each selected registry name."""

    def __init__(self):
        self.name = "pickup"
        self.bounding_box = None
        self.selected_registry_names: list[str] = []

    def set_asset(self, registry_name, spawn_cfg):
        self.selected_registry_names.append(registry_name)


@pytest.fixture
def resolved_registry_names(monkeypatch):
    """Resolve registry names without building the registered assets, and return the names resolved so far.

    Building the registered assets needs a SimulationApp, so the tests that use this fixture run in plain Python.
    """
    from isaaclab_arena.variations import asset_selection_variation

    resolved_registry_names = []

    def resolve_spawn_cfg(registry_name):
        resolved_registry_names.append(registry_name)
        return f"spawn cfg of {registry_name}"

    monkeypatch.setattr(asset_selection_variation, "_get_rigid_object_spawn_cfg", resolve_spawn_cfg)
    return resolved_registry_names


@pytest.mark.parametrize("random_choice", [False, True])
def test_random_choice_set_without_apply_cfg_fails_the_build(resolved_registry_names, random_choice):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    selection_cfg = AssetSelectionVariationCfg(
        enabled=True, random_choice=random_choice, registry_names=["mustard_bottle", "sugar_box"]
    )
    variation = AssetSelectionVariation(_RecordingRigidObject(), selection_cfg)
    # The sampler bound by apply_cfg stays, so it no longer matches random_choice.
    variation.cfg.random_choice = not random_choice

    with pytest.raises(AssertionError, match="Change random_choice by passing a cfg to apply_cfg"):
        variation.configure_at_build_time(seed=0, variation_key=SELECTION_KEY)


def test_random_choice_override_swaps_the_sampler(resolved_registry_names):
    from isaaclab_arena.variations import variations_hydra
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation
    from isaaclab_arena.variations.choice_sampler import ChoiceSampler
    from isaaclab_arena.variations.sequential_sampler import SequentialSampler

    rigid_object = _RecordingRigidObject()
    variation = AssetSelectionVariation(rigid_object)
    variations = {"pickup": [variation]}
    notified_registry_names = []
    variation.add_sample_listener(lambda sample, _env_ids: notified_registry_names.extend(sample))

    variations_hydra.apply_overrides(
        variations,
        [
            f"{SELECTION_KEY}.enabled=true",
            f"{SELECTION_KEY}.random_choice=true",
            f"{SELECTION_KEY}.registry_names=[mustard_bottle,sugar_box]",
        ],
    )
    assert isinstance(variation.sampler, ChoiceSampler)

    variations_hydra.apply_overrides(variations, [f"{SELECTION_KEY}.random_choice=false"])
    assert isinstance(variation.sampler, SequentialSampler)
    # Rebuild 1 takes the second name, and the listener added before both swaps is notified of it.
    variation.configure_at_build_time(seed=0, variation_key=SELECTION_KEY, rebuild_index=1)
    assert rigid_object.selected_registry_names == ["sugar_box"]
    assert notified_registry_names == ["sugar_box"]


def test_in_order_selection_takes_repeated_names_in_list_order(resolved_registry_names):
    from isaaclab_arena.variations.asset_selection_variation import AssetSelectionVariation, AssetSelectionVariationCfg

    registry_names = ["mustard_bottle", "sugar_box", "mustard_bottle", "tomato_soup_can"]
    rigid_object = _RecordingRigidObject()
    variation = AssetSelectionVariation(
        rigid_object, AssetSelectionVariationCfg(enabled=True, registry_names=registry_names)
    )

    num_rebuilds = 6
    for rebuild_index in range(num_rebuilds):
        variation.configure_at_build_time(seed=0, variation_key=SELECTION_KEY, rebuild_index=rebuild_index)

    # Rebuild k takes entry k % 4, so mustard_bottle spawns in rebuilds 0, 2 and 4.
    assert rigid_object.selected_registry_names == registry_names + registry_names[:2]
    # Each build resolves the repeated name once.
    assert resolved_registry_names == ["mustard_bottle", "sugar_box", "tomato_soup_can"] * num_rebuilds
