# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that relation placement in ArenaEnvBuilder sees the changes build-time variations make."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

HEADLESS = True

TABLE_TOP_Z = 0.1
ON_CLEARANCE_M = 0.01
BUILD_TIME_SPHERE_RADIUS = 0.42


class SphereRadiusVariation(BuildTimeVariationBase):
    """Build-time variation that sets its sphere's radius to BUILD_TIME_SPHERE_RADIUS."""

    def __init__(self, asset, name: str = "sphere_radius"):
        sampler_cfg = UniformSamplerCfg(low=[BUILD_TIME_SPHERE_RADIUS], high=[BUILD_TIME_SPHERE_RADIUS])
        super().__init__(cfg=VariationBaseCfg(sampler_cfg=sampler_cfg), name=name)
        self._asset = asset

    def _realize_at_build_time(self) -> None:
        self._asset.object_cfg.spawn.radius = float(self.sampler.sample(num_samples=1)[0, 0])


def _test_placement_uses_geometry_set_by_build_time_variation(simulation_app):
    """Relation placement puts the sphere on the table with the radius the build-time variation set."""
    import isaaclab.sim as sim_utils

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    # A fixed 2 m by 2 m table, 0.1 m thick, with its top at TABLE_TOP_Z.
    table = Object(
        name="table",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.CuboidCfg(size=(2.0, 2.0, 0.1)),
        initial_pose=Pose(position_xyz=(0.0, 0.0, 0.05), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        relations=[IsAnchor()],
    )
    sphere = Object(
        name="sphere",
        object_type=ObjectType.RIGID,
        spawner_cfg=sim_utils.SphereCfg(radius=0.1, rigid_props=sim_utils.RigidBodyPropertiesCfg()),
        relations=[On(table, clearance_m=ON_CLEARANCE_M)],
    )
    variation = SphereRadiusVariation(sphere)
    variation.enable()
    sphere.add_variation(variation)
    arena_env = IsaacLabArenaEnvironment(
        name="test_placement_after_build_time_variations",
        scene=Scene(assets=[table, sphere]),
    )
    # resolve_on_reset=False stores the solved pose per environment on the object. Only the config is composed.
    builder_cfg = ArenaEnvBuilderCfg(num_envs=1, placement_seed=0, resolve_on_reset=False)
    ArenaEnvBuilder(arena_env, builder_cfg).compose_manager_cfg()

    sphere_z = sphere.get_initial_pose().poses[0].position_xyz[2]
    expected_z = TABLE_TOP_Z + ON_CLEARANCE_M + BUILD_TIME_SPHERE_RADIUS
    assert sphere_z == pytest.approx(expected_z, abs=1e-3), (
        f"Sphere placed at z={sphere_z:.4f}, expected {expected_z:.4f}: "
        "placement must use the geometry left by build-time variations."
    )
    return True


def test_placement_uses_geometry_set_by_build_time_variation():
    assert run_function_with_persistent_simulation_app(
        _test_placement_uses_geometry_set_by_build_time_variation,
        headless=HEADLESS,
    )
