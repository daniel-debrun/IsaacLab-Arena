# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check build-time object friction through normal material spawning and replay."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_object_friction_build_and_spawn(_simulation_app, asset_path, procedural, backend):
    from isaaclab.sim.utils import create_new_stage
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import Sphere
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tests.test_asset_physics_config import _write_asset

    _write_asset(asset_path)
    obj = (
        Sphere(instance_name="ball")
        if procedural
        else Object(name="ball", usd_path=str(asset_path), object_type=ObjectType.BASE)
    )
    arena_env = IsaacLabArenaEnvironment(name="object_friction_test", scene=Scene(assets=[obj]))
    builder_cfg = ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, presets=backend)
    original = obj.object_cfg.spawn.physics_material
    _, disabled_kwargs = ArenaEnvBuilder(arena_env, builder_cfg).compose_manager_cfg()
    assert not obj.get_variation("friction").enabled
    assert obj.object_cfg.spawn.physics_material is original
    assert "ball.friction" not in disabled_kwargs["variation_recorder"]
    cfg, kwargs = ArenaEnvBuilder(
        arena_env,
        builder_cfg,
        hydra_overrides=[
            "ball.friction.enabled=true",
            "ball.friction.sampler_cfg.low=[0.6]",
            "ball.friction.sampler_cfg.high=[0.8]",
        ],
    ).compose_manager_cfg()
    record = kwargs["variation_recorder"]["ball.friction"]
    friction = float(record.sample_for_episode(0, 0)[0])
    assert 0.6 <= friction <= 0.8
    assert float(record.sample_for_episode(1, 10)[0]) == friction
    spawn = cfg.scene.ball.spawn
    stage = create_new_stage()
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    spawn.func("/World/env_.*/Object", spawn)
    for index in range(2):
        colliders = [
            prim
            for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/env_{index}/Object"))
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        assert colliders
        for collider in colliders:
            material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
            physics = UsdPhysics.MaterialAPI(material.GetPrim())
            assert physics.GetStaticFrictionAttr().Get() == pytest.approx(friction)
            assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(friction)
    if backend == "newton":
        import newton

        builder = newton.ModelBuilder()
        builder.add_usd(stage, root_path="/World/env_0/Object", load_visual_shapes=False)
        assert len(builder.shape_material_mu) > 0
        assert all(value == pytest.approx(friction) for value in builder.shape_material_mu)
    return True


@pytest.mark.parametrize("procedural", [False, True])
@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_object_friction_build_and_spawn(tmp_path, procedural, backend):
    assert run_function_with_persistent_simulation_app(
        _test_object_friction_build_and_spawn,
        asset_path=tmp_path / "object.usda",
        procedural=procedural,
        backend=backend,
    )


def _test_object_friction_replay_and_preservation(_simulation_app):
    from copy import deepcopy

    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab_physx.sim.spawners.materials import PhysxMaterialCfg, RigidBodyMaterialCfg

    from isaaclab_arena.assets.object_library import Sphere
    from isaaclab_arena.variations.object_friction_variation import ObjectFrictionVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    usd = UsdPhysicsRigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.1, restitution=0.3, density=500)
    physx = PhysxMaterialCfg(friction_combine_mode="multiply", compliant_contact_stiffness=100)
    materials = [None, usd, physx, [usd, physx], (usd, physx), RigidBodyMaterialCfg(restitution=0.3)]
    for original in materials:
        shared_spawn = deepcopy(Sphere.default_spawner_cfg)
        shared_spawn.physics_material = original
        obj = Sphere(instance_name="ball", spawner_cfg=shared_spawn)
        other = Sphere(instance_name="other", spawner_cfg=shared_spawn)
        variation = obj.get_variation("friction")
        recorded = []
        variation.add_sample_listener(lambda sample, env_ids: recorded.append(float(sample[0, 0])))
        for friction in (0.5, 0.9):
            variation.set_replay_sampler(lambda count, env_ids: [[friction]])
            variation.configure_at_build_time(seed=42, variation_key="ball.friction")
            material = obj.object_cfg.spawn.physics_material
            fragments = material if isinstance(material, list) else [material]
            friction_fragments = [
                f for f in fragments if isinstance(f, (UsdPhysicsRigidBodyMaterialCfg, RigidBodyMaterialCfg))
            ]
            assert len(friction_fragments) == 1
            assert friction_fragments[0].static_friction == pytest.approx(friction)
            assert friction_fragments[0].dynamic_friction == pytest.approx(friction)
            if original is usd or isinstance(original, (list, tuple, RigidBodyMaterialCfg)):
                assert friction_fragments[0].restitution == 0.3
            if original is usd or isinstance(original, (list, tuple)):
                assert friction_fragments[0].density == 500
            if original is physx or isinstance(original, (list, tuple)):
                backend_cfg = next(f for f in fragments if isinstance(f, PhysxMaterialCfg))
                assert backend_cfg.friction_combine_mode == "multiply"
                assert backend_cfg.compliant_contact_stiffness == 100
            assert other.object_cfg.spawn.physics_material == original
            assert shared_spawn.physics_material is original
        assert recorded == pytest.approx([0.5, 0.9])
        assert usd.static_friction == 0.2
        previous_spawn = obj.object_cfg.spawn
        for invalid in (-0.1, float("nan"), float("inf")):
            variation.set_replay_sampler(lambda count, env_ids: [[invalid]])
            with pytest.raises(AssertionError, match="finite and nonnegative"):
                variation.configure_at_build_time(seed=42, variation_key="ball.friction")
            assert obj.object_cfg.spawn is previous_spawn
        variation.set_replay_sampler(None)
        variation.apply_cfg(ObjectFrictionVariationCfg(sampler_cfg=UniformSamplerCfg(low=[0, 0], high=[1, 1])))
        with pytest.raises(AssertionError, match="one scalar"):
            variation.configure_at_build_time(seed=42, variation_key="ball.friction")
    return True


def test_object_friction_replay_and_preservation():
    assert run_function_with_persistent_simulation_app(_test_object_friction_replay_and_preservation)


def _test_object_friction_in_physics(_simulation_app):
    import torch

    import warp as wp

    from isaaclab_arena.assets.object_library import Sphere
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.variations.object_friction_variation import ObjectFrictionVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    sphere = Sphere(initial_pose=Pose(position_xyz=(0.0, 0.0, 1.0)))
    sphere.get_variation("friction").apply_cfg(
        ObjectFrictionVariationCfg(enabled=True, sampler_cfg=UniformSamplerCfg(low=[0.65], high=[0.65]))
    )
    arena_env = IsaacLabArenaEnvironment(name="object_friction_physics_test", scene=Scene(assets=[sphere]))
    env = ArenaEnvBuilder(
        arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, presets="physx")
    ).make_registered()
    try:
        env.reset()
        base_env = env.unwrapped
        materials = wp.to_torch(base_env.scene["sphere"].root_view.get_material_properties()).clone()
        torch.testing.assert_close(materials[..., :2], torch.full_like(materials[..., :2], 0.65))
        torch.testing.assert_close(materials[0], materials[1])
        actions = torch.zeros((2, base_env.action_manager.total_action_dim), device=base_env.device)
        for _ in range(3):
            env.step(actions)
        env.reset()
        torch.testing.assert_close(wp.to_torch(base_env.scene["sphere"].root_view.get_material_properties()), materials)
        record = base_env.variation_recorder["sphere.friction"]
        assert record.sample_for_episode(1, base_env.get_episode_index(1))[0] == pytest.approx(0.65)
    finally:
        env.close()
    return True


def test_object_friction_in_physics():
    assert run_function_with_persistent_simulation_app(_test_object_friction_in_physics)
