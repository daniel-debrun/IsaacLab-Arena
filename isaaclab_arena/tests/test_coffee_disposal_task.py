# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Observed coffee-disposal history, containment, retention, and indexed reset."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_coffee_disposal_order_and_reset(simulation_app):
    import math
    import torch
    from types import SimpleNamespace

    import warp as wp
    from isaaclab.utils.warp.proxy_array import ProxyArray

    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena_environments.coffee_disposal import layout
    from isaaclab_arena_environments.coffee_disposal.task import CoffeeDisposalTask, dropped_object, final_state

    origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, 2.0, 0.2]])

    def obj(position, joints=()):
        asset = SimpleNamespace(
            joint_names=list(joints),
            find_joints=lambda name: ([list(joints).index(name)], [name]),
            data=SimpleNamespace(
                root_pos_w=torch.tensor(position).expand(2, 3).clone() + origins,
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
                root_vel_w=torch.zeros(2, 6),
                joint_pos=torch.zeros(2, len(joints)),
                joint_vel=torch.zeros(2, len(joints)),
            ),
        )
        # Actual Isaac Lab wrappers catch TorchScript and tensor-method boundary mistakes.
        for name, tensor in vars(asset.data).items():
            setattr(asset.data, name, ProxyArray(wp.from_torch(tensor)))
        return asset

    class Scene(dict):
        env_origins = origins

    scene = Scene(
        coffee_machine=obj(layout.MACHINE_POSITION, ("tray_slide", "spout_slide")),
        waste_cabinet=obj(layout.CABINET_POSITION, ("drawer_slide",)),
        grounds_container=obj(layout.CUP_HOME),
        bin_lid=obj(layout.LID_HOME),
    )
    for name, position in layout.GROUND_POSITIONS.items():
        scene[name] = obj(position)
    scene["ee_frame"] = SimpleNamespace(data=SimpleNamespace(target_pos_w=origins[:, None] + 3.0))
    env = SimpleNamespace(
        scene=scene,
        num_envs=2,
        device="cpu",
        coffee_grasps={
            "grounds_container": torch.zeros(2, dtype=torch.bool),
            "bin_lid": torch.zeros(2, dtype=torch.bool),
        },
    )
    config = CoffeeDisposalTask().get_termination_cfg()
    assert len(config.success) == 11
    tracker = ProgressTracker(config.success, 2, "cpu", env=env, subtasks_are_sequential=True)
    steps = torch.zeros(2, dtype=torch.long)

    def advance(count=25):
        for _ in range(count):
            tracker.step(env, step_index=steps)
            steps.add_(1)

    def pose(name, position):
        scene[name].data.root_pos_w[:] = torch.as_tensor(position) + origins

    def drawer_pose(offset):
        scene["waste_cabinet"].data.joint_pos[:, 0] = offset

    def grounds_in_bin(offset=layout.DRAWER_EXTENSION):
        base = torch.tensor(layout.CABINET_POSITION) + torch.tensor(layout.BIN_IN_DRAWER)
        base[1] += offset
        for index, name in enumerate(layout.GROUND_NAMES):
            pose(name, base + torch.tensor([(-0.035, 0.035)[index % 2], 0.0, 0.05 + index // 2 * 0.037]))

    cup_out = torch.tensor(layout.CUP_HOME) + torch.tensor([0.0, layout.TRAY_EXTENSION, 0.0])
    lid_home = torch.tensor(layout.LID_HOME) + torch.tensor([0.0, layout.DRAWER_EXTENSION, 0.0])
    lid_park = torch.tensor(layout.CABINET_POSITION) + torch.tensor(layout.LID_PARK_IN_DRAWER)
    lid_park[1] += layout.DRAWER_EXTENSION
    above_bin = torch.tensor(layout.CABINET_POSITION) + torch.tensor(layout.BIN_IN_DRAWER)
    above_bin += torch.tensor([0.0, layout.DRAWER_EXTENSION, layout.BIN_HEIGHT + 0.24])

    # Final positions cannot replace evidence that the service sequence happened.
    scene["coffee_machine"].data.joint_pos[:, 1] = layout.SPOUT_RAISED
    grounds_in_bin(0.0)
    advance()
    assert not tracker.get_subtask_completion().any()
    scene["coffee_machine"].data.joint_pos[:, 1] = 0.0
    for name, position in layout.GROUND_POSITIONS.items():
        pose(name, position)
    advance(3)
    scene["coffee_machine"].data.joint_pos[:, 1] = layout.SPOUT_RAISED
    advance()
    assert tracker.get_subtask_completion()[:, 0].all()
    scene["coffee_machine"].data.joint_pos[:, 0] = layout.TRAY_EXTENSION
    pose("grounds_container", cup_out)
    for name, position in layout.GROUND_POSITIONS.items():
        pose(name, torch.tensor(position) + torch.tensor([0.0, layout.TRAY_EXTENSION, 0.0]))
    advance()
    drawer_pose(layout.DRAWER_EXTENSION)
    pose("bin_lid", lid_home)
    advance()
    # Direct parking is insufficient: the removable lid must first clear its bin.
    pose("bin_lid", lid_park)
    advance()
    assert not tracker.get_subtask_completion()[:, 3].any()
    pose("bin_lid", lid_home + torch.tensor([0.0, 0.0, 0.12]))
    advance(5)
    pose("bin_lid", lid_park)
    scene["bin_lid"].data.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    advance()
    assert not tracker.get_subtask_completion()[:, 3].any(), "An overturned lid is not parked"
    scene["bin_lid"].data.root_quat_w[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    advance()
    assert tracker.get_subtask_completion()[:, :4].all()
    pose("grounds_container", cup_out + torch.tensor([0.0, 0.0, 0.18]))
    advance()
    pose("grounds_container", above_bin)
    for index, name in enumerate(layout.GROUND_NAMES):
        pose(name, above_bin + torch.tensor([(-0.025, 0.025)[index % 2], 0.0, 0.025 + index // 2 * 0.017]))
    advance()
    assert tracker.get_subtask_completion()[:, :6].all()
    # Grounds in the right bin are insufficient without the observed tipping motion.
    grounds_in_bin()
    advance()
    assert not tracker.get_subtask_completion()[:, 6].any()
    angle = math.radians(145.0)
    scene["grounds_container"].data.root_quat_w[:] = torch.tensor([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])
    advance()
    assert not tracker.get_subtask_completion()[:, 6].any(), "Empty-cup tipping after disposal is not discharge"
    # Motion of clumps already inside the bin cannot replace observed discharge.
    scene["grounds_7"].data.root_vel_w[:, 2] = -0.5
    advance()
    scene["grounds_7"].data.root_vel_w[:].zero_()
    advance()
    assert not tracker.get_subtask_completion()[:, 6].any(), "Bin containment must be checked independently of settling"
    # Grounds arriving while the cup is nearly upright still cannot count as tipping.
    angle = math.radians(25.0)
    scene["grounds_container"].data.root_quat_w[:] = torch.tensor([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])
    pose("grounds_7", above_bin + torch.tensor([0.0, 0.0, -0.08]))
    advance(3)
    grounds_in_bin()
    angle = math.radians(145.0)
    scene["grounds_container"].data.root_quat_w[:] = torch.tensor([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])
    advance()
    assert not tracker.get_subtask_completion()[:, 6].any(), "Discharge onset requires a meaningful cup tilt"
    # Real discharge can finish before the cup reaches the full-inversion threshold.
    angle = math.radians(45.0)
    scene["grounds_container"].data.root_quat_w[:] = torch.tensor([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])
    pose("grounds_7", above_bin + torch.tensor([0.0, 0.0, -0.08]))
    advance(1)
    grounds_in_bin()
    advance(3)
    assert not tracker.get_subtask_completion()[:, 6].any(), "The cup must still reach its full tipping pose"
    angle = math.radians(145.0)
    scene["grounds_container"].data.root_quat_w[:] = torch.tensor([math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)])
    advance(5)
    # A single loose clump outside the organic bin prevents bulk-disposal credit.
    scene["grounds_7"].data.root_pos_w[:, 0] -= 0.30
    advance()
    assert not tracker.get_subtask_completion()[:, 6].any()
    scene["grounds_7"].data.root_pos_w[:, 2] = origins[:, 2] + 0.04
    assert dropped_object(env).all()
    grounds_in_bin()
    scene["grounds_7"].data.root_vel_w[:, 2] = -1.0
    assert not dropped_object(env).any(), "Falling inside the bin is not a spill"
    advance()
    assert not tracker.get_subtask_completion()[:, 6].any(), "Grounds must settle"
    scene["grounds_7"].data.root_vel_w[:].zero_()
    advance()
    assert tracker.get_subtask_completion()[:, 6].all()
    # A held, tilted, or misaligned cup cannot count as reseated.
    pose("grounds_container", cup_out)
    advance()
    assert not tracker.get_subtask_completion()[:, 7].any()
    scene["grounds_container"].data.root_quat_w[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    scene["grounds_container"].data.root_pos_w[:, 0] += 0.025
    advance()
    assert not tracker.get_subtask_completion()[:, 7].any(), "Cup must align with its recess"
    scene["grounds_container"].data.root_pos_w[:, 0] -= 0.025
    env.coffee_grasps["grounds_container"][:] = True
    advance()
    assert not tracker.get_subtask_completion()[:, 7].any()
    env.coffee_grasps["grounds_container"][:] = False
    advance()
    scene["coffee_machine"].data.joint_pos[:, 0] = 0.0
    pose("grounds_container", layout.CUP_HOME)
    advance()
    assert tracker.get_subtask_completion()[:, :9].all()
    pose("bin_lid", lid_home)
    advance()
    assert not tracker.get_subtask_completion()[:, 9].any(), "Replacement must include pickup from the support"
    pose("bin_lid", lid_park + torch.tensor([0.0, 0.0, 0.12]))
    advance(5)
    pose("bin_lid", lid_home)
    advance()
    assert tracker.get_subtask_completion()[:, :10].all()
    drawer_pose(0.0)
    pose("bin_lid", layout.LID_HOME)
    grounds_in_bin(0.0)
    scene["grounds_container"].data.root_pos_w[:, 0] += 0.10
    advance()
    assert not tracker.is_complete().any(), "Final closure must retain the seated cup"
    scene["grounds_container"].data.root_pos_w[:, 0] -= 0.10
    scene["grounds_7"].data.root_pos_w[:, 0] -= 0.30
    advance()
    assert not tracker.is_complete().any(), "Final closure must retain all loose grounds"
    grounds_in_bin(0.0)
    advance(14)
    assert not tracker.is_complete().any()
    advance(1)
    assert tracker.is_complete().all()
    assert final_state(env).all()
    tracker.reset(torch.tensor([0]))
    assert not tracker.is_complete()[0] and tracker.is_complete()[1]
    advance(30)
    assert not tracker.get_subtask_completion()[0].any(), "Partial reset must clear initial-state history"
    assert tracker.is_complete()[1]
    scene["coffee_machine"].data.joint_pos[0, 1] = 0.0
    advance(1)
    scene["coffee_machine"].data.joint_pos[0, 1] = layout.SPOUT_RAISED
    advance(2)
    assert not tracker.get_subtask_completion()[0, 0]
    tracker.reset(torch.tensor([0]))
    advance(5)
    assert not tracker.get_subtask_completion()[0, 0], "Reset must discard the partial dwell and low-spout history"
    assert tracker.is_complete()[1]
    return True


def test_coffee_disposal_order_and_reset():
    assert run_function_with_persistent_simulation_app(_test_coffee_disposal_order_and_reset)


def _test_coffee_disposal_joint_reset_and_observations(simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena_environments.coffee_disposal import layout
    from isaaclab_arena_environments.coffee_disposal.task import object_state, reset_joints

    class Articulation:
        def __init__(self, count):
            self.data = SimpleNamespace(
                joint_pos=torch.full((2, count), 0.3),
                joint_vel=torch.full((2, count), 0.4),
                root_pos_w=torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 0.0]]),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
                root_vel_w=torch.zeros(2, 6),
            )
            self.targets = self.data.joint_pos.clone()

        def write_joint_state_to_sim(self, position, velocity, env_ids):
            self.data.joint_pos[env_ids] = position
            self.data.joint_vel[env_ids] = velocity

        def set_joint_position_target(self, position, env_ids):
            self.targets[env_ids] = position

    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 0.0]])

    scene = Scene(coffee_machine=Articulation(2), waste_cabinet=Articulation(1))
    for name in ("grounds_container", "bin_lid", *layout.GROUND_NAMES):
        scene[name] = Articulation(0)
    env = SimpleNamespace(scene=scene)
    before = object_state(env)
    assert torch.equal(before[0], before[1]), "Observation positions must exclude environment origins"
    reset_joints(env, torch.tensor([0]))
    for name in ("coffee_machine", "waste_cabinet"):
        asset = scene[name]
        assert not asset.data.joint_pos[0].any()
        assert not asset.data.joint_vel[0].any()
        assert not asset.targets[0].any()
        assert (asset.data.joint_pos[1] == 0.3).all()
        assert (asset.data.joint_vel[1] == 0.4).all()
        assert (asset.targets[1] == 0.3).all()
    after = object_state(env)
    assert torch.equal(before[1], after[1])
    assert not torch.equal(before[0], after[0]), "Observations must expose slider changes"
    return True


def test_coffee_disposal_joint_reset_and_observations():
    assert run_function_with_persistent_simulation_app(_test_coffee_disposal_joint_reset_and_observations)


def _test_coffee_ground_hull_containment(simulation_app):
    import math
    import torch
    from types import SimpleNamespace

    from isaaclab_arena_environments.coffee_disposal import layout
    from isaaclab_arena_environments.coffee_disposal.task import grounds_in_bin

    class Scene(dict):
        env_origins = torch.zeros(2, 3)

    scene = Scene()
    cabinet = SimpleNamespace(
        root_pos_w=torch.tensor(layout.CABINET_POSITION).repeat(2, 1),
        root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        joint_pos=torch.full((2, 1), layout.DRAWER_EXTENSION),
        joint_vel=torch.zeros(2, 1),
    )
    scene["waste_cabinet"] = SimpleNamespace(data=cabinet, find_joints=lambda name: ([0], [name]))
    bin_origin = cabinet.root_pos_w + torch.tensor(layout.BIN_IN_DRAWER)
    bin_origin[:, 1] += layout.DRAWER_EXTENSION
    for name in layout.GROUND_NAMES:
        scene[name] = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=bin_origin + torch.tensor([0.0, 0.0, 0.06]),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
                root_vel_w=torch.zeros(2, 6),
            )
        )
    env = SimpleNamespace(scene=scene, num_envs=2, device="cpu")
    ground = scene["grounds_0"].data
    vertices = torch.tensor(layout.GROUND_COLLISION_VERTICES)
    # Local +Z becomes bin +X under this 90-degree Y rotation.
    ground.root_quat_w[:] = torch.tensor([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    rotated_extent_x = vertices[:, 2].max()
    ground.root_pos_w[:, 0] = bin_origin[:, 0] + layout.BIN_HALF_INTERIOR[0] - rotated_extent_x - 0.0001
    assert grounds_in_bin(env).all(), "The flattened convex hull fits despite its conservative bounding sphere"
    ground.root_pos_w[0, 0] += 0.0004
    assert grounds_in_bin(env).tolist() == [False, True], "Reject actual hull protrusion without affecting other envs"
    ground.root_pos_w[0, 0] -= 0.0004
    ground.root_quat_w[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert not grounds_in_bin(env).any(), "Containment must follow the actual clump orientation"
    ground.root_pos_w[:] = bin_origin + torch.tensor([0.0, 0.0, 0.06])
    ground.root_pos_w[:, 2] = bin_origin[:, 2] + layout.BIN_HEIGHT - vertices[:, 2].max() + 0.0003
    assert not grounds_in_bin(env).any(), "A clump protruding above the rim is not contained"
    ground.root_pos_w[:, 2] = bin_origin[:, 2] - vertices[:, 2].min() + 0.006
    assert not grounds_in_bin(env).any(), "The interior floor is above the bin asset origin"
    ground.root_pos_w[:, 2] = bin_origin[:, 2] - vertices[:, 2].min() + 0.012 - 0.0003
    assert not grounds_in_bin(env).any(), "A clump protruding through the actual floor is not contained"
    ground.root_pos_w[:, 2] = bin_origin[:, 2] - vertices[:, 2].min() + 0.012 - 0.0001
    assert grounds_in_bin(env).all(), "Allow only a small solver-contact tolerance at the floor"
    # Recorded near-stationary contact residual from the second demo environment.
    ground.root_quat_w[:] = torch.tensor([-0.32240778, -0.71973550, 0.11546630, -0.60390514])
    ground.root_pos_w[:] = bin_origin + torch.tensor([0.0, 0.0, 0.06])
    ground.root_vel_w[:] = torch.tensor([-0.00148237, -0.00077474, 0.00378837, 0.24302869, -0.18559389, -0.26459336])
    assert grounds_in_bin(env).all(), "Small clumps settle by surface speed, not a cup-sized angular cutoff"
    ground.root_vel_w[:] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.5])
    assert not grounds_in_bin(env).any(), "Fast clump spin must prevent completion"
    ground.root_vel_w[:] = torch.tensor([0.011, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert not grounds_in_bin(env).any(), "Slow drifting clump centers must prevent completion"
    ground.root_quat_w[:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    ground.root_vel_w[:] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    assert grounds_in_bin(env).all(), "This slow rotation alone is below the surface-speed limit"
    ground.root_vel_w[:] = torch.tensor([0.009, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert grounds_in_bin(env).all(), "This slow translation alone is below both speed limits"
    ground.root_vel_w[:, 5] = 0.7
    assert not grounds_in_bin(env).any(), "Combined translation and rotation can exceed the surface-speed limit"
    ground.root_vel_w[:].zero_()
    assert grounds_in_bin(env).all()
    return True


def test_coffee_ground_hull_containment():
    assert run_function_with_persistent_simulation_app(_test_coffee_ground_hull_containment)


def _test_coffee_clump_collision_bounds_after_live_drop(simulation_app):
    import math
    import torch
    from pathlib import Path

    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.coffee_disposal import layout
    from isaaclab_arena_environments.coffee_disposal.task import bin_position, grounds_in_bin, lid_position, settled
    from isaaclab_arena_environments.coffee_disposal_environment import (
        CoffeeDisposalEnvironment,
        CoffeeDisposalEnvironmentCfg,
    )

    asset_stage = Usd.Stage.Open(str(Path(layout.__file__).with_name("assets") / "grounds.usda"))
    collision_meshes = []
    for prim in asset_stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_meshes.append(UsdGeom.Mesh(prim))
    assert len(collision_meshes) == 1
    vertices = tuple(tuple(point) for point in collision_meshes[0].GetPointsAttr().Get())
    assert len(vertices) <= 64, "The explicit collision hull must stay below PhysX GPU cooking limits"
    assert torch.allclose(torch.tensor(vertices), torch.tensor(layout.GROUND_COLLISION_VERTICES), atol=1e-7)
    builder = ArenaEnvBuilder(
        CoffeeDisposalEnvironment().build(CoffeeDisposalEnvironmentCfg()),
        ArenaEnvBuilderCfg(num_envs=1, solve_relations=False),
    )
    env = builder.make_registered()
    try:
        env.reset()
        base = env.unwrapped
        cabinet = base.scene["waste_cabinet"]
        joints = cabinet.data.joint_pos[:].clone()
        joints[:, cabinet.find_joints("drawer_slide")[0][0]] = layout.DRAWER_EXTENSION
        cabinet.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        lid = base.scene["bin_lid"]
        lid_state = lid.data.root_state_w[:].clone()
        lid_state[:, :3] = lid_position(base, parked=True)
        lid_state[:, 3:7] = cabinet.data.root_quat_w[:]
        lid_state[:, 7:] = 0.0
        lid.write_root_state_to_sim(lid_state)
        origin = bin_position(base)
        drop_xy = (
            (-0.083, -0.123),
            (0.083, -0.123),
            (-0.083, 0.123),
            (0.083, 0.123),
            (-0.083, 0.0),
            (0.083, 0.0),
            (0.0, -0.123),
            (0.0, 0.123),
        )
        for index, name in enumerate(layout.GROUND_NAMES):
            ground = base.scene[name]
            state = ground.data.root_state_w[:].clone()
            state[:, :3] = origin + origin.new_tensor((*drop_xy[index], layout.BIN_HEIGHT + 0.09 + index * 0.005))
            angle = math.radians(17.0 + index * 31.0)
            axis = 0 if index % 2 else 1
            state[:, 3:7] = 0.0
            state[:, 3 + axis] = math.sin(angle / 2)
            state[:, 6] = math.cos(angle / 2)
            state[:, 7:] = 0.0
            ground.write_root_state_to_sim(state)
        action = torch.zeros(env.action_space.shape, device=base.device)
        action[:, -1] = 1.0
        for _ in range(45):
            env.step(action)
        origin = bin_position(base)
        vertices_G = origin.new_tensor(vertices)
        q_W_B = cabinet.data.root_quat_w[0].expand(len(vertices), -1)
        bounds = {}
        for name in layout.GROUND_NAMES:
            ground = base.scene[name].data
            assert settled(ground.root_vel_w[:]).all(), f"{name} must settle after its three-second drop"
            q_W_G = ground.root_quat_w[0].expand(len(vertices), -1)
            vertices_W = quat_apply(q_W_G, vertices_G) + ground.root_pos_w[0]
            vertices_B = quat_apply_inverse(q_W_B, vertices_W - origin[0])
            minimum = vertices_B.amin(dim=0)
            maximum = vertices_B.amax(dim=0)
            bounds[name] = (minimum.tolist(), maximum.tolist())
        tolerance = 0.0002
        for name, (minimum, maximum) in bounds.items():
            assert minimum[0] >= -layout.BIN_HALF_INTERIOR[0] - tolerance, (name, bounds)
            assert maximum[0] <= layout.BIN_HALF_INTERIOR[0] + tolerance, (name, bounds)
            assert minimum[1] >= -layout.BIN_HALF_INTERIOR[1] - tolerance, (name, bounds)
            assert maximum[1] <= layout.BIN_HALF_INTERIOR[1] + tolerance, (name, bounds)
            assert minimum[2] >= layout.BIN_FLOOR_THICKNESS - tolerance, (name, bounds)
            assert maximum[2] <= layout.BIN_HEIGHT + tolerance, (name, bounds)
        assert grounds_in_bin(base).all(), "Task containment must agree with the actual simulated collision mesh"
        return True
    finally:
        env.close()


def test_coffee_clump_collision_bounds_after_live_drop():
    assert run_function_with_persistent_simulation_app(_test_coffee_clump_collision_bounds_after_live_drop)


def _test_coffee_saved_pile_settles_in_two_environments(simulation_app):
    import json
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.coffee_disposal import layout
    from isaaclab_arena_environments.coffee_disposal.task import grounds_contained, grounds_in_bin, lid_position
    from isaaclab_arena_environments.coffee_disposal_environment import (
        CoffeeDisposalEnvironment,
        CoffeeDisposalEnvironmentCfg,
    )
    from isaaclab_arena_examples.coffee_disposal.motion import configure_demo_drives

    # A contained pile whose reported angular velocity remained ~0.4 rad/s for ten
    # seconds despite unchanged poses, captured from the second environment.
    # Each row is environment-local xyz, xyzw, linear velocity, angular velocity.
    recorded = (
        (
            -0.20240116119384766,
            -0.8726496696472168,
            0.3216080367565155,
            -0.3224077820777893,
            -0.7197355031967163,
            0.11546630412340164,
            -0.6039051413536072,
            -0.0014823655365034938,
            -0.0007747444324195385,
            0.0037883692421019077,
            0.2430286854505539,
            -0.1855938881635666,
            -0.26459336280822754,
        ),
        (
            -0.14510440826416016,
            -0.8641296625137329,
            0.3298061490058899,
            0.9467702507972717,
            0.058383334428071976,
            -0.023404378443956375,
            0.3157050311565399,
            -0.0002398111391812563,
            -0.0010824603959918022,
            0.0015257614431902766,
            -0.1328190118074417,
            0.039865899831056595,
            -0.1169731467962265,
        ),
        (
            -0.16928529739379883,
            -0.8719463348388672,
            0.3098830580711365,
            -0.8921065330505371,
            0.06739354133605957,
            0.3388631343841553,
            -0.2911626696586609,
            -0.00039311416912823915,
            0.0005526635213755071,
            -0.0009074321715161204,
            -0.09657014906406403,
            0.02775796502828598,
            0.2548880875110626,
        ),
        (
            -0.15781641006469727,
            -0.7155305743217468,
            0.3067554831504822,
            -0.9928535223007202,
            0.11933891475200653,
            -5.640406470774906e-06,
            -2.9694353997911094e-06,
            0.0003849616041406989,
            -0.00019676360534504056,
            -0.0004820536123588681,
            0.021448785439133644,
            0.035489313304424286,
            -0.0036826112773269415,
        ),
        (
            -0.2643404006958008,
            -0.8496834635734558,
            0.30997005105018616,
            -0.03554053232073784,
            -0.7178462147712708,
            -0.03432824835181236,
            0.6944459080696106,
            0.0007718923734501004,
            -0.000676341587677598,
            -0.0004431900451891124,
            0.01452972088009119,
            -0.04412328824400902,
            -0.04829995334148407,
        ),
        (
            -0.15401315689086914,
            -0.7885257005691528,
            0.3080570697784424,
            -0.9394619464874268,
            0.04634209722280502,
            0.016727831214666367,
            -0.339092493057251,
            -2.469017454131972e-05,
            -0.00041137830703519285,
            -0.0004747179918922484,
            0.03783268481492996,
            -0.002892350312322378,
            -0.002170688472688198,
        ),
        (
            -0.23076200485229492,
            -0.8562749028205872,
            0.30736324191093445,
            -0.04350832849740982,
            0.013320129364728928,
            0.29129743576049805,
            -0.9555497765541077,
            0.0007446638192050159,
            -0.0027026108000427485,
            0.0005818204372189939,
            0.17007774114608765,
            -0.1522904932498932,
            0.056293193250894547,
        ),
        (
            -0.12182807922363281,
            -0.8722150325775146,
            0.3090240955352783,
            -0.9314025640487671,
            0.11117243021726608,
            -0.3236317038536072,
            -0.12406633794307709,
            -0.0022741330321878195,
            0.001096166088245809,
            -0.00026356455055065453,
            -0.11904629319906235,
            -0.08556673675775528,
            0.061493173241615295,
        ),
    )
    builder = ArenaEnvBuilder(
        CoffeeDisposalEnvironment().build(CoffeeDisposalEnvironmentCfg()),
        ArenaEnvBuilderCfg(num_envs=2, env_spacing=12.0, solve_relations=False),
    )
    cfg, kwargs = builder.compose_manager_cfg()
    configure_demo_drives(cfg)
    env = builder.make_registered(cfg, kwargs)
    try:
        env.reset()
        base = env.unwrapped
        cabinet = base.scene["waste_cabinet"]
        joints = cabinet.data.joint_pos[:].clone()
        drawer_index = cabinet.find_joints("drawer_slide")[0][0]
        joints[:, drawer_index] = -0.5500414967536926
        cabinet.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        target = joints.clone()
        target[:, drawer_index] = layout.DRAWER_EXTENSION
        cabinet.set_joint_position_target(target)
        lid = base.scene["bin_lid"]
        lid_state = lid.data.root_state_w[:].clone()
        lid_state[:, :3] = lid_position(base, parked=True)
        lid_state[:, 3:7] = cabinet.data.root_quat_w[:]
        lid_state[:, 7:] = 0.0
        lid.write_root_state_to_sim(lid_state)
        for row, name in zip(recorded, layout.GROUND_NAMES):
            ground = base.scene[name]
            state = ground.data.root_state_w[:].clone()
            state[:] = state.new_tensor(row)
            state[:, :3] += base.scene.env_origins
            ground.write_root_state_to_sim(state)
            ground.root_view.wake_up()
        action = torch.zeros(env.action_space.shape, device=base.device)
        action[:, -1] = 1.0
        consecutive = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
        previous_positions = torch.stack([base.scene[name].data.root_pos_w[:] for name in layout.GROUND_NAMES], dim=1)
        diagnostics = []
        for index in range(75):
            env.step(action)
            consecutive = torch.where(grounds_in_bin(base), consecutive + 1, 0)
            if (index + 1) % 15 == 0:
                positions = torch.stack([base.scene[name].data.root_pos_w[:] for name in layout.GROUND_NAMES], dim=1)
                velocities = torch.stack([base.scene[name].data.root_vel_w[:] for name in layout.GROUND_NAMES], dim=1)
                diagnostics.append({
                    "seconds": (index + 1) * base.step_dt,
                    "max_displacement": (
                        torch.linalg.vector_norm(positions - previous_positions, dim=-1).amax(dim=1).tolist()
                    ),
                    "max_linear": torch.linalg.vector_norm(velocities[:, :, :3], dim=-1).amax(dim=1).tolist(),
                    "max_angular": torch.linalg.vector_norm(velocities[:, :, 3:], dim=-1).amax(dim=1).tolist(),
                    "drawer_velocity": cabinet.data.joint_vel[:, drawer_index].tolist(),
                    "consecutive_settled": consecutive.tolist(),
                })
                previous_positions = positions.clone()
        print("COFFEE_PILE_REPLAY " + json.dumps(diagnostics), flush=True)
        assert grounds_contained(base).all(), "The replay pile must retain every collision hull in both bins"
        assert (consecutive >= 8).all(), diagnostics
        # Normal contact with the moving drawer must carry the settled clumps.
        maximum_speed = torch.zeros(base.num_envs, device=base.device)
        for index in range(30):
            fraction = (index + 1) / 30
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            target[:, drawer_index] = layout.DRAWER_EXTENSION * (1.0 - blend)
            cabinet.set_joint_position_target(target)
            env.step(action)
            velocities = torch.stack([base.scene[name].data.root_vel_w[:] for name in layout.GROUND_NAMES], dim=1)
            speed = torch.linalg.vector_norm(velocities[:, :, :3], dim=-1).amax(dim=1)
            maximum_speed = torch.maximum(maximum_speed, speed)
        consecutive.zero_()
        for _ in range(30):
            env.step(action)
            consecutive = torch.where(grounds_in_bin(base), consecutive + 1, 0)
        print(
            "COFFEE_PILE_DRAWER_WAKE "
            + json.dumps({
                "maximum_speed": maximum_speed.tolist(),
                "consecutive_settled": consecutive.tolist(),
                "drawer_position": cabinet.data.joint_pos[:, drawer_index].tolist(),
            }),
            flush=True,
        )
        assert (maximum_speed > 0.02).all(), "Drawer contact must carry the clumps with the bin"
        assert cabinet.data.joint_pos[:, drawer_index].abs().max() < 0.01
        assert grounds_contained(base).all(), "All clumps must remain in the moving bin"
        assert (consecutive >= 8).all(), "Clumps must settle again after drawer closure"
        return True
    finally:
        env.close()


def test_coffee_saved_pile_settles_in_two_environments():
    assert run_function_with_persistent_simulation_app(_test_coffee_saved_pile_settles_in_two_environments)
