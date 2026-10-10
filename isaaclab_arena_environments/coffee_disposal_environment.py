# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reference-video coffee grounds disposal in a kitchen with a sliding waste drawer."""

from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


@dataclass
class CoffeeDisposalEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the kitchen and the ordered grounds-disposal task."""

    teleop_device: str | None = None
    episode_length_s: float = 300.0


@register_environment
class CoffeeDisposalEnvironment(ArenaEnvironmentFactory[CoffeeDisposalEnvironmentCfg]):
    name = "coffee_disposal"
    _legacy_argparse_cfg_type = CoffeeDisposalEnvironmentCfg

    def build(self, cfg: CoffeeDisposalEnvironmentCfg):
        from pathlib import Path

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.coffee_disposal.layout import (
            CABINET_POSITION,
            CUP_HOME,
            GROUND_POSITIONS,
            LID_HOME,
            MACHINE_POSITION,
        )
        from isaaclab_arena_environments.coffee_disposal.task import CoffeeDisposalTask

        directory = Path(__file__).with_name("coffee_disposal") / "assets"
        objects = {
            "kitchen": ("kitchen", (0.0, 0.0, 0.0), ObjectType.BASE),
            "coffee_machine": ("coffee_machine", MACHINE_POSITION, ObjectType.ARTICULATION),
            "waste_cabinet": ("waste_cabinet", CABINET_POSITION, ObjectType.ARTICULATION),
            "grounds_container": ("grounds_cup", CUP_HOME, ObjectType.RIGID),
            "bin_lid": ("bin_lid", LID_HOME, ObjectType.RIGID),
        }
        for name, position in GROUND_POSITIONS.items():
            objects[name] = ("grounds", position, ObjectType.RIGID)
        assets = []
        for name, (filename, position, object_type) in objects.items():
            addons = {"collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)}
            if object_type == ObjectType.RIGID:
                addons["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.05,
                    angular_damping=0.1,
                )
            obj = Object(
                name=name,
                usd_path=str(directory / f"{filename}.usda"),
                object_type=object_type,
                initial_pose=Pose(position_xyz=position),
                spawn_cfg_addon=addons,
            )
            if object_type == ObjectType.ARTICULATION:
                obj.object_cfg.actuators = {
                    "slides": ImplicitActuatorCfg(
                        joint_names_expr=[".*_slide"],
                        stiffness=0.0,
                        damping=8.0,
                        effort_limit_sim=150.0,
                        # 3.5 N holds the 0.30 kg dispenser against gravity after release.
                        friction=(
                            {"spout_slide": 3.5, "tray_slide": 0.3}
                            if name == "coffee_machine"
                            else {"drawer_slide": 0.5}
                        ),
                        dynamic_friction=(
                            {"spout_slide": 2.0, "tray_slide": 0.2}
                            if name == "coffee_machine"
                            else {"drawer_slide": 0.3}
                        ),
                    )
                }
            assets.append(obj)
        for name, spawn, position, rotation in (
            ("ambient", sim_utils.DomeLightCfg(intensity=900.0, color=(0.92, 0.95, 1.0)), (0, 0, 0), (0, 0, 0, 1)),
            (
                "key_light",
                sim_utils.DiskLightCfg(intensity=1600.0, radius=1.1, color=(1.0, 0.95, 0.87)),
                (0.1, -0.2, 3.0),
                (0, 0, 0, 1),
            ),
            (
                "front_fill",
                sim_utils.DiskLightCfg(intensity=700.0, radius=0.85, color=(0.90, 0.95, 1.0)),
                (0.1, -1.5, 2.0),
                (0.46175, 0, 0, 0.88701),
            ),
        ):
            assets.append(
                Object(
                    name=name,
                    object_type=ObjectType.BASE,
                    spawner_cfg=spawn,
                    initial_pose=Pose(position_xyz=position, rotation_xyzw=rotation),
                )
            )
        robot = FrankaIKEmbodiment(enable_cameras=cfg.enable_cameras)
        # Mount the bare arm on the countertop; the default floor stand would intersect the cabinet.
        robot.scene_config.robot.spawn.usd_path = FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path
        robot.set_initial_pose(Pose(position_xyz=(0.05, -0.45, 0.90)))
        robot.set_initial_joint_pose([2.5, -0.6, 0.0, -2.4, 0.0, 1.8, 0.7854, 0.04, 0.04])
        robot.event_config.randomize_franka_joint_state.params["std"] = 0.0
        if cfg.enable_cameras:
            from isaaclab_arena_environments.coffee_disposal.cameras import CoffeeDisposalCameraCfg

            robot.camera_config = CoffeeDisposalCameraCfg()
        teleop = self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device else None
        return IsaacLabArenaEnvironment(
            name=self.name,
            scene=Scene(assets=assets),
            embodiment=robot,
            task=CoffeeDisposalTask(cfg.episode_length_s),
            teleop_device=teleop,
        )
