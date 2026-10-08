# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Local-asset PC upgrade scene with a Franka arm and ordered service task."""

from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


@dataclass
class PCUpgradeEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the PC-upgrade scene."""

    teleop_device: str | None = None
    episode_length_s: float = 300.0


@register_environment
class PCUpgradeEnvironment(ArenaEnvironmentFactory[PCUpgradeEnvironmentCfg]):
    name = "pc_upgrade"
    _legacy_argparse_cfg_type = PCUpgradeEnvironmentCfg

    def build(self, cfg: PCUpgradeEnvironmentCfg):
        from pathlib import Path

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg

        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.pc_upgrade.layout import BIN_POSITION, CARD_POSITIONS, PC_POSITION
        from isaaclab_arena_environments.pc_upgrade.task import PCUpgradeTask

        directory = Path(__file__).with_name("pc_upgrade") / "assets"
        assets = []
        for name, position in {
            "bench": (0.0, 0.0, 0.0),
            "recycling_bin": BIN_POSITION,
            "pc_chassis": PC_POSITION,
            **CARD_POSITIONS,
        }.items():
            object_type = ObjectType.RIGID if name in CARD_POSITIONS else ObjectType.BASE
            if name == "pc_chassis":
                object_type = ObjectType.ARTICULATION
            spawn_addon = {}
            if name == "pc_chassis":
                spawn_addon["collision_props"] = sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0)
            if name in CARD_POSITIONS:
                # Millimetre contact margins fit the connector channel; bounded
                # depenetration prevents large impulses at the thin supports.
                spawn_addon = {
                    "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=12,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.02,
                        angular_damping=0.05,
                    ),
                    "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                }
            obj = Object(
                name=name,
                usd_path=str(directory / f"{name}.usda"),
                object_type=object_type,
                initial_pose=Pose(position_xyz=position),
                spawn_cfg_addon=spawn_addon,
            )
            if name == "pc_chassis":
                obj.object_cfg.actuators = {
                    "hinge": ImplicitActuatorCfg(
                        joint_names_expr=["panel_hinge"], stiffness=0.0, damping=0.8, effort_limit_sim=20.0
                    )
                }
            assets.append(obj)
        assets.append(
            Object(
                name="lighting",
                prim_path="/World/PCUpgradeDome",
                object_type=ObjectType.BASE,
                spawner_cfg=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.88, 0.93, 1.0)),
            )
        )
        assets.append(
            Object(
                name="floor",
                prim_path="/World/PCUpgradeFloor",
                object_type=ObjectType.BASE,
                spawner_cfg=sim_utils.CuboidCfg(
                    size=(200.0, 200.0, 0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.14, 0.17), roughness=0.85),
                ),
                initial_pose=Pose(position_xyz=(0.0, 0.0, -0.05)),
            )
        )
        assets.append(
            Object(
                name="work_light",
                object_type=ObjectType.BASE,
                spawner_cfg=sim_utils.DiskLightCfg(intensity=1600.0, radius=0.8, color=(1.0, 0.94, 0.85)),
                initial_pose=Pose(position_xyz=(0.4, 0.0, 2.7)),
            )
        )
        assets.append(
            Object(
                name="front_fill_light",
                object_type=ObjectType.BASE,
                spawner_cfg=sim_utils.DiskLightCfg(intensity=700.0, radius=0.65, color=(0.88, 0.94, 1.0)),
                initial_pose=Pose(
                    position_xyz=(0.65, -0.90, 1.90),
                    rotation_xyzw=(0.461748613, 0.0, 0.0, 0.887010833),
                ),
            )
        )
        robot = FrankaIKEmbodiment(enable_cameras=cfg.enable_cameras)
        if cfg.enable_cameras:
            from isaaclab_arena_environments.pc_upgrade.cameras import PCUpgradeCameraCfg

            robot.camera_config = PCUpgradeCameraCfg()
        robot.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.75)))
        robot.set_initial_joint_pose([0.0, -0.6, 0.0, -2.4, 0.0, 1.8, 0.7854, 0.04, 0.04])
        # Reproducible setup for teleoperation and scripted controller development.
        robot.event_config.randomize_franka_joint_state.params["std"] = 0.0
        teleop = self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device else None
        return IsaacLabArenaEnvironment(
            name=self.name,
            scene=Scene(assets=assets),
            embodiment=robot,
            task=PCUpgradeTask(cfg.episode_length_s),
            teleop_device=teleop,
        )
