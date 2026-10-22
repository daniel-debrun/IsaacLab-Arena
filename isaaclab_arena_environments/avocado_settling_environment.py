# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Drop an avocado onto the Maple table before moving the Franka gripper."""

from dataclasses import dataclass

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


@dataclass
class AvocadoSettlingEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the avocado drop and subsequent gripper approach."""

    drop_height_m: float = 0.15
    """Additional height above the relation-solved tabletop placement."""
    sideways_speed_m_s: float = 0.08
    """Initial world-Y speed to make the avocado roll on landing."""
    spin_rad_s: float = 1.5
    """Initial world-X angular speed."""
    settled_steps: int = 15
    """Consecutive control steps below the object velocity thresholds."""
    episode_length_s: float = 20.0

    def __post_init__(self):
        assert self.drop_height_m > 0
        assert self.settled_steps > 0
        assert self.episode_length_s > 0


@register_environment
class AvocadoSettlingEnvironment(ArenaEnvironmentFactory[AvocadoSettlingEnvironmentCfg]):
    """Build a tabletop settling demonstration using the standard Franka IK embodiment."""

    name = "avocado_settling"

    def build(self, cfg: AvocadoSettlingEnvironmentCfg):
        import torch

        from isaaclab.envs.common import ViewerCfg
        from isaaclab.managers import EventTermCfg
        from isaaclab.sensors import CameraCfg, TiledCameraCfg
        from isaaclab.sim import PinholeCameraCfg
        from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, RequiresReachability
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.cameras import ArenaCameraCfg
        from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass
        from isaaclab_arena_examples.policy.avocado_settling import AvocadoSettlingTask, launch_avocado

        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        table = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table.add_relation(IsAnchor())
        avocado = self.asset_registry.get_asset_by_name("avocado01_fruits_veggies_robolab")(instance_name="avocado")
        avocado.add_relation(On(table, edge_margin_m=0.20))
        avocado.add_relation(RequiresReachability())
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(1500.0)
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")(enable_cameras=cfg.enable_cameras)
        # Match the controller's tool point to the frame used by the policy and success check.
        embodiment.action_config.arm_action.body_offset.pos = (0.0, 0.0, 0.1034)

        if cfg.enable_cameras:
            t_E_C = torch.tensor([[1.3, -1.4, 1.1]])
            t_E_target = torch.tensor([[0.4, 0.0, 0.20]])
            q_E_C = quat_from_matrix(create_rotation_matrix_from_view(t_E_C, t_E_target))[0]
            scene_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/scene_cam",
                height=480,
                width=640,
                data_types=["rgb"],
                spawn=PinholeCameraCfg(focal_length=18.0, horizontal_aperture=36.0),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(t_E_C[0].tolist()), rot=tuple(q_E_C.tolist()), convention="opengl"
                ),
            )
            SceneCameraCfg = make_configclass("AvocadoSceneCameraCfg", [("scene_cam", TiledCameraCfg, scene_camera)])
            embodiment.camera_config = combine_configclass_instances(
                "AvocadoCameraRigCfg", embodiment.camera_config, SceneCameraCfg(), bases=(ArenaCameraCfg,)
            )

        def configure_demo(env_cfg):
            # Append after Arena's placement reset so each episode starts above its solved support pose.
            env_cfg.events.launch_avocado = EventTermCfg(
                func=launch_avocado,
                mode="reset",
                params={
                    "drop_height_m": cfg.drop_height_m,
                    "sideways_speed_m_s": cfg.sideways_speed_m_s,
                    "spin_rad_s": cfg.spin_rad_s,
                },
            )
            env_cfg.viewer = ViewerCfg(eye=(1.5, -1.5, 1.2), lookat=(0.5, 0.0, 0.15))
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            scene=Scene(assets=[background, table, avocado, light]),
            embodiment=embodiment,
            task=AvocadoSettlingTask(cfg.episode_length_s, cfg.settled_steps),
            env_cfg_callback=configure_demo,
        )
