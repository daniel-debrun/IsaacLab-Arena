# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from copy import deepcopy

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.rigid_asset_preparation import align_rigid_body_paths


class RigidObjectSet(Object):
    """A set of rigid objects with one member selected per environment."""

    def __init__(
        self,
        name: str,
        objects: list[Object],
        prim_path: str | None = None,
        random_choice: bool = False,
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        """
        Args:
            name: The name of the object set.
            objects: The list of objects to be included in the object set. Each member spawns from a copy
                of its own object_cfg.spawn, taken when the set is built.
            prim_path: The prim path of the object set. Note that for all environments, the object set
                prim path must be the same.
            random_choice: Whether to randomly choose an object from the object set to spawn in
                each environment. If False, variants are assigned by repeating
                the member order across environments.
            initial_pose: The initial pose of the object from this object set.
        """
        assert len(objects) >= 1, f"Object set {name} must contain at least 1 object."
        for member in objects:
            assert not isinstance(
                member, RigidObjectSet
            ), f"Object set {name} cannot contain another object set ({member.name}); add its members instead."
            assert member.object_type == ObjectType.RIGID, f"Object set {name} must contain only rigid objects."

        # Isaac Lab finds the rigid body in one environment and uses its path in all of them,
        # so the members' spawn configurations are aligned to one rigid-body path.
        member_spawn_cfgs = align_rigid_body_paths([member.object_cfg.spawn for member in objects])
        if len(member_spawn_cfgs) == 1:
            spawn_cfg = member_spawn_cfgs[0]
        else:
            spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=member_spawn_cfgs,
                # Arena assigns members to environments. Isaac Lab ignores this flag and warns when it is True.
                random_choice=False,
                # A value here would replace every member's own setting.
                activate_contact_sensors=None,
            )

        self.objects: list[Object] = objects
        self.random_choice = random_choice
        self.variant_indices_by_env: list[int] | None = None

        if prim_path is None:
            prim_path = f"{{ENV_REGEX_NS}}/{name}"

        super().__init__(
            name=name,
            object_type=ObjectType.RIGID,
            spawner_cfg=spawn_cfg,
            prim_path=prim_path,
            initial_pose=initial_pose,
            **kwargs,
        )

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Return one local bbox for callers that cannot vary by env.

        The returned bbox has shape (1, 3) and uses the member with the
        greatest z-extent. Heterogeneous placement uses
        get_bounding_box_per_env() after assign_variants() so each env
        uses its actual variant geometry.
        """
        return max(self.objects, key=lambda obj: obj.get_bounding_box().size[0, 2].item()).get_bounding_box()

    def assign_variants(self, num_envs: int, variant_seed: int | None = None) -> None:
        """Fix one member-variant index per environment.

        The assignment is fixed for the lifetime of the object set so spawned
        USDs and per-env bboxes stay aligned across placement refills.
        Subsequent calls with the same num_envs are no-ops. A call with a
        different num_envs regenerates with a warning. When random_choice is True, each env
        independently samples one variant; otherwise assignments repeat the
        member order across environments.
        Regeneration is safe before the scene is spawned; afterwards, per-env
        bboxes can desync from the spawned USDs.

        Callers invoke this once num_envs is known, before reading
        variant_indices_by_env or get_bounding_box_per_env.

        Args:
            num_envs: Number of environments to assign variants for.
            variant_seed: Optional seed used when random_choice=True.
        """
        if self.variant_indices_by_env is not None:
            if len(self.variant_indices_by_env) == num_envs:
                return
            print(f"Warning: RigidObjectSet '{self.name}' regenerating variant assignments for {num_envs} envs.")
        self._set_variant_indices_by_env(self._generate_variant_indices(num_envs, variant_seed=variant_seed))

    def get_bounding_box_per_env(self, num_envs: int) -> AxisAlignedBoundingBox:
        """Return each env's actual variant bbox.

        Requires assign_variants(num_envs) to have been called first. The
        returned bbox has shape (num_envs, 3).

        Args:
            num_envs: Number of environments. Must match the assignment.

        Returns:
            AxisAlignedBoundingBox with min_point / max_point of
            shape (num_envs, 3).
        """
        assert self.variant_indices_by_env is not None, (
            f"RigidObjectSet '{self.name}' has no variant assignment; "
            "call assign_variants(num_envs) before get_bounding_box_per_env()."
        )
        assert len(self.variant_indices_by_env) == num_envs, (
            f"RigidObjectSet '{self.name}' got request for {num_envs} envs, "
            f"but is assigned for {len(self.variant_indices_by_env)} envs."
        )
        bounding_boxes = [obj.get_bounding_box() for obj in self.objects]

        min_pts = torch.stack([bounding_boxes[idx].min_point[0] for idx in self.variant_indices_by_env], dim=0)
        max_pts = torch.stack([bounding_boxes[idx].max_point[0] for idx in self.variant_indices_by_env], dim=0)
        return AxisAlignedBoundingBox(min_point=min_pts, max_point=max_pts)

    def get_contact_sensor_prim_path(self) -> str:
        """Return the contact-sensor path shared by all spawn entries of the set."""
        spawn_cfg = self.object_cfg.spawn
        if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
            member_spawn_cfgs = spawn_cfg.assets_cfg
        else:
            member_spawn_cfgs = [spawn_cfg]
        return self.prim_path + self._get_shared_contact_body_path(member_spawn_cfgs)

    def _generate_variant_indices(self, num_envs: int, variant_seed: int | None = None) -> list[int]:
        """Return one member index per env.

        Ordered sets repeat member order. Random sets sample independently per
        env, using a local generator when variant_seed is set.
        """
        n = len(self.objects)
        if not self.random_choice:
            return [env_idx % n for env_idx in range(num_envs)]
        if variant_seed is None:
            return torch.randint(low=0, high=n, size=(num_envs,)).tolist()
        generator = torch.Generator()
        generator.manual_seed(variant_seed)
        return torch.randint(low=0, high=n, size=(num_envs,), generator=generator).tolist()

    def _set_variant_indices_by_env(self, variant_indices_by_env: list[int]) -> None:
        """Validate and store each environment's member index, then give each environment its member's spawn entry."""
        n = len(self.objects)
        assert all(
            0 <= idx < n for idx in variant_indices_by_env
        ), f"RigidObjectSet '{self.name}' variant indices must be in [0, {n}); got {variant_indices_by_env}."
        self.variant_indices_by_env = variant_indices_by_env
        if isinstance(self.spawner_cfg, sim_utils.MultiAssetSpawnerCfg):
            # With one spawn entry per environment, Isaac Lab's default sequential cloning gives environment i
            # the entry i. Entries come from the declared members, so a second assignment does not compound;
            # set-wide settings made on the live config are kept.
            member_spawn_cfgs = self.spawner_cfg.assets_cfg
            spawn_cfgs_by_env = [deepcopy(member_spawn_cfgs[idx]) for idx in variant_indices_by_env]
            self.object_cfg.spawn = self.object_cfg.spawn.replace(assets_cfg=spawn_cfgs_by_env)

    def _generate_articulation_cfg(self):
        raise NotImplementedError("Articulation configuration is not supported for object sets")

    def _generate_base_cfg(self):
        raise NotImplementedError("Base configuration is not supported for object sets")

    def _generate_spawner_cfg(self):
        raise NotImplementedError("Spawner configuration is not supported for object sets")
