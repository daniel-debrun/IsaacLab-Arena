# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box no-overlap collision loss computation."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import CollisionMode, object_uses_mesh_collision
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import On
from isaaclab_arena.utils.bounding_box import ProjectedOverlapVolume

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache


@dataclass(frozen=True)
class NoOverlapPair:
    """One directed penalty, with gradients applied only to the subject."""

    subject: PlaceableAsset
    """Asset whose position receives this penalty's gradient."""
    obstacle: PlaceableAsset | CollisionObject
    """Object treated as fixed for this directed penalty."""


@dataclass(frozen=True)
class NoOverlapPairCache:
    """Fixed pair routing and face projections for one solve and collision configuration."""

    box_pairs: list[NoOverlapPair]
    """Pairs whose collision geometry matches box validation, in directed penalty order."""
    mesh_fallback_pairs: list[NoOverlapPair]
    """Pairs using enclosing-box mesh proxies, in directed penalty order."""
    projections: ProjectedOverlapVolume | None
    """Prepared projections for box_pairs, or None when every collision box is an AxisAlignedBoundingBox."""


def compute_no_overlap_loss_bbox(
    state: RelationSolverState,
    no_collision_strategy: NoCollisionLossStrategy,
    clearance_m: float,
    mesh_manager: WarpMeshAndSphereCache | None,
    default_collision_mode: CollisionMode = CollisionMode.BBOX,
    skip_mesh_pairs: bool = False,
    debug: bool = False,
) -> tuple[torch.Tensor, int]:
    """Sum projected-overlap penalties over directed pairs, returning (B,) losses and pair count.

    Movable pairs contribute in both directions. On-linked pairs and pairs handled
    by MESH are excluded. Fixed obstacles receive no position gradients.

    Args:
        state: Current positions and fixed collision geometry for one solve.
        no_collision_strategy: Penalty scale and aligned-box interval loss.
        clearance_m: Nonnegative inflation of each directed pair's obstacle projections.
        mesh_manager: Collision mesh cache, or None when no meshes are available.
        default_collision_mode: Mode for objects without an explicit override.
        skip_mesh_pairs: Exclude pairs represented by the mesh collision loss.
        debug: Print the mean loss for each directed pair.

    Returns:
        Per-candidate losses of shape (B,) and the number of directed pairs.
    """
    cache = state.get_no_overlap_pair_cache(mesh_manager, default_collision_mode, skip_mesh_pairs)
    total = torch.zeros(state.batch_size, device=state.device)
    for group, enclosing_bounds in ((cache.box_pairs, False), (cache.mesh_fallback_pairs, True)):
        if not group:
            continue
        if enclosing_bounds or state.collision_boxes_axis_aligned:
            pair_loss = _compute_aabb_pair_losses(state, group, no_collision_strategy, clearance_m, enclosing_bounds)
        else:
            pair_loss = _compute_oriented_pair_losses(
                state, group, cache.projections, no_collision_strategy, clearance_m
            )
        total = total + pair_loss.sum(dim=0)
        if debug:
            for pair, loss in zip(group, pair_loss):
                print(f"  [NoOverlap] {pair.subject.name} vs {pair.obstacle.name}: loss={loss.mean().item():.6f}")
    return total, len(cache.box_pairs) + len(cache.mesh_fallback_pairs)


def prepare_no_overlap_pair_cache(
    state: RelationSolverState,
    mesh_manager: WarpMeshAndSphereCache | None,
    default_collision_mode: CollisionMode,
    skip_mesh_pairs: bool,
) -> NoOverlapPairCache:
    """Prepare fixed pair routing and projections after mesh geometry is available.

    Args:
        state: Solver state with fixed collision bounds and object relations.
        mesh_manager: Cache supplying available collision meshes, or None.
        default_collision_mode: Mode for objects without an explicit override.
        skip_mesh_pairs: Exclude pairs represented by the mesh collision loss.

    Returns:
        Ordered pair groups and prepared projections for this solve.
    """
    movable = state.optimizable_objects
    anchors = list(state.anchor_objects)
    fixed = anchors + state.collision_objects
    on_pairs = set()
    for obj in [*movable, *anchors]:
        for relation in obj.get_relations():
            if isinstance(relation, On):
                on_pairs.add((id(obj), id(relation.parent)))
                on_pairs.add((id(relation.parent), id(obj)))

    pairs = []
    for child in movable:
        for obstacle in fixed:
            if (id(child), id(obstacle)) in on_pairs:
                continue
            if (
                skip_mesh_pairs
                and mesh_manager is not None
                and _fixed_pair_is_covered_by_mesh_collision(
                    state, child, obstacle, mesh_manager, default_collision_mode
                )
            ):
                continue
            pairs.append(NoOverlapPair(child, obstacle))
    for index, child in enumerate(movable):
        for other in movable[index + 1 :]:
            if (id(child), id(other)) in on_pairs:
                continue
            if (
                skip_mesh_pairs
                and mesh_manager is not None
                and _dynamic_pair_is_covered_by_mesh_collision(
                    state, child, other, mesh_manager, default_collision_mode
                )
            ):
                continue
            pairs.extend((NoOverlapPair(child, other), NoOverlapPair(other, child)))
    if not pairs:
        return NoOverlapPairCache([], [], None)

    mesh_objects = set()
    if skip_mesh_pairs and mesh_manager is not None:
        for obj in [*movable, *fixed]:
            if (
                object_uses_mesh_collision(obj, default_collision_mode)
                and mesh_manager.get_collision_mesh(obj) is not None
            ):
                mesh_objects.add(obj)
    box_pairs, mesh_fallback_pairs = [], []
    background_objects = set(state.collision_objects)
    for pair in pairs:
        uses_mesh_validation = pair.obstacle in mesh_objects or (
            pair.subject in mesh_objects and pair.obstacle not in background_objects
        )
        if uses_mesh_validation:
            # Mesh validation uses enclosing-box sphere proxies for these unsupported pairs.
            mesh_fallback_pairs.append(pair)
        else:
            box_pairs.append(pair)

    projections = None
    if box_pairs and not state.collision_boxes_axis_aligned:
        geometry = {}
        for pair in box_pairs:
            for obj in (pair.subject, pair.obstacle):
                if obj not in geometry:
                    geometry[obj] = state.get_collision_geometry(obj)
        projections = ProjectedOverlapVolume(
            torch.stack([geometry[p.subject][1] for p in box_pairs]),
            torch.stack([geometry[p.subject][2] for p in box_pairs]),
            torch.stack([geometry[p.obstacle][1].detach() for p in box_pairs]),
            torch.stack([geometry[p.obstacle][2].detach() for p in box_pairs]),
        )
    return NoOverlapPairCache(box_pairs, mesh_fallback_pairs, projections)


def _compute_aabb_pair_losses(
    state: RelationSolverState,
    pairs: list[NoOverlapPair],
    strategy: NoCollisionLossStrategy,
    clearance_m: float,
    enclosing_bounds: bool,
) -> torch.Tensor:
    """Return (P, B) interval-volume penalties, keeping each obstacle detached."""
    extents = {}
    for obj in state.optimizable_objects:
        box = state.get_enclosing_collision_bbox(obj) if enclosing_bounds else state.get_collision_bbox(obj)
        offset = state.get_position(obj)
        extents[obj] = (
            (box.min_point + offset).expand(state.batch_size, 3),
            (box.max_point + offset).expand(state.batch_size, 3),
        )
    for obj in [*state.anchor_objects, *state.collision_objects]:
        box = state.get_enclosing_collision_bbox(obj) if enclosing_bounds else state.get_collision_bbox(obj)
        extents[obj] = (box.min_point.expand(state.batch_size, 3), box.max_point.expand(state.batch_size, 3))
    return strategy.compute_loss_batched(
        clearance_m,
        torch.stack([extents[p.subject][0] for p in pairs]),
        torch.stack([extents[p.subject][1] for p in pairs]),
        torch.stack([extents[p.obstacle][0].detach() for p in pairs]),
        torch.stack([extents[p.obstacle][1].detach() for p in pairs]),
    )


def _compute_oriented_pair_losses(
    state: RelationSolverState,
    pairs: list[NoOverlapPair],
    projections: ProjectedOverlapVolume,
    strategy: NoCollisionLossStrategy,
    clearance_m: float,
) -> torch.Tensor:
    """Return (P, B) face-projection penalties from cached rotations and current positions."""
    centers = {}
    for obj in [*state.optimizable_objects, *state.anchor_objects, *state.collision_objects]:
        center, _, _ = state.get_collision_geometry(obj)
        if obj in state.optimizable_objects:
            center = center + state.get_position(obj)
        centers[obj] = center
    subject_centers = torch.stack([centers[p.subject] for p in pairs])
    obstacle_centers = torch.stack([centers[p.obstacle].detach() for p in pairs])
    return strategy.slope * projections.compute(subject_centers, obstacle_centers, margin=clearance_m)


def _fixed_pair_is_covered_by_mesh_collision(
    state: RelationSolverState,
    subject: PlaceableAsset,
    obstacle: CollisionObject,
    mesh_manager: WarpMeshAndSphereCache,
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when MESH loss handles subject vs fixed obstacle."""
    obstacle_mesh = (
        mesh_manager.get_collision_mesh(obstacle)
        if object_uses_mesh_collision(obstacle, default_collision_mode)
        else None
    )
    return obstacle_mesh is not None and _has_mesh_or_invariant_bbox(
        state, subject, mesh_manager, default_collision_mode
    )


def _dynamic_pair_is_covered_by_mesh_collision(
    state: RelationSolverState,
    a: PlaceableAsset,
    b: PlaceableAsset,
    mesh_manager: WarpMeshAndSphereCache,
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when MESH loss handles a non-anchor object pair."""
    a_mesh = mesh_manager.get_collision_mesh(a) if object_uses_mesh_collision(a, default_collision_mode) else None
    b_mesh = mesh_manager.get_collision_mesh(b) if object_uses_mesh_collision(b, default_collision_mode) else None
    if a_mesh is None and b_mesh is None:
        return False
    return _has_mesh_or_invariant_bbox(state, a, mesh_manager, default_collision_mode) and _has_mesh_or_invariant_bbox(
        state, b, mesh_manager, default_collision_mode
    )


def _has_mesh_or_invariant_bbox(
    state: RelationSolverState,
    obj: PlaceableAsset,
    mesh_manager: WarpMeshAndSphereCache,
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when MESH loss can represent obj as mesh or one bbox proxy."""
    mesh = mesh_manager.get_collision_mesh(obj) if object_uses_mesh_collision(obj, default_collision_mode) else None
    if mesh is not None:
        return True
    return state.get_bbox(obj).is_batch_invariant()
