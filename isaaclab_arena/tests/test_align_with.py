# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Alignment geometry, validation, and solver integration."""

import torch

import pytest

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.relation_loss_strategies import AlignWithLossStrategy
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import AlignWith
from isaaclab_arena.relations.validation.pre_physics import AlignWithValidator, build_validators
from isaaclab_arena.relations.validation.types import PlacementCheck
from isaaclab_arena.tests.dummy_object import DummyObject, make_candidate_batch
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def _objects():
    # Both origins differ from their geometric centres; equating origins is incorrect.
    reference = DummyObject("reference", AxisAlignedBoundingBox((1.0, 2.0, 3.0), (3.0, 4.0, 5.0)))
    subject = DummyObject("subject", AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.2, 0.4, 0.6)))
    return subject, reference


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("offset_m", [-0.2, 0.0, 0.3])
def test_alignment_uses_centres_and_leaves_other_axes_free(axis, offset_m):
    subject, reference = _objects()
    relation = AlignWith(reference, axis, offset_m=offset_m)
    axis_index = "xyz".index(axis)
    t_W_C = torch.tensor([-20.0, 10.0, 8.0])
    t_W_C[axis_index] = [1.9, 2.8, 3.7][axis_index] + offset_m
    strategy = AlignWithLossStrategy()
    assert strategy.compute_loss(relation, t_W_C, subject.bounding_box, reference.bounding_box).item() < 1e-4
    t_W_C[axis_index] += 0.1
    assert strategy.compute_loss(relation, t_W_C, subject.bounding_box, reference.bounding_box).item() == pytest.approx(
        8.0, abs=1e-4
    )


def test_alignment_batch_gradients_move_both_objects_together():
    subject, reference = _objects()
    relation = AlignWith(reference, "x", relation_loss_weight=2.0)
    t_W_C = torch.tensor([[2.0, 10.0, 8.0], [1.7, -10.0, -8.0]], requires_grad=True)
    t_W_P = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    parent_W = AxisAlignedBoundingBox(
        reference.bounding_box.min_point + t_W_P, reference.bounding_box.max_point + t_W_P
    )
    loss = AlignWithLossStrategy().compute_loss(relation, t_W_C, subject.bounding_box, parent_W)
    assert torch.allclose(loss, torch.tensor([16.0, 36.0]), atol=1e-4)
    loss.sum().backward()
    assert torch.equal(t_W_C.grad, torch.tensor([[200.0, 0.0, 0.0], [-200.0, 0.0, 0.0]]))
    assert torch.equal(t_W_P.grad, -t_W_C.grad)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_alignment_has_no_loss_or_gradient_inside_band(axis):
    bounds = AxisAlignedBoundingBox((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    reference = DummyObject("reference", bounds)
    relation = AlignWith(reference, axis, tolerance_m=0.125, relation_loss_weight=3.0)
    axis_index = "xyz".index(axis)
    origins_W = torch.zeros((7, 3))
    origins_W[:, axis_index] = torch.tensor([-0.25, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.25])
    origins_W.requires_grad_()
    loss = AlignWithLossStrategy(slope=20.0).compute_loss(relation, origins_W, bounds, bounds)
    assert torch.equal(loss, torch.tensor([7.5, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5]))
    loss.sum().backward()
    assert torch.equal(origins_W.grad[2:5], torch.zeros((3, 3)))
    assert origins_W.grad[0, axis_index] == -60.0
    assert origins_W.grad[-1, axis_index] == 60.0


def test_alignment_zero_tolerance_retains_exact_target_loss():
    bounds = AxisAlignedBoundingBox((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    reference = DummyObject("reference", bounds)
    relation = AlignWith(reference, "x", tolerance_m=0.0)
    strategy = AlignWithLossStrategy(slope=20.0)
    for x in (-0.25, 0.0, 0.25):
        loss = strategy.compute_loss(relation, torch.tensor([x, 0.0, 0.0]), bounds, bounds)
        assert loss.ndim == 0
        assert loss.item() == 20.0 * abs(x)


def test_alignment_validator_rejects_bad_candidates_and_nonfinite_positions():
    subject, reference = _objects()
    subject.add_relation(AlignWith(reference, "x", offset_m=-0.2, tolerance_m=0.025))
    bboxes = {subject: subject.bounding_box, reference: reference.bounding_box}
    positions = [{subject: (x, 9.0, 10.0), reference: (0.0, 0.0, 0.0)} for x in (1.7, 1.72, 1.74, float("nan"))]
    batch = make_candidate_batch(positions, [{}] * 4, [bboxes] * 4)
    params = ObjectPlacerParams(verbose=False)
    assert AlignWithValidator(params).validate_batch(batch, []) == [True, True, False, False]
    assert PlacementCheck.ALIGN_WITH in {validator.check for validator in build_validators(params)}


def test_alignment_failure_is_required_by_default():
    from isaaclab_arena.relations.placement_validation_runner import PlacementValidationRunner

    subject, reference = _objects()
    subject.add_relation(AlignWith(reference, "x", tolerance_m=0.02))
    positions = {subject: (2.0, 9.0, 10.0), reference: (0.0, 0.0, 0.0)}
    bboxes = {subject: subject.bounding_box, reference: reference.bounding_box}
    batch = make_candidate_batch([positions], [{}], [bboxes])
    params = ObjectPlacerParams(verbose=False)
    PlacementValidationRunner(params, build_validators(params)).validate_candidates(batch, [])
    results = batch.candidates[0].validation
    assert not results.validation_results[PlacementCheck.ALIGN_WITH]
    assert not results.do_all_required_validation_checks_pass()


def test_alignment_accepts_per_candidate_bounds_after_rotation():
    subject, reference = _objects()
    relation = AlignWith(reference, "x")
    # A 180-degree world-Z rotation moves the off-centre bounds to the opposite side of the origin.
    child_bounds = AxisAlignedBoundingBox(
        min_point=torch.tensor([[0.0, 0.0, 0.0], [-0.2, -0.4, 0.0]]),
        max_point=torch.tensor([[0.2, 0.4, 0.6], [0.0, 0.0, 0.6]]),
    )
    origins_W = torch.tensor([[1.9, 0.0, 0.0], [2.1, 0.0, 0.0]])
    loss = AlignWithLossStrategy().compute_loss(relation, origins_W, child_bounds, reference.bounding_box)
    assert torch.allclose(loss, torch.zeros(2), atol=1e-4)


@pytest.mark.parametrize(
    "params",
    [
        {"axis": "xy"},
        {"axis": "x", "offset_m": float("nan")},
        {"axis": "x", "tolerance_m": -0.1},
        {"axis": "x", "tolerance_m": float("inf")},
    ],
)
def test_alignment_rejects_invalid_parameters(params):
    _, reference = _objects()
    with pytest.raises(AssertionError):
        AlignWith(reference, **params)


def test_alignment_requires_a_distinct_participating_reference():
    subject, reference = _objects()
    with pytest.raises(AssertionError, match="different"):
        AlignWith(subject, "x").validate_placement_configuration(subject, {subject})
    with pytest.raises(AssertionError, match="participate"):
        AlignWith(reference, "x").validate_placement_configuration(subject, {subject})


def test_alignment_yaml_parameters_reach_the_registered_relation():
    import yaml

    from isaaclab_arena.assets.registries import ObjectRelationLibraryRegistry
    from isaaclab_arena.environment_spec.arena_env_graph_types import SpatialRelationSpec

    entry = SpatialRelationSpec.model_validate(yaml.safe_load("""
kind: align_with
subject: robot
reference: bowl
params:
  axis: y
  offset_m: -0.05
  tolerance_m: 0.03
"""))
    _, reference = _objects()
    relation_type = ObjectRelationLibraryRegistry().get_object_relation_by_name(entry.kind)
    relation = relation_type(reference, **entry.params)
    assert isinstance(relation, AlignWith)
    assert relation.parent is reference
    assert relation.axis == "y"
    assert relation.offset_m == -0.05
    assert relation.tolerance_m == 0.03
    with pytest.raises(ValueError, match="reference"):
        SpatialRelationSpec(kind="align_with", subject="robot")


def test_alignment_solver_handles_moving_reference_and_different_heights():
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.pose import Pose

    table = DummyObject("table", AxisAlignedBoundingBox((-1.0, -1.0, -0.1), (1.0, 1.0, 0.0)))
    table.set_initial_pose(Pose.identity(), create_reset_event=False)
    table.add_relation(IsAnchor())
    subject = DummyObject("subject", AxisAlignedBoundingBox((-0.1, -0.1, 0.0), (0.1, 0.1, 0.2)))
    reference = DummyObject("reference", AxisAlignedBoundingBox((-0.1, -0.1, 0.0), (0.1, 0.1, 0.2)))
    subject.add_relation(On(table, clearance_m=0.2))
    reference.add_relation(On(table))
    subject.add_relation(AlignWith(reference, "x", offset_m=0.1, tolerance_m=0.02))
    initial = {table: (0.0, 0.0, 0.0), subject: (-0.5, -0.5, 0.2), reference: (0.5, 0.5, 0.01)}
    solver = RelationSolver(RelationSolverParams(verbose=False, save_position_history=False))
    final = solver.solve([table, subject, reference], [initial])[0]
    assert abs(final[subject][0] - final[reference][0] - 0.1) <= 0.02
    assert final[subject][2] > final[reference][2] + 0.1


def test_alignment_overrides_next_to_soft_cross_axis_preference():
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.relations import IsAnchor, NextTo, On, PositionLimitsBox
    from isaaclab_arena.utils.pose import Pose

    counter = DummyObject("counter", AxisAlignedBoundingBox((-3.0, -0.5, -0.1), (3.0, 0.5, 0.0)))
    counter.set_initial_pose(Pose.identity(), create_reset_event=False)
    counter.add_relation(IsAnchor())
    target = DummyObject("target", AxisAlignedBoundingBox((-0.1, -0.1, 0.0), (0.1, 0.1, 0.2)))
    robot = DummyObject("robot", AxisAlignedBoundingBox((-0.2, -0.2, 0.0), (0.2, 0.2, 0.8)))
    target.add_relation(On(counter))
    target.add_relation(PositionLimitsBox(x_min=2.0, x_max=2.2))
    robot.add_relation(NextTo(counter, side="negative_y", distance_m=0.2))
    robot.add_relation(AlignWith(target, axis="x", tolerance_m=0.04))
    initial = {counter: (0.0, 0.0, 0.0), target: (2.1, 0.0, 0.01), robot: (0.0, -0.9, 0.0)}
    solver = RelationSolver(RelationSolverParams(verbose=False, save_position_history=False))
    final = solver.solve([counter, target, robot], [initial])[0]
    assert abs(final[robot][0] - final[target][0]) <= 0.04
    assert 1.99 <= final[target][0] <= 2.21
