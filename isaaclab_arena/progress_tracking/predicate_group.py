# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Definitions for recurring observations independent of task success."""

from dataclasses import dataclass

from isaaclab_arena.progress_tracking.progress_tracking_utils import Predicate, _is_predicate


@dataclass
class TrackedEventCfg:
    """Record recurring completions, requiring the observed condition to clear between occurrences."""

    name: str
    """Unique name within the task, used in recorded event occurrences."""

    predicates: list[Predicate]
    """Predicates whose completion history is cleared before each new occurrence."""

    predicates_are_sequential: bool = False
    """Require list order; the next predicate starts on the following control step."""

    desired_predicate_state: list[bool | None] | None = None
    """Optional current states in addition to history; None entries ignore that member.

    A False entry requires the predicate to complete first, then become false.
    Ignored members still gate later members when predicates_are_sequential is True.
    Omitting the list uses completion history alone.
    """

    def __post_init__(self):
        assert isinstance(self.name, str) and self.name.strip(), "Tracked events require a nonempty name."
        assert (
            isinstance(self.predicates, list) and self.predicates
        ), "Tracked events require a nonempty predicate list."
        assert all(_is_predicate(predicate) for predicate in self.predicates), "Invalid event predicate."
        assert not any(
            isinstance(predicate, type) for predicate in self.predicates
        ), "Configure predicate classes with TerminationTermCfg(func=MyPredicate)."
        assert isinstance(self.predicates_are_sequential, bool), "predicates_are_sequential must be a bool."
        if self.desired_predicate_state is not None:
            assert len(self.desired_predicate_state) == len(self.predicates), "Provide one desired state per predicate."
            assert all(
                state is None or isinstance(state, bool) for state in self.desired_predicate_state
            ), "Desired predicate states must be True, False, or None."
            assert any(state is not None for state in self.desired_predicate_state), "At least one member is required."


@dataclass(frozen=True)
class TrackedEventOccurrence:
    """One completed occurrence, retained until the environment starts a new episode."""

    env_idx: int
    """Index of the parallel environment."""

    name: str
    """Name of the event whose predicate group completed."""

    step: int
    """Episode control step when the group completed."""
