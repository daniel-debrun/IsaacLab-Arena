# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Definitions for recurring observations independent of task success."""

from dataclasses import dataclass

from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.progress_tracking.progress_tracking_utils import Predicate


@dataclass
class TrackedEventCfg:
    """Record an ordered predicate sequence each time it completes during an episode."""

    name: str
    """Unique name within the task, used in recorded event occurrences."""

    predicate_sequence: list[Predicate]
    """Nonempty ordered predicates; use a one-element list for a single condition.

    Earlier predicates need not remain true when the sequence completes. The final
    predicate must become false before this event resets and can occur again.
    """

    def __post_init__(self):
        assert isinstance(self.name, str) and self.name.strip(), "Tracked events require a nonempty name."
        criteria = CompletionCriteria(name=self.name, predicate_sequence=self.predicate_sequence)
        for sequence in criteria.canonical_predicate_sequences.values():
            assert not any(
                isinstance(predicate, type) for predicate, _ in sequence
            ), "Configure predicate classes with TerminationTermCfg(func=MyPredicate)."


@dataclass(frozen=True)
class TrackedEventOccurrence:
    """One completed occurrence, retained until the environment starts a new episode."""

    env_idx: int
    """Index of the parallel environment."""

    name: str
    """Name of the event that occurred."""

    step: int
    """Episode control step when the event occurred."""
