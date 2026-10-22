# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configure automatic placement preparation before policy evaluation."""

from dataclasses import dataclass, field

from isaaclab_arena.offline_placement.settled_placement_params import SettledPlacementParams


@dataclass
class SettledPlacementPreparationCfg:
    """Collect one validated settled layout per evaluation episode before rollout.

    Standard robot joint reset noise is disabled in preparation and replay. Custom events,
    enabled variations and floating-base robots are unsupported.
    """

    max_batches: int = 100
    """Maximum candidate batches per rebuild; insufficient accepted layouts fail the run."""
    settle: SettledPlacementParams = field(
        default_factory=lambda: SettledPlacementParams(num_steps=120, required_settled_steps=10)
    )
    """Physics duration, sustained-rest window and post-physics acceptance checks."""

    def __post_init__(self):
        assert self.max_batches > 0, "max_batches must be positive"
