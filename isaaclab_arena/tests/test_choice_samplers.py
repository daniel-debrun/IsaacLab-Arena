# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ChoiceSampler draws with an explicit generator."""

import torch

from isaaclab_arena.variations.choice_sampler import ChoiceSampler


def test_choice_generator_is_reproducible_and_independent_of_global_random_state():
    sampler = ChoiceSampler()
    original_global_state = torch.random.get_rng_state()

    first_draw = sampler.sample(20, choices=range(100), generator=torch.Generator().manual_seed(42))
    assert torch.equal(torch.random.get_rng_state(), original_global_state)

    with torch.random.fork_rng(devices=[]):
        torch.rand(15)
        repeated_draw = sampler.sample(20, choices=range(100), generator=torch.Generator().manual_seed(42))
    assert repeated_draw == first_draw


def test_choice_replay_notifies_without_advancing_generator():
    sampler = ChoiceSampler()
    env_ids = torch.tensor([4, 1])
    replay_requests = []
    received_samples = []

    def replay_samples(num_samples, requested_env_ids):
        replay_requests.append((num_samples, requested_env_ids))
        return ["orange", "orange"]

    sampler.set_replay_sampler(replay_samples)
    sampler.add_listener(lambda sample, ids: received_samples.append((sample, ids)))
    generator = torch.Generator().manual_seed(11)
    original_generator_state = generator.get_state()

    samples = sampler.sample(2, choices=["banana", "orange"], env_ids=env_ids, generator=generator)

    assert samples == ["orange", "orange"]
    assert replay_requests[0][0] == 2
    assert replay_requests[0][1] is env_ids
    assert received_samples[0][0] is samples
    assert received_samples[0][1] is env_ids
    assert torch.equal(generator.get_state(), original_generator_state)
