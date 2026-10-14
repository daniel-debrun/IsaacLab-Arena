# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify parallel camera compression preserves native episode data and reader compatibility."""

import h5py
import numpy as np
import torch

import pytest
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from isaaclab_arena.recording.parallel_camera_hdf5_file_handler import ParallelCameraHDF5DatasetFileHandler


@pytest.mark.parametrize("dataset_compression", [True, False])
@pytest.mark.parametrize("custom_ids", [True, False])
def test_camera_export_matches_native_writer(tmp_path, dataset_compression, custom_ids):
    """Native readers must recover every image, state, and attribute across successive episodes."""
    episode = EpisodeData()
    # Odd dimensions exercise partially filled boundary chunks; transposing makes input strided.
    rgb = torch.randint(0, 256, (17, 131, 79, 3), dtype=torch.uint8).transpose(1, 2)
    episode.data = {
        "actions": torch.rand(17, 7),
        "states": {"robot": {"joint_position": torch.rand(17, 7)}},
        "camera_obs": {
            "rgb": rgb,
            "nested": {"depth": torch.rand(17, 79, 131, 1)},
            "empty": torch.empty((0, 79, 131, 3), dtype=torch.uint8),
        },
    }
    episode.seed = 42
    episode.success = True
    original_data = episode.data
    paths = [tmp_path / "native.hdf5", tmp_path / "parallel.hdf5"]
    for writer_type, path in zip((HDF5DatasetFileHandler, ParallelCameraHDF5DatasetFileHandler), paths, strict=True):
        writer = writer_type()
        writer.create(str(path), env_name="gear_insertion")
        try:
            for demo_id in [7, 11] if custom_ids else [None, None]:
                writer.write_episode(episode, demo_id=demo_id, dataset_compression=dataset_compression)
            writer.flush()
            assert writer.get_num_episodes() == (0 if custom_ids else 2)
        finally:
            writer.close()
    assert episode.data is original_data
    assert episode.data["camera_obs"]["rgb"] is rgb

    with h5py.File(paths[0], "r") as expected, h5py.File(paths[1], "r") as actual:

        def compare(name, original):
            exported = actual[name]
            assert dict(exported.attrs) == dict(original.attrs)
            if isinstance(original, h5py.Dataset):
                assert exported.dtype == original.dtype
                assert exported.chunks == original.chunks
                assert exported.compression == original.compression
                assert exported.compression_opts == original.compression_opts
                np.testing.assert_array_equal(exported[:], original[:])

        expected.visititems(compare)
        assert actual["data"].attrs["total"] == 34

    reader = HDF5DatasetFileHandler()
    reader.open(str(paths[1]))
    try:
        loaded = reader.load_episode("demo_7" if custom_ids else "demo_0", device="cpu")
        torch.testing.assert_close(loaded.data["camera_obs"]["rgb"], rgb)
        assert loaded.seed == 42
        assert loaded.success
    finally:
        reader.close()


@pytest.mark.parametrize("camera_only", [True, False])
def test_empty_episode_and_single_group_exports(tmp_path, camera_only):
    """Empty episodes are skipped and camera-only or state-only recordings remain valid."""
    writer = ParallelCameraHDF5DatasetFileHandler(compression_workers=1)
    path = tmp_path / "dataset.hdf5"
    writer.create(str(path))
    try:
        writer.write_episode(EpisodeData())
        assert writer.get_num_episodes() == 0
        episode = EpisodeData()
        episode.data = (
            {"camera_obs": {"rgb": torch.zeros(1, 5, 7, 3, dtype=torch.uint8)}}
            if camera_only
            else {"actions": torch.ones(2, 7)}
        )
        writer.write_episode(episode)
        assert writer.get_num_episodes() == 1
    finally:
        writer.close()
    with h5py.File(path, "r") as dataset:
        assert len(dataset["data"]) == 1
        assert dataset["data/demo_0"].attrs["num_samples"] == (0 if camera_only else 2)
