# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Measure image zoom from matched render features and create comparison figures."""

import argparse
import json
import matplotlib
import numpy as np
from pathlib import Path

import cv2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--capture-root", type=Path, required=True)
root = parser.parse_args().capture_root
results = {}


def measure(reference, image):
    sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in pairs if a.distance < 0.7 * b.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    assert matrix is not None and mask.sum() >= 10, "Insufficient matched image features"
    projected = p1 @ matrix[:, :2].T + matrix[:, 2]
    residual = np.linalg.norm(projected - p2, axis=1)[mask[:, 0].astype(bool)]
    return {
        "rendered_scale": float(np.linalg.norm(matrix[:, 0])),
        "inliers": int(mask.sum()),
        "median_error_px": float(np.median(residual)),
        "affine": matrix.tolist(),
    }


def read(folder, label, i):
    return cv2.cvtColor(cv2.imread(str(folder / f"{label}_env{i}.png")), cv2.COLOR_BGR2RGB)


for name in ["single", "multi"]:
    folder = root / (name + "_captures")
    data = json.loads((folder / "measurements.json").read_text())
    num_envs = data["num_envs"]
    by_label = {c["label"]: c for c in data["captures"]}
    measured = {}
    for label in ["wide", "tele", "restored"] + (["mixed", "env0_only"] if num_envs > 1 else []):
        measured[label] = []
        for i in range(num_envs):
            item = measure(read(folder, "baseline", i), read(folder, label, i))
            item["reported_scale"] = by_label[label]["focal_scales"][i]
            measured[label].append(item)
    # Check fixed-scene camera poses really stayed unchanged.
    baseline = by_label["baseline"]
    for label in measured:
        assert np.allclose(by_label[label]["camera_t_W"], baseline["camera_t_W"], atol=1e-6)
        assert np.allclose(by_label[label]["camera_q_W_C"], baseline["camera_q_W_C"], atol=1e-6)
    results[name] = measured

    if name == "single":
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        for ax, label, title in zip(axes, ["wide", "baseline", "tele"], ["Wide", "Nominal", "Tele"]):
            scale = by_label[label]["focal_scales"][0]
            observed = 1.0 if label == "baseline" else measured[label][0]["rendered_scale"]
            fx = by_label[label]["intrinsic_matrices"][0][0][0]
            ax.imshow(read(folder, label, 0))
            ax.set_title(f"{title}: fx = fy = {fx:.1f} px\nRequested {scale:.2f}x | image {observed:.2f}x", fontsize=13)
            ax.axis("off")
        fig.suptitle("Single environment: changing intrinsics changes the rendered zoom", fontsize=17)
        fig.tight_layout()
        fig.savefig(root / "single_comparison.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    else:
        for output_name, labels, heading in [
            (
                "multi_comparison",
                ["baseline", "mixed"],
                "Separate render products: requested and rendered zoom per environment",
            ),
            ("multi_env0_control", ["mixed", "env0_only"], "Separate render products: changing only env 0 intrinsics"),
        ]:
            fig, axes = plt.subplots(2, 4, figsize=(20, 7))
            for row, label in enumerate(labels):
                for i in range(num_envs):
                    ax = axes[row, i]
                    scale = by_label[label]["focal_scales"][i]
                    observed = 1.0 if label == "baseline" else measured[label][i]["rendered_scale"]
                    ax.imshow(read(folder, label, i))
                    title = f'Env {i} | {label.replace("_", " ")}\nReported {scale:.2f}x | image {observed:.2f}x'
                    ax.set_title(title, fontsize=12)
                    ax.axis("off")
            fig.suptitle(heading, fontsize=16)
            fig.tight_layout()
            fig.savefig(root / f"{output_name}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
(root / "image_analysis.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
