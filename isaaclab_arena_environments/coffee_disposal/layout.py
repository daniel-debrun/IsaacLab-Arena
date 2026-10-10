# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Metre-scale geometry contract; machine and cabinet fronts face negative Y."""

import math

COUNTER_Z = 0.90
MACHINE_POSITION = (0.66, 0.28, COUNTER_Z)
CABINET_POSITION = (-0.30, -0.18, 0.12)
TRAY_IN_MACHINE = (0.0, -0.10, 0.012)
CUP_IN_TRAY = (0.0, 0.055, 0.014)
TRAY_EXTENSION = -0.40
SPOUT_RAISED = 0.065
DRAWER_EXTENSION = -0.55
BIN_IN_DRAWER = (0.12, -0.02, 0.16)
BIN_HALF_INTERIOR = (0.102, 0.142)
BIN_HEIGHT = 0.38
BIN_FLOOR_THICKNESS = 0.012
LID_IN_DRAWER = (0.12, -0.02, 0.545)
LID_PARK_IN_DRAWER = (-0.19, -0.02, 0.545)
CUP_SIZE = (0.144, 0.160, 0.110)
LID_SIZE = (0.232, 0.320, 0.012)
GROUND_RADIUS = 0.018
GROUND_ROWS = 14
GROUND_COLUMNS = 24
GROUND_VERTICES = []
for row in range(GROUND_ROWS + 1):
    phi = math.pi * row / GROUND_ROWS
    for column in range(GROUND_COLUMNS):
        theta = math.tau * column / GROUND_COLUMNS
        radius = GROUND_RADIUS * (0.88 + 0.08 * math.sin(theta * 5 + phi * 7) + 0.035 * math.cos(theta * 9))
        GROUND_VERTICES.append((
            radius * math.sin(phi) * math.cos(theta),
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi) * 0.83,
        ))
GROUND_VERTICES = tuple(GROUND_VERTICES)

# A rhombicuboctahedron has 24 vertices and 26 planar faces, below both GPU cooking limits.
GROUND_COLLISION_FACES = (
    (8, 0, 16),
    (1, 0, 8, 9),
    (17, 1, 9),
    (16, 0, 2, 18),
    (2, 0, 1, 3),
    (3, 1, 17, 19),
    (18, 2, 10),
    (10, 2, 3, 11),
    (11, 3, 19),
    (12, 8, 16, 20),
    (9, 8, 12, 13),
    (17, 9, 13, 21),
    (20, 16, 18, 22),
    (19, 17, 21, 23),
    (18, 10, 14, 22),
    (14, 10, 11, 15),
    (15, 11, 19, 23),
    (20, 4, 12),
    (12, 4, 5, 13),
    (13, 5, 21),
    (6, 4, 20, 22),
    (5, 4, 6, 7),
    (21, 5, 7, 23),
    (14, 6, 22),
    (7, 6, 14, 15),
    (23, 7, 15),
)


def _make_ground_collision_vertices():
    """Return a convex low-polygon enclosure of the detailed coffee-clump surface."""
    vertices = []
    for major_axis in range(3):
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    point = [x, y, z]
                    point[major_axis] *= 1 + math.sqrt(2)
                    point[2] *= 0.83
                    vertices.append(point)
    scale = 0
    for face in GROUND_COLLISION_FACES:
        a, b, c = (vertices[index] for index in face[:3])
        u = [b[index] - a[index] for index in range(3)]
        v = [c[index] - a[index] for index in range(3)]
        normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        length = math.sqrt(sum(value * value for value in normal))
        normal = [value / length for value in normal]
        distance = sum(normal[index] * a[index] for index in range(3))
        for point in GROUND_VERTICES:
            support = sum(normal[index] * point[index] for index in range(3))
            scale = max(scale, (support + 0.00005) / distance)
    return tuple(tuple(value * scale for value in point) for point in vertices)


GROUND_COLLISION_VERTICES = _make_ground_collision_vertices()

GROUND_NAMES = tuple(f"grounds_{index}" for index in range(8))
CUP_HOME = tuple(a + b + c for a, b, c in zip(MACHINE_POSITION, TRAY_IN_MACHINE, CUP_IN_TRAY))
LID_HOME = tuple(a + b for a, b in zip(CABINET_POSITION, LID_IN_DRAWER))
GROUND_POSITIONS = {}
for index, name in enumerate(GROUND_NAMES):
    x = (-0.032, 0.032)[index % 2]
    y = (-0.033, 0.033)[(index // 2) % 2]
    z = 0.026 + 0.036 * (index // 4)
    GROUND_POSITIONS[name] = (CUP_HOME[0] + x, CUP_HOME[1] + y, CUP_HOME[2] + z)
