# PC upgrade assets

Geometry and generated label layouts: original work for Isaac Lab-Arena, 2026,
Apache-2.0 (the repository license). Rebuild with `build_assets.py` in this folder's
parent directory using the Arena interpreter. No downloaded CAD or paid models.

`nvidia.svg` is the NVIDIA eye/wordmark symbol extracted from the official
https://www.nvidia.com/en-us/ page on 2026-10-07; `nvidia.png` is its rasterization.
NVIDIA and GeForce RTX are NVIDIA trademarks. The original logo artwork retains
its owner's rights; it is not relicensed under Apache-2.0. It is used to identify
the requested simulated card. This scene does not imply endorsement.

RTX 5090 footprint reference (304 × 137 mm):
https://www.nvidia.com/en-au/geforce/graphics-cards/50-series/rtx-5090/
The model is representative geometry, not a mechanical or thermal replica.

VORTEX VX 880 and CINDER C700 are fictional labels used only in this scene.
The tower represents a vertical motherboard with a horizontal GPU inserted into
a side-facing PCIe socket. Its rear bracket ledge and anti-sag arm support the
card. Connector latching, power wiring and screw fastening are outside the task.

## How the assets were created

The scene props are custom, code-generated models. Their shapes, dimensions,
materials and details are explicitly authored in Python in
[`build_assets.py`](../build_assets.py), which writes USD geometry and PNG
textures. They were not downloaded as finished 3D models or sculpted in a
separate modeling application.

| Asset | Creation method and source |
| --- | --- |
| Tower PC and interior | Custom procedural USD geometry: case panels, hinged glass door, handle, motherboard, CPU cooler, RAM, PCIe slots, rear bracket ledge and GPU support. The script also authors the hinge joint and collision shapes. |
| NVIDIA RTX 5090 | Custom representative GPU model with a PCB, connector contacts, heatsink, two fans and branded surfaces. The published 304 × 137 mm footprint is a reference; the mesh is not NVIDIA CAD. The NVIDIA logo is the externally sourced artwork described above. |
| VORTEX VX 880 and CINDER C700 | Custom procedural models with fictional names and label designs. VORTEX uses three fans; CINDER uses a single blower. Their geometry shares the GPU builder with the RTX model. |
| Workbench and service mat | Custom procedural table, legs and mat. The oak grain is drawn mathematically into `oak.png`; the grid and instruction strip are generated locally, rather than sourced from photographs. |
| E-waste recycling bin | Custom procedural tapered shell, rounded rim, recessed handles and separate collision walls. The recycling arrows and text are drawn locally with Pillow. |
| Labels and materials | Pillow generates the PNG label layouts using DejaVu Sans fonts and the saved NVIDIA logo. USD Preview Surface materials define metal, plastic, PCB, rubber and glass appearance. No generative-image service was used for these textures. |
| Franka robot | Reused from the existing Isaac Lab Franka asset through Arena's `FrankaIKEmbodiment`; it was not modeled for this task. Its upstream asset distribution and terms remain separate from the original scene props. |
| Floor and lighting | Configured using Isaac Lab's cuboid, dome-light and disk-light spawners in `pc_upgrade_environment.py`. No downloaded environment map is used. |

Visual meshes contain the detailed surfaces; simpler collision geometry is
created separately where needed for stable simulation. The six generated USD
props and their textures are committed locally, so ordinary runs do not rebuild
them. The Franka robot continues to use the existing Isaac Lab asset setup.

To rebuild the props from the repository root inside the initialized Arena
runtime:

```bash
/isaac-sim/python.sh isaaclab_arena_environments/pc_upgrade/build_assets.py
```

The builder requires Pillow, USD Python bindings and DejaVu Sans fonts. It reads
the committed `nvidia.png`; it does not download or re-extract the logo. The
original `nvidia.svg` is retained alongside the rasterized PNG for provenance.
