# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental RGB renderer with an ordinary render product per environment."""

from dataclasses import dataclass, replace
from uuid import uuid4

from isaaclab.utils.warp import ProxyArray
from isaaclab_physx.renderers.isaac_rtx_renderer import IsaacRtxRenderData, IsaacRtxRenderer


@dataclass
class SeparateCameraRenderData:
    spec: object
    children: list


class SeparateCameraRenderer(IsaacRtxRenderer):
    """Render RGB cameras independently and write into the usual batched buffers."""

    def create_render_data(self, spec):
        import omni.replicator.core as rep

        assert set(spec.cfg.data_types) <= {"rgb", "rgba"}, "Prototype supports RGB/RGBA only"
        assert spec.cfg.isp_cfg is None, "Prototype does not support ISP"
        assert spec.cfg.background_color is None, "Prototype uses the scene background"
        children = []
        for path in spec.camera_prim_paths:
            child_spec = replace(spec, camera_prim_paths=(path,), num_instances=1, view_count=1)
            product = rep.create.render_product(
                camera=path,
                resolution=(spec.cfg.width, spec.cfg.height),
                name=f"independent_camera_{uuid4().hex}",
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device=spec.device, do_array_copy=False)
            annotator.attach([product.path])
            children.append(
                IsaacRtxRenderData(
                    annotators={"rgba": annotator},
                    render_product=product,
                    spec=child_spec,
                )
            )
        return SeparateCameraRenderData(spec=spec, children=children)

    def set_outputs(self, render_data, output_data):
        for i, child in enumerate(render_data.children):
            outputs = {name: ProxyArray(value.warp[i : i + 1]) for name, value in output_data.items()}
            super().set_outputs(child, outputs)

    def render(self, render_data):
        for child in render_data.children:
            super().render(child)

    def read_output(self, render_data, camera_data):
        for name in camera_data.info:
            camera_data.info[name] = None

    def cleanup(self, render_data):
        if render_data is not None:
            for child in render_data.children:
                if child.render_product is not None:
                    super().cleanup(child)
