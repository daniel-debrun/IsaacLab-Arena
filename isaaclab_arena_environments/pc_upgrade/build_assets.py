# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the original PC service-bench assets using USD and Pillow inside Arena's runtime."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ASSETS = Path(__file__).with_name("assets")


class Model:
    """Author a metre-scale Z-up model with reusable materials and simple convex collisions."""

    def __init__(self, name):
        self.stage = Usd.Stage.CreateNew(str(ASSETS / f"{name}.usda"))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        self.root = UsdGeom.Xform.Define(self.stage, "/Asset").GetPrim()
        self.stage.SetDefaultPrim(self.root)
        self.materials = {}
        for key, color, metallic, roughness in [
            ("steel", (0.28, 0.31, 0.35), 0.85, 0.26),
            ("black", (0.018, 0.023, 0.029), 0.55, 0.3),
            ("rubber", (0.008, 0.012, 0.016), 0, 0.8),
            ("silver", (0.60, 0.65, 0.7), 0.95, 0.2),
            ("pcb", (0.015, 0.055, 0.035), 0.15, 0.55),
            ("gold", (0.8, 0.48, 0.12), 0.9, 0.23),
            ("green", (0.30, 0.65, 0.005), 0.3, 0.3),
            ("red", (0.6, 0.075, 0.025), 0.4, 0.3),
            ("blue", (0.035, 0.23, 0.65), 0.4, 0.3),
            ("wood", (0.30, 0.20, 0.105), 0, 0.6),
            ("mat", (0.035, 0.075, 0.08), 0, 0.88),
        ]:
            mat = UsdShade.Material.Define(self.stage, f"/Asset/Materials/{key}")
            shader = UsdShade.Shader.Define(self.stage, f"{mat.GetPath()}/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self.materials[key] = mat

    def body(self, path, mass):
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        return prim

    def box(self, path, position, size, material="black", collision=False):
        shape = UsdGeom.Cube.Define(self.stage, path)
        shape.CreateSizeAttr(1)
        shape.AddTranslateOp().Set(Gf.Vec3d(*position))
        shape.AddScaleOp().Set(Gf.Vec3d(*size))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(self.materials[material])
        if collision:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        return shape

    def cylinder(self, path, position, radius, height, material="steel"):
        shape = UsdGeom.Cylinder.Define(self.stage, path)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        shape.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(self.materials[material])
        return shape

    def fan(self, path, position, radius):
        x, y, z = position
        self.cylinder(f"{path}/Housing", position, radius, 0.004, "steel")
        self.cylinder(f"{path}/Well", (x, y, z + 0.003), radius * 0.94, 0.004, "rubber")
        for index in range(11):
            angle = index * 2 * math.pi / 11
            blade = self.box(
                f"{path}/Blade{index}",
                (x + math.cos(angle) * radius * 0.5, y + math.sin(angle) * radius * 0.5, z + 0.007),
                (radius * 0.78, radius * 0.16, 0.002),
                "black",
            )
            blade.AddRotateZOp().Set(math.degrees(angle) + 25)
            translation, scale, rotation = blade.GetOrderedXformOps()
            blade.SetXformOpOrder([translation, rotation, scale])
        self.cylinder(f"{path}/Hub", (x, y, z + 0.009), radius * 0.23, 0.005, "silver")
        self.cylinder(f"{path}/HubCap", (x, y, z + 0.012), radius * 0.18, 0.002, "black")

    def label(self, path, texture, position, width, height):
        x, y, z = position
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([
            (x - width / 2, y - height / 2, z),
            (x + width / 2, y - height / 2, z),
            (x + width / 2, y + height / 2, z),
            (x - width / 2, y + height / 2, z),
        ])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            [(0, 0), (1, 0), (1, 1), (0, 1)]
        )
        material = UsdShade.Material.Define(self.stage, path + "Material")
        shader = UsdShade.Shader.Define(self.stage, path + "Material/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        reader = UsdShade.Shader.Define(self.stage, path + "Material/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(self.stage, path + "Material/Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def save(self):
        self.stage.GetRootLayer().Save()


def make_label(name, title, subtitle, color, nvidia=False):
    image = Image.new("RGB", (1536, 384), (15, 20, 25))
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 100)
    small = ImageFont.truetype(font_path, 38)
    draw.rectangle((0, 0, 20, 384), fill=color)
    if nvidia:
        logo = Image.open(ASSETS / "nvidia.png").convert("RGBA")
        # The wordmark is black in the original artwork; use a white field intact.
        logo.thumbnail((610, 115))
        draw.rectangle((55, 24, 695, 153), fill="white")
        image.paste(logo, (70, 30), logo)
        draw.text((55, 166), title, font=font, fill="white")
    else:
        draw.text((55, 45), title, font=font, fill="white")
    draw.text((55, 302), subtitle, font=small, fill=color)
    image.save(ASSETS / f"{name}.png")


def build_card(name, title, accent, color):
    make_label(
        name,
        title,
        "32 GB  /  GDDR7" if name == "rtx_5090" else "FICTIONAL HARDWARE  /  DEMO PROP",
        color,
        name == "rtx_5090",
    )
    m = Model(name)
    m.body("/Asset", 1.4)
    m.box("/Asset/Chassis", (0, 0, 0), (0.304, 0.137, 0.038), "black", True)
    m.box("/Asset/Backplate", (0, 0, -0.020), (0.301, 0.133, 0.002), "steel")
    m.box("/Asset/PCB", (0, 0, -0.016), (0.296, 0.13, 0.002), "pcb")
    for i in range(51):
        m.box(f"/Asset/Fins/Fin{i}", (-0.145 + i * 0.0058, 0, 0.006), (0.0012, 0.122, 0.027), "silver")
    for y in (-0.065, 0.065):
        m.box(f"/Asset/Rail{int(y > 0)}", (0, y, 0.018), (0.304, 0.007, 0.010), "steel")
    m.fan("/Asset/FanLeft", (-0.092, 0, 0.021), 0.052)
    m.fan("/Asset/FanRight", (0.092, 0, 0.021), 0.052)
    m.box("/Asset/Bridge", (0, 0, 0.025), (0.073, 0.137, 0.010), "black")
    m.label(f"/Asset/Brand_{name}", f"{name}.png", (0, 0, 0.0302), 0.126, 0.0315)
    UsdGeom.Xformable(m.stage.GetPrimAtPath(f"/Asset/Brand_{name}")).AddRotateZOp().Set(90)
    m.box("/Asset/Accent", (0, -0.064, 0.031), (0.075, 0.004, 0.003), accent)
    for i in range(20):
        m.box(f"/Asset/Contacts/Pin{i}", (-0.07 + i * 0.0065, -0.069, -0.008), (0.004, 0.003, 0.008), "gold")
    m.box("/Asset/IOBracket", (-0.151, 0, 0), (0.002, 0.134, 0.042), "steel")
    for i in range(4):
        m.box(f"/Asset/Ports/Port{i}", (-0.1525, -0.045 + i * 0.03, 0), (0.001, 0.021, 0.009), "rubber")
    for i, (x, y) in enumerate([(-0.14, -0.057), (-0.14, 0.057), (0.14, -0.057), (0.14, 0.057)]):
        m.cylinder(f"/Asset/Screw{i}", (x, y, 0.024), 0.0025, 0.002, "silver")
    m.save()


def build_pc():
    m = Model("pc_chassis")
    m.body("/Asset/Base", 12)
    UsdPhysics.ArticulationRootAPI.Apply(m.root)
    fixed = UsdPhysics.FixedJoint.Define(m.stage, "/Asset/FixedBase")
    fixed.CreateBody1Rel().SetTargets(["/Asset/Base"])
    m.box("/Asset/Base/Floor", (0, 0, 0.015), (0.37, 0.32, 0.03), "steel", True)
    for i, x in enumerate((-0.181, 0.181)):
        m.box(f"/Asset/Base/End{i}", (x, 0, 0.115), (0.008, 0.32, 0.20), "black", True)
    for i, y in enumerate((-0.156, 0.156)):
        m.box(f"/Asset/Base/Side{i}", (0, y, 0.115), (0.37, 0.008, 0.20), "black", True)
    m.box("/Asset/Base/Motherboard", (0, 0, 0.035), (0.34, 0.28, 0.004), "pcb")
    m.box("/Asset/Base/PSU", (0.10, 0.096, 0.065), (0.13, 0.10, 0.055), "black", True)
    m.fan("/Asset/Base/PSUFan", (0.10, 0.096, 0.095), 0.04)
    m.box("/Asset/Base/CPU", (-0.09, 0.10, 0.045), (0.068, 0.064, 0.009), "silver")
    for i in range(13):
        m.box(f"/Asset/Base/CPUFin{i}", (-0.12 + i * 0.005, 0.10, 0.062), (0.002, 0.065, 0.03), "steel")
    for i in range(4):
        m.box(f"/Asset/Base/RAM{i}", (-0.02 + i * 0.008, 0.10, 0.066), (0.003, 0.083, 0.044), "black")
    # Horizontal riser tray: the GPU is installed face-up into a keyed service carrier.
    m.box("/Asset/Base/Carrier", (0, -0.025, 0.090), (0.32, 0.149, 0.009), "steel", True)
    for i, y in enumerate((-0.0995, 0.0495)):
        m.box(f"/Asset/Base/Guide{i}", (0, y, 0.10), (0.32, 0.004, 0.023), "rubber", True)
    for i, x in enumerate((-0.158, 0.158)):
        m.box(f"/Asset/Base/Stop{i}", (x, -0.025, 0.102), (0.004, 0.145, 0.023), "black", True)
    m.box("/Asset/Base/RiserSocket", (0, -0.085, 0.07), (0.12, 0.015, 0.025), "black")
    for i in range(12):
        m.cylinder(f"/Asset/Base/Capacitor{i}", (-0.15 + i * 0.027, -0.135, 0.044), 0.004, 0.018, "silver")
    lid = m.body("/Asset/Panel", 0.6)
    UsdGeom.Xformable(lid).AddTranslateOp().Set((0, 0.16, 0.22))
    m.box("/Asset/Panel/Frame", (0, -0.16, 0), (0.38, 0.33, 0.008), "black", True)
    m.box("/Asset/Panel/Inset", (0, -0.16, 0.0045), (0.348, 0.294, 0.002), "steel")
    # Graspable raised service handle at the edge opposite the hinge.
    for i, x in enumerate((-0.035, 0.035)):
        m.box(f"/Asset/Panel/HandleFoot{i}", (x, -0.292, 0.015), (0.008, 0.012, 0.025), "steel", True)
    m.box("/Asset/Panel/Handle", (0, -0.292, 0.028), (0.078, 0.012, 0.01), "silver", True)
    for i in range(18):
        m.box(f"/Asset/Panel/Vent{i}", (-0.12 + i * 0.014, -0.09, 0.006), (0.006, 0.075, 0.001), "rubber")
    make_label("pc_label", "ARENA // WORKSTATION", "TOOL-FREE GPU SERVICE BAY", (118, 185, 0))
    m.label("/Asset/Panel/Label", "pc_label.png", (0, -0.20, 0.006), 0.26, 0.06)
    joint = UsdPhysics.RevoluteJoint.Define(m.stage, "/Asset/panel_hinge")
    joint.CreateBody0Rel().SetTargets(["/Asset/Base"])
    joint.CreateBody1Rel().SetTargets(["/Asset/Panel"])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0.16, 0.22))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(-110)
    joint.CreateUpperLimitAttr(0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateStiffnessAttr(0)
    drive.CreateDampingAttr(0.8)
    m.save()


def build_bench():
    m = Model("bench")
    m.box("/Asset/Top", (0.4, 0, 0.7225), (1.50, 1.40, 0.055), "wood", True)
    m.box("/Asset/ESDMat", (0.42, -0.20, 0.751), (1.0, 0.86, 0.002), "mat")
    for i, x in enumerate((-0.23, 1.02)):
        for j, y in enumerate((-0.58, 0.58)):
            m.box(f"/Asset/Leg{i}{j}", (x, y, 0.35), (0.045, 0.045, 0.69), "steel", True)
    m.box("/Asset/BackRail", (0.4, 0.59, 0.23), (1.26, 0.03, 0.04), "steel")
    # Visible anti-static mat grid and brushed tabletop seam lines.
    for i in range(24):
        m.box(f"/Asset/GridX{i}", (-0.055 + i * 0.04, -0.2, 0.7521), (0.00045, 0.84, 0.0001), "steel")
    for i in range(21):
        m.box(f"/Asset/GridY{i}", (0.42, -0.6 + i * 0.04, 0.7521), (0.97, 0.00045, 0.0001), "steel")
    make_label("bench_label", "GPU UPGRADE LAB", "01 OPEN   02 INSTALL RTX 5090   03 CLOSE   04 RECYCLE", (118, 185, 0))
    m.label("/Asset/Instruction", "bench_label.png", (0.45, -0.65, 0.751), 0.80, 0.075)
    m.save()


def build_bin():
    m = Model("recycling_bin")
    m.box("/Asset/Floor", (0, 0, 0.005), (0.426, 0.346, 0.01), "blue", True)
    for i, x in enumerate((-0.209, 0.209)):
        m.box(f"/Asset/End{i}", (x, 0, 0.15), (0.008, 0.346, 0.30), "blue", True)
    for i, y in enumerate((-0.169, 0.169)):
        m.box(f"/Asset/Side{i}", (0, y, 0.15), (0.426, 0.008, 0.30), "blue", True)
    for i, x in enumerate((-0.214, 0.214)):
        m.box(f"/Asset/RimX{i}", (x, 0, 0.299), (0.019, 0.364, 0.018), "black")
    for i, y in enumerate((-0.174, 0.174)):
        m.box(f"/Asset/RimY{i}", (0, y, 0.299), (0.438, 0.018, 0.018), "black")
    make_label("recycle_label", "E-WASTE", "RECYCLE ELECTRONICS", (80, 160, 255))
    # Label lies on a lip, facing the overhead work camera.
    m.box("/Asset/LabelLip", (0, -0.19, 0.293), (0.30, 0.07, 0.008), "black")
    m.label("/Asset/Label", "recycle_label.png", (0, -0.19, 0.298), 0.28, 0.064)
    m.save()


def main():
    """Regenerate all committed models and decal textures."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    build_card("rtx_5090", "GEFORCE RTX 5090", "green", (118, 185, 0))
    build_card("vortex_vx880", "VORTEX VX 880", "blue", (60, 145, 255))
    build_card("cinder_c700", "CINDER C700", "red", (245, 95, 40))
    build_pc()
    build_bench()
    build_bin()


if __name__ == "__main__":
    main()
