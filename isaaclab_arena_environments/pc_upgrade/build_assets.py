# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build detailed PC upgrade props with USD render meshes and simple physics colliders."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ASSETS = Path(__file__).with_name("assets")


def rounded_loop(half_x, half_y, radius, z, center=(0, 0)):
    """Return a counterclockwise rounded rectangle at a fixed height."""
    points = []
    for corner in range(4):
        angle_start = corner * math.pi / 2
        cx = (half_x - radius) * (1 if corner in (0, 3) else -1)
        cy = (half_y - radius) * (1 if corner in (0, 1) else -1)
        for index in range(7):
            angle = angle_start + index * math.pi / 12
            points.append((center[0] + cx + radius * math.cos(angle), center[1] + cy + radius * math.sin(angle), z))
    return points


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
            ("blue_dark", (0.010, 0.045, 0.11), 0.05, 0.38),
            ("fan_blade", (0.048, 0.054, 0.063), 0.35, 0.30),
            ("copper", (0.48, 0.23, 0.085), 0.90, 0.24),
            ("ivory", (0.39, 0.42, 0.39), 0.15, 0.4),
            ("glass", (0.10, 0.14, 0.18), 0.08, 0.10),
            ("wood", (0.30, 0.20, 0.105), 0, 0.6),
            ("mat", (0.035, 0.075, 0.08), 0, 0.88),
        ]:
            mat = UsdShade.Material.Define(self.stage, f"/Asset/Materials/{key}")
            shader = UsdShade.Shader.Define(self.stage, f"{mat.GetPath()}/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            if key == "wood" and name == "bench":
                reader = UsdShade.Shader.Define(self.stage, f"{mat.GetPath()}/UV")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                texture = UsdShade.Shader.Define(self.stage, f"{mat.GetPath()}/Grain")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("oak.png"))
                texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
                texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    texture.ConnectableAPI(), "rgb"
                )
            if key == "glass":
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.22)
                shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.48)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self.materials[key] = mat
        physics_material = UsdShade.Material.Define(self.stage, "/Asset/Materials/Contact")
        physics = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
        physics.CreateStaticFrictionAttr(0.75)
        physics.CreateDynamicFrictionAttr(0.65)
        physics.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(self.root).Bind(physics_material, materialPurpose="physics")

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

    def mesh(self, path, points, faces, material="black"):
        shape = UsdGeom.Mesh.Define(self.stage, path)
        shape.CreatePointsAttr(points)
        shape.CreateFaceVertexCountsAttr([len(face) for face in faces])
        indices = []
        for face in faces:
            indices.extend(face)
        shape.CreateFaceVertexIndicesAttr(indices)
        shape.CreateSubdivisionSchemeAttr("none")
        shape.CreateDoubleSidedAttr(True)
        if material == "wood":
            min_x, max_x = min(p[0] for p in points), max(p[0] for p in points)
            min_y, max_y = min(p[1] for p in points), max(p[1] for p in points)
            coordinates = []
            for x, y, _ in points:
                coordinates.append(((x - min_x) / (max_x - min_x), (y - min_y) / (max_y - min_y)))
            UsdGeom.PrimvarsAPI(shape).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            ).Set(coordinates)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(self.materials[material])
        return shape

    def rounded(self, path, position, size, material="black", radius=0.004, collision=False):
        """Create a bevelled rounded rectangular housing and optional invisible box collider."""
        x, y, z = position
        sx, sy, sz = size
        radius = min(radius, sx / 3, sy / 3, sz / 2)
        bevel = min(radius * 0.4, sz * 0.22)
        points = []
        layers = [(-sz / 2, bevel), (-sz / 2 + bevel, 0), (sz / 2 - bevel, 0), (sz / 2, bevel)]
        for height, inset in layers:
            points.extend(rounded_loop(sx / 2 - inset, sy / 2 - inset, radius, z + height, (x, y)))
        count = len(points) // 4
        faces = [list(reversed(range(count))), list(range(3 * count, 4 * count))]
        for layer in range(3):
            for index in range(count):
                next_index = (index + 1) % count
                faces.append([
                    layer * count + index,
                    layer * count + next_index,
                    (layer + 1) * count + next_index,
                    (layer + 1) * count + index,
                ])
        shape = self.mesh(path, points, faces, material)
        if collision:
            collider = self.box(path + "Collision", position, size, material, True)
            collider.CreateVisibilityAttr("invisible")
        return shape

    def ring(self, path, position, outer, inner, depth, material="steel", segments=48):
        x, y, z = position
        points = []
        for height, radius in [(-depth / 2, outer), (depth / 2, outer), (-depth / 2, inner), (depth / 2, inner)]:
            for index in range(segments):
                angle = index * math.tau / segments
                points.append((x + radius * math.cos(angle), y + radius * math.sin(angle), z + height))
        faces = []
        for index in range(segments):
            j = (index + 1) % segments
            faces.extend([
                [index, j, segments + j, segments + index],
                [segments + index, segments + j, 3 * segments + j, 3 * segments + index],
                [2 * segments + j, 2 * segments + index, 3 * segments + index, 3 * segments + j],
                [j, index, 2 * segments + index, 2 * segments + j],
            ])
        return self.mesh(path, points, faces, material)

    def group(self, path, position=(0, 0, 0), rotation=(0, 0, 0)):
        group = UsdGeom.Xform.Define(self.stage, path)
        group.AddTranslateOp().Set(Gf.Vec3d(*position))
        group.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        return group

    def fan(self, path, position, radius, accent="steel", blade_count=9):
        x, y, z = position
        self.cylinder(f"{path}/Well", (x, y, z), radius, 0.003, "rubber")
        self.ring(f"{path}/Rim", (x, y, z + 0.003), radius, radius * 0.92, 0.006, accent)
        for index in range(blade_count):
            angle = index * math.tau / blade_count
            points = []
            for r, offset in [(0.21, -0.2), (0.60, -0.35), (0.90, -0.12), (0.88, 0.19), (0.43, 0.29)]:
                offset *= min(1.0, 9.0 / blade_count)
                points.append((
                    x + math.cos(angle + offset) * radius * r,
                    y + math.sin(angle + offset) * radius * r,
                    z + 0.005 + 0.002 * r,
                ))
            self.mesh(f"{path}/Blade{index}", points, [list(range(5))], "fan_blade")
        self.cylinder(f"{path}/Hub", (x, y, z + 0.006), radius * 0.23, 0.007, "black")
        self.cylinder(f"{path}/HubCap", (x, y, z + 0.010), radius * 0.17, 0.001, accent)

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
        layer = self.stage.GetRootLayer()
        layer.Save()
        path = Path(layer.realPath)
        path.write_text(path.read_text().rstrip() + "\n")


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
    """Build a recognizable consumer GPU with a thin exposed PCIe edge connector."""
    make_label(
        name,
        title,
        "32 GB  /  GDDR7" if name == "rtx_5090" else "FICTIONAL HARDWARE  /  DEMO PROP",
        color,
        name == "rtx_5090",
    )
    m = Model(name)
    m.body("/Asset", 1.4)
    # Keep the contact tongue thin so it enters the split motherboard socket.
    body = m.box("/Asset/BodyCollision", (0, 0.00575, 0.0065), (0.304, 0.1255, 0.055), collision=True)
    body.CreateVisibilityAttr("invisible")
    connector = m.box("/Asset/ConnectorCollision", (-0.035, -0.0645, 0), (0.140, 0.013, 0.003), collision=True)
    connector.CreateVisibilityAttr("invisible")
    m.rounded("/Asset/Backplate", (0, 0.004, -0.0195), (0.301, 0.129, 0.003), "steel", 0.009)
    m.rounded("/Asset/PCB", (0, 0.003, 0), (0.294, 0.129, 0.002), "pcb", 0.004)
    m.box("/Asset/ContactTongue", (-0.035, -0.066, 0), (0.140, 0.010, 0.002), "pcb")
    for index in range(36):
        if index in (8, 9):
            continue
        x = -0.101 + index * 0.0038
        for side in (-1, 1):
            m.box(
                f"/Asset/Contacts/Pin{index}_{side + 1}", (x, -0.067, side * 0.00115), (0.0024, 0.007, 0.0003), "gold"
            )
    for index in range(62):
        m.box(f"/Asset/Heatsink/Fin{index}", (-0.144 + index * 0.0047, 0.004, 0.010), (0.0008, 0.117, 0.017), "silver")
    for y in (-0.049, 0.045):
        m.cylinder(f"/Asset/Heatpipe{int(y > 0)}", (0, y, 0.006), 0.003, 0.285, "copper").CreateAxisAttr("X")
    for i, y in enumerate((-0.051, 0.064)):
        m.rounded(
            f"/Asset/Shroud/Rail{i}",
            (0, y, 0.025),
            (0.300, 0.009, 0.016),
            "silver" if name == "rtx_5090" else "black",
            0.004,
        )
    if name == "vortex_vx880":
        # Three compact axial fans and bright blue angular trim.
        for index, x in enumerate((-0.097, 0, 0.097)):
            m.fan(f"/Asset/Fan{index}", (x, 0.005, 0.020), 0.045, "blue")
        for index, x in enumerate((-0.05, 0.05)):
            m.rounded(f"/Asset/Shroud/Brace{index}", (x, 0.006, 0.027), (0.008, 0.111, 0.012), "blue")
        m.label(f"/Asset/TopBrand_{name}", f"{name}.png", (0, 0.053, 0.0331), 0.220, 0.014)
    elif name == "cinder_c700":
        # A ducted blower cooler gives this fictional card a different silhouette.
        m.rounded("/Asset/Shroud/Duct", (-0.042, 0.004, 0.025), (0.204, 0.120, 0.016), "black", 0.008)
        m.fan("/Asset/Blower", (0.097, 0.004, 0.020), 0.044, "red", blade_count=24)
        m.label(f"/Asset/TopBrand_{name}", f"{name}.png", (-0.046, 0.006, 0.0331), 0.174, 0.0435)
        for index in range(6):
            m.box(f"/Asset/Shroud/Stripe{index}", (-0.120 + index * 0.029, -0.049, 0.033), (0.014, 0.012, 0.001), "red")
    else:
        for index, x in enumerate((-0.094, 0.094)):
            m.fan(f"/Asset/Fan{index}", (x, 0.004, 0.020), 0.051, "silver")
        m.rounded("/Asset/Shroud/Bridge", (0, 0.003, 0.025), (0.072, 0.122, 0.016), "black", 0.005)
        m.group(f"/Asset/TopBrand_{name}", (0, 0.004, 0.0331), (0, 0, 90))
        m.label(f"/Asset/TopBrand_{name}/Decal", f"{name}.png", (0, 0, 0), 0.117, 0.029)
    # Brand on the exposed long edge and backplate remains readable after installation.
    m.group(f"/Asset/EdgeBrand_{name}", (0, 0.0686, 0.014), (90, 0, 0))
    m.label(f"/Asset/EdgeBrand_{name}/Decal", f"{name}.png", (0, 0, 0), 0.190, 0.017)
    m.group(f"/Asset/BackBrand_{name}", (0, 0.006, -0.0211), (180, 0, 0))
    m.label(f"/Asset/BackBrand_{name}/Decal", f"{name}.png", (0, 0, 0), 0.185, 0.046)
    for i, x in enumerate((-0.131, 0.131)):
        for j, y in enumerate((-0.047, 0.052)):
            m.cylinder(f"/Asset/Screws/Screw{i}{j}", (x, y, 0.032), 0.0023, 0.001, "silver")
            m.box(f"/Asset/Screws/Slot{i}{j}", (x, y, 0.0326), (0.003, 0.0005, 0.0002), "rubber")
    m.rounded("/Asset/IOBracket", (-0.151, 0.003, 0.005), (0.002, 0.131, 0.052), "steel", 0.001)
    for index in range(4):
        m.rounded(
            f"/Asset/Ports/Bezel{index}",
            (-0.1523, -0.043 + index * 0.030, 0.005),
            (0.001, 0.023, 0.010),
            "silver",
            0.0004,
        )
        m.box(f"/Asset/Ports/Socket{index}", (-0.1529, -0.043 + index * 0.030, 0.005), (0.0003, 0.018, 0.006), "rubber")
    m.rounded("/Asset/PowerSocket", (0.037, 0.065, -0.006), (0.018, 0.007, 0.010), "rubber", 0.002)
    m.save()


def build_pc():
    """Build an upright ATX tower with an outward-opening hinged glass side panel."""
    m = Model("pc_chassis")
    m.body("/Asset/Base", 12)
    UsdPhysics.ArticulationRootAPI.Apply(m.root)
    fixed = UsdPhysics.FixedJoint.Define(m.stage, "/Asset/FixedBase")
    fixed.CreateBody1Rel().SetTargets(["/Asset/Base"])
    base = "/Asset/Base"
    for index, x in enumerate((-0.176, 0.176)):
        for j, y in enumerate((-0.084, 0.084)):
            m.rounded(f"{base}/Foot{index}{j}", (x, y, 0.009), (0.055, 0.048, 0.018), "rubber", collision=True)
    m.rounded(f"{base}/Floor", (0, 0, 0.024), (0.438, 0.256, 0.014), "black", collision=True)
    m.rounded(f"{base}/Roof", (0, 0, 0.453), (0.440, 0.260, 0.014), "black", collision=True)
    m.rounded(f"{base}/Rear", (-0.217, 0, 0.240), (0.006, 0.252, 0.418), "black", collision=True)
    m.rounded(f"{base}/Front", (0.214, 0, 0.240), (0.012, 0.254, 0.418), "black", collision=True)
    m.rounded(f"{base}/CableSide", (0, 0.127, 0.240), (0.428, 0.006, 0.418), "black", collision=True)
    m.rounded(f"{base}/PSUShroud", (-0.005, 0, 0.065), (0.418, 0.240, 0.065), "black", collision=True)
    # Opening facing -Y has narrow structural rails, leaving the entire insertion path clear.
    for index, z in enumerate((0.036, 0.440)):
        m.rounded(f"{base}/OpeningRail{index}", (0, -0.125, z), (0.428, 0.010, 0.012), "steel", collision=True)
    m.rounded(f"{base}/Motherboard", (-0.026, 0.120, 0.275), (0.340, 0.006, 0.285), "pcb", collision=True)
    # Board details are authored in XY and rotated to face the open side (-Y).
    board = f"{base}/BoardDetails"
    m.group(board, (0, 0.115, 0), (90, 0, 0))
    for index in range(6):
        x = 0.098 + (index % 2) * 0.020
        z = 0.238 + (index // 2) * 0.043
        m.rounded(f"{board}/Controller{index}", (x, z, 0.002), (0.012, 0.020, 0.004), "black", 0.001)
        for pin in range(5):
            m.box(f"{board}/Solder{index}_{pin}", (x, z - 0.008 + pin * 0.004, 0.001), (0.017, 0.001, 0.001), "silver")
            m.box(
                f"{board}/Trace{index}_{pin}",
                (x - 0.017, z - 0.008 + pin * 0.004, 0.0003),
                (0.018, 0.0003, 0.0003),
                "copper",
            )
    m.rounded(f"{board}/CPUSocket", (-0.060, 0.335, 0.005), (0.082, 0.082, 0.010), "silver")
    for index in range(20):
        m.box(f"{board}/CPUFin{index}", (-0.108 + index * 0.005, 0.335, 0.022), (0.0015, 0.096, 0.034), "silver")
    m.fan(f"{board}/CPUFan", (-0.060, 0.335, 0.044), 0.046)
    for index in range(4):
        x = 0.014 + index * 0.018
        m.rounded(f"{board}/RAMSocket{index}", (x, 0.333, 0.004), (0.009, 0.127, 0.008), "rubber")
        m.rounded(f"{board}/RAM{index}", (x, 0.333, 0.015), (0.005, 0.116, 0.026), "steel")
        m.box(f"{board}/RAMStripe{index}", (x, 0.333, 0.029), (0.005, 0.102, 0.003), "green")
    m.rounded(f"{board}/Chipset", (0.080, 0.156, 0.008), (0.062, 0.045, 0.014), "steel")
    for index in range(8):
        m.box(f"{board}/ChipsetRib{index}", (0.055 + index * 0.007, 0.156, 0.017), (0.002, 0.040, 0.005), "silver")
    for index in range(14):
        x = -0.170 + (index % 7) * 0.020
        z = 0.260 + (index // 7) * 0.146
        m.cylinder(f"{board}/Capacitor{index}", (x, z, 0.006), 0.004, 0.012, "silver")
        m.box(f"{board}/CapCross{index}", (x, z, 0.0122), (0.006, 0.0007, 0.0003), "rubber")
    for index in range(4):
        m.rounded(f"{board}/VRM{index}", (-0.157, 0.305 + index * 0.025, 0.009), (0.036, 0.020, 0.016), "black")
    for index, z in enumerate((0.144, 0.169)):
        m.rounded(f"{base}/SparePCIe{index}", (-0.090, 0.108, z), (0.143, 0.016, 0.008), "black")
    # A split socket with a 7 mm collision channel gives the 3 mm tongue a lead-in clearance.
    for index, z in enumerate((0.1985, 0.2115)):
        m.rounded(f"{base}/PCIeSocket/Jaw{index}", (-0.095, 0.110, z), (0.151, 0.014, 0.006), "ivory", 0.001, True)
    m.box(f"{base}/PCIeSocket/Back", (-0.095, 0.118, 0.205), (0.151, 0.002, 0.017), "ivory")
    m.rounded(f"{base}/PCIeSocket/Latch", (-0.016, 0.110, 0.205), (0.007, 0.014, 0.017), "black", 0.001)
    for index in range(36):
        m.box(
            f"{base}/PCIeSocket/Contact{index}",
            (-0.162 + index * 0.0038, 0.112, 0.2016),
            (0.0014, 0.007, 0.0003),
            "gold",
        )
    # Rear bracket ledge and a narrow front anti-sag arm support the fan-side edges.
    m.rounded(f"{base}/RearBracketSupport", (-0.204, 0.039, 0.163), (0.012, 0.148, 0.012), "steel", 0.002, True)
    m.rounded(f"{base}/AntiSag/Post", (0.092, 0.072, 0.132), (0.012, 0.022, 0.069), "steel", collision=True)
    m.rounded(f"{base}/AntiSag/Arm", (0.087, 0.036, 0.163), (0.022, 0.127, 0.012), "rubber", 0.002, True)
    # Three front intake fans, grille slots, rear expansion-slot vents, and top ventilation.
    m.group(f"{base}/Intake", (0.203, 0, 0), (0, -90, 0))
    for index, z in enumerate((0.149, 0.263, 0.377)):
        m.fan(f"{base}/Intake/Fan{index}", (z, 0, 0), 0.048, "steel")
    for index in range(23):
        m.box(f"{base}/FrontMesh/Slit{index}", (0.2202, 0, 0.113 + index * 0.014), (0.0005, 0.194, 0.005), "rubber")
    for index in range(20):
        m.box(f"{base}/RoofVent{index}", (-0.160 + index * 0.016, 0.015, 0.4602), (0.006, 0.151, 0.0004), "rubber")
    for index in range(8):
        m.box(f"{base}/ExpansionVent{index}", (-0.2202, 0.045, 0.117 + index * 0.013), (0.0005, 0.136, 0.005), "rubber")
    m.rounded(f"{base}/RearIO", (-0.2205, 0.058, 0.340), (0.001, 0.104, 0.154), "steel", 0.0004)
    for index in range(6):
        m.box(f"{base}/RearPort{index}", (-0.2212, 0.058, 0.285 + index * 0.021), (0.0005, 0.024, 0.012), "rubber")
    m.cylinder(f"{base}/PowerButton", (0.183, -0.070, 0.461), 0.007, 0.0015, "silver")
    m.cylinder(f"{base}/PowerLED", (0.183, -0.088, 0.461), 0.002, 0.0015, "green")
    panel = m.body("/Asset/Panel", 1.1)
    UsdGeom.Xformable(panel).AddTranslateOp().Set((-0.223, -0.132, 0.240))
    # Hollow perimeter frame with inset smoky glass; hinge coordinate is the rear vertical edge.
    for index, x in enumerate((0.007, 0.439)):
        m.rounded(f"/Asset/Panel/Upright{index}", (x, 0, 0), (0.014, 0.008, 0.440), "black", collision=True)
    for index, z in enumerate((-0.213, 0.213)):
        m.rounded(f"/Asset/Panel/Crossbar{index}", (0.223, 0, z), (0.418, 0.008, 0.014), "black", collision=True)
    m.rounded("/Asset/Panel/Glass", (0.223, 0, 0), (0.418, 0.004, 0.412), "glass", 0.002, True)
    for index, z in enumerate((-0.150, 0.150)):
        m.cylinder(f"{base}/HingeBarrel{index}", (-0.223, -0.132, 0.240 + z), 0.008, 0.044, "steel")
        m.cylinder(f"/Asset/Panel/Hinge{index}", (0, 0, z), 0.009, 0.020, "silver")
    m.rounded(f"{base}/MagneticCatch", (0.208, -0.123, 0.337), (0.015, 0.009, 0.026), "rubber", 0.002)
    m.rounded("/Asset/Panel/CatchPlate", (0.432, 0.005, 0.097), (0.011, 0.002, 0.025), "silver", 0.001)
    for index, z in enumerate((-0.040, 0.040)):
        m.rounded(f"/Asset/Panel/HandleFoot{index}", (0.408, -0.016, z), (0.014, 0.028, 0.012), "steel", collision=True)
    m.rounded("/Asset/Panel/Handle", (0.408, -0.030, 0), (0.014, 0.012, 0.092), "silver", collision=True)
    make_label("pc_label", "ARENA // WORKSTATION", "RTX 5090  |  PERFORMANCE SYSTEM", (118, 185, 0))
    m.group("/Asset/Panel/Brand", (0.223, -0.0041, -0.171), (90, 0, 0))
    m.label("/Asset/Panel/Brand/Decal", "pc_label.png", (0, 0, 0), 0.266, 0.033)
    joint = UsdPhysics.RevoluteJoint.Define(m.stage, "/Asset/panel_hinge")
    joint.CreateBody0Rel().SetTargets(["/Asset/Base"])
    joint.CreateBody1Rel().SetTargets(["/Asset/Panel"])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(-0.223, -0.132, 0.240))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLowerLimitAttr(-110)
    joint.CreateUpperLimitAttr(0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateStiffnessAttr(0)
    drive.CreateDampingAttr(0.8)
    m.save()


def build_bench():
    # Long, low-contrast grain follows the tabletop X axis; fixed arithmetic is deterministic.
    image = Image.new("RGB", (1024, 512), (158, 121, 82))
    draw = ImageDraw.Draw(image)
    for row in range(0, 512, 2):
        tone = int(6 * math.sin(row * 0.73) + 4 * math.cos(row * 0.19))
        points = []
        for x in range(0, 1025, 8):
            y = row + 2.4 * math.sin(x * 0.008 + row * 0.17)
            points.append((x, y))
        draw.line(points, fill=(154 + tone, 117 + tone, 79 + tone), width=1)
    image.save(ASSETS / "oak.png", optimize=True)
    m = Model("bench")
    m.rounded("/Asset/Top", (0.4, 0, 0.7225), (1.50, 1.40, 0.055), "wood", 0.012, True)
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
    m.label("/Asset/Instruction", "bench_label.png", (0.45, -0.668, 0.751), 0.90, 0.057)
    m.save()


def build_bin():
    """Build a hollow moulded recycling tub with a rolled rim and visible recycling marks."""
    m = Model("recycling_bin")
    # These five invisible primitives preserve the rectangular usable interior exactly.
    floor = m.box("/Asset/FloorCollision", (0, 0, 0.005), (0.426, 0.346, 0.010), "blue", True)
    floor.CreateVisibilityAttr("invisible")
    for i, x in enumerate((-0.209, 0.209)):
        wall = m.box(f"/Asset/EndCollision{i}", (x, 0, 0.155), (0.008, 0.330, 0.290), "blue", True)
        wall.CreateVisibilityAttr("invisible")
    for i, y in enumerate((-0.169, 0.169)):
        wall = m.box(f"/Asset/SideCollision{i}", (0, y, 0.155), (0.426, 0.008, 0.290), "blue", True)
        wall.CreateVisibilityAttr("invisible")
    m.rounded("/Asset/Bottom", (0, 0, 0.005), (0.460, 0.380, 0.010), "blue", 0.004)
    # The rounded interior encloses the required clear rectangle; taper grows outward.
    outer_bottom = rounded_loop(0.234, 0.194, 0.025, 0.005)
    outer_top = rounded_loop(0.250, 0.210, 0.030, 0.290)
    inner_top = rounded_loop(0.239, 0.199, 0.024, 0.290)
    inner_bottom = rounded_loop(0.221, 0.181, 0.016, 0.010)
    count = len(outer_bottom)
    points = outer_bottom + outer_top + inner_top + inner_bottom
    faces = []
    for ring in range(3):
        for index in range(count):
            j = (index + 1) % count
            faces.append([ring * count + index, ring * count + j, (ring + 1) * count + j, (ring + 1) * count + index])
    m.mesh("/Asset/MouldedShell", points, faces, "blue")
    # Rolled rectangular rim made from rounded loops rather than four square bars.
    rim_points = []
    for hx, hy, radius, z in [
        (0.251, 0.211, 0.031, 0.281),
        (0.255, 0.215, 0.032, 0.288),
        (0.255, 0.215, 0.032, 0.298),
        (0.247, 0.207, 0.028, 0.302),
        (0.235, 0.195, 0.023, 0.298),
        (0.235, 0.195, 0.023, 0.288),
    ]:
        rim_points.extend(rounded_loop(hx, hy, radius, z))
    faces = []
    for ring in range(6):
        next_ring = (ring + 1) % 6
        for index in range(count):
            j = (index + 1) % count
            faces.append([ring * count + index, ring * count + j, next_ring * count + j, next_ring * count + index])
    m.mesh("/Asset/RolledRim", rim_points, faces, "blue_dark")
    for index, x in enumerate((-0.254, 0.254)):
        m.rounded(f"/Asset/HandleBacking{index}", (x, 0, 0.253), (0.007, 0.114, 0.034), "blue_dark", 0.003)
        m.rounded(f"/Asset/HandleRecess{index}", (x * 1.015, 0, 0.253), (0.002, 0.077, 0.015), "rubber", 0.001)
    for index in range(5):
        for side in (-1, 1):
            m.rounded(
                f"/Asset/BodyRib{index}_{side + 1}",
                (-0.160 + index * 0.080, side * 0.196, 0.105),
                (0.008, 0.009, 0.165),
                "blue_dark",
                0.003,
            )
    make_recycling_label()
    m.group("/Asset/FrontMark", (0, -0.207, 0.168), (90, 0, 0))
    m.label("/Asset/FrontMark/Decal", "recycle_label.png", (0, 0, 0), 0.240, 0.132)
    m.rounded("/Asset/LabelLip", (0, -0.214, 0.287), (0.295, 0.065, 0.012), "blue_dark", 0.005)
    m.label("/Asset/RimMark", "recycle_label.png", (0, -0.214, 0.2932), 0.185, 0.052)
    m.save()


def make_recycling_label():
    """Draw a high-contrast three-arrow recycling emblem and electronics label."""
    image = Image.new("RGB", (1024, 560), (14, 44, 86))
    draw = ImageDraw.Draw(image)
    color = (235, 247, 255)
    cx, cy = 510, 173
    # Three bent arrows following a triangular loop, with deliberately large heads.
    arrow = [(-29, -114), (20, -114), (69, -31), (95, -46), (78, 18), (14, 3), (40, -12), (4, -75), (-29, -75)]
    for index in range(3):
        angle = index * math.tau / 3
        polygon = []
        for x, y in arrow:
            polygon.append(
                (cx + x * math.cos(angle) - y * math.sin(angle), cy + x * math.sin(angle) + y * math.cos(angle))
            )
        draw.polygon(polygon, fill=color)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    draw.text((512, 330), "E-WASTE", font=ImageFont.truetype(font_path, 110), anchor="mm", fill=color)
    draw.text(
        (512, 445), "RECYCLE ELECTRONICS", font=ImageFont.truetype(font_path, 49), anchor="mm", fill=(134, 191, 242)
    )
    image.save(ASSETS / "recycle_label.png")


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
