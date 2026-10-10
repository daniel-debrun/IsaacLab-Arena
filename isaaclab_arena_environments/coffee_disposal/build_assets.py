# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the original reference-inspired coffee-disposal props and textures."""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_arena_environments.coffee_disposal.layout import (
    BIN_FLOOR_THICKNESS,
    BIN_HALF_INTERIOR,
    BIN_HEIGHT,
    BIN_IN_DRAWER,
    CUP_SIZE,
    GROUND_COLLISION_FACES,
    GROUND_COLLISION_VERTICES,
    GROUND_COLUMNS,
    GROUND_RADIUS,
    GROUND_ROWS,
    GROUND_VERTICES,
    LID_SIZE,
    TRAY_IN_MACHINE,
)
from isaaclab_arena_environments.pc_upgrade.build_assets import Model as PrimitiveModel
from isaaclab_arena_environments.pc_upgrade.build_assets import rounded_loop

ASSETS = Path(__file__).with_name("assets")


class Model(PrimitiveModel):
    """Reuse geometry helpers with an independent output directory and material palette."""

    def __init__(self, name):
        self.stage = Usd.Stage.CreateNew(str(ASSETS / f"{name}.usda"))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        self.root = UsdGeom.Xform.Define(self.stage, "/Asset").GetPrim()
        self.stage.SetDefaultPrim(self.root)
        self.materials = {}
        palette = [
            ("black", (0.025, 0.028, 0.032), 0.15, 0.30),
            ("rubber", (0.012, 0.014, 0.016), 0, 0.84),
            ("silver", (0.62, 0.66, 0.69), 0.90, 0.25),
            ("steel", (0.28, 0.31, 0.33), 0.83, 0.31),
            ("white", (0.83, 0.82, 0.78), 0, 0.27),
            ("stone", (0.85, 0.83, 0.78), 0, 0.22),
            ("bin", (0.063, 0.069, 0.068), 0.02, 0.58),
            ("lid", (0.19, 0.205, 0.21), 0.12, 0.45),
            ("blue", (0.025, 0.29, 0.49), 0, 0.33),
            ("coffee", (0.062, 0.025, 0.008), 0, 0.96),
            ("coffee_light", (0.15, 0.066, 0.022), 0, 0.96),
            ("coffee_dark", (0.027, 0.012, 0.006), 0, 1),
            ("leaf", (0.18, 0.23, 0.052), 0, 0.83),
            ("glass", (0.24, 0.32, 0.34), 0.12, 0.12),
            ("water", (0.13, 0.26, 0.30), 0.08, 0.13),
            ("wood", (0.36, 0.22, 0.12), 0, 0.55),
            ("sky", (0.28, 0.45, 0.58), 0, 0.80),
            ("wall", (0.57, 0.555, 0.52), 0, 0.90),
            ("sponge", (0.59, 0.44, 0.10), 0, 0.93),
        ]
        for key, color, metallic, roughness in palette:
            material = UsdShade.Material.Define(self.stage, f"/Asset/Materials/{key}")
            shader = UsdShade.Shader.Define(self.stage, f"{material.GetPath()}/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            if key in ("glass", "water"):
                shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.36)
                shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.45)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self.materials[key] = material
        material = UsdShade.Material.Define(self.stage, "/Asset/Materials/Contact")
        contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        contact.CreateStaticFrictionAttr(0.85)
        contact.CreateDynamicFrictionAttr(0.70)
        contact.CreateRestitutionAttr(0)
        UsdShade.MaterialBindingAPI.Apply(self.root).Bind(material, materialPurpose="physics")

    def fixed_base(self, mass):
        """Create a fixed-base articulation with its base rigid body at the asset origin."""
        self.body("/Asset/Base", mass)
        UsdPhysics.ArticulationRootAPI.Apply(self.root)
        fixed = UsdPhysics.FixedJoint.Define(self.stage, "/Asset/FixedBase")
        fixed.CreateBody1Rel().SetTargets(["/Asset/Base"])

    def slider(self, body, name, position, axis, lower, upper, mass):
        """Create a passive prismatic body whose zero position matches its authored transform."""
        prim = self.body(f"/Asset/{body}", mass)
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*position))
        joint = UsdPhysics.PrismaticJoint.Define(self.stage, f"/Asset/{name}")
        joint.CreateBody0Rel().SetTargets(["/Asset/Base"])
        joint.CreateBody1Rel().SetTargets([f"/Asset/{body}"])
        joint.CreateAxisAttr(axis)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*position))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLowerLimitAttr(lower)
        joint.CreateUpperLimitAttr(upper)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateStiffnessAttr(0)
        drive.CreateDampingAttr(2)

    def sphere(self, path, position, scale, material):
        """Create a decorative ellipsoid without collision."""
        shape = UsdGeom.Sphere.Define(self.stage, path)
        shape.CreateRadiusAttr(1)
        shape.AddTranslateOp().Set(Gf.Vec3d(*position))
        shape.AddScaleOp().Set(Gf.Vec3d(*scale))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(self.materials[material])
        return shape

    def rim(self, path, position, outer, inner, height, material):
        """Create a rounded rectangular open rim, with four separate contact strips."""
        x, y, z = position
        points = []
        for hx, hy, h in [(*outer, z), (*outer, z + height), (*inner, z + height), (*inner, z)]:
            points.extend(rounded_loop(hx, hy, min(0.008, hx / 5), h, (x, y)))
        count = len(points) // 4
        faces = []
        for ring in range(4):
            for index in range(count):
                following = (index + 1) % count
                next_ring = (ring + 1) % 4
                faces.append([
                    ring * count + index,
                    ring * count + following,
                    next_ring * count + following,
                    next_ring * count + index,
                ])
        self.mesh(path, points, faces, material)
        for i, sign in enumerate((-1, 1)):
            self.box(
                f"{path}ColliderX{i}",
                (x + sign * (outer[0] + inner[0]) / 2, y, z + height / 2),
                (outer[0] - inner[0], 2 * inner[1], height),
                material,
                True,
            ).CreateVisibilityAttr("invisible")
            self.box(
                f"{path}ColliderY{i}",
                (x, y + sign * (outer[1] + inner[1]) / 2, z + height / 2),
                (2 * outer[0], outer[1] - inner[1], height),
                material,
                True,
            ).CreateVisibilityAttr("invisible")

    def tub(self, path, position, interior, height, wall, material="bin", floor=BIN_FLOOR_THICKNESS):
        """Build a hollow receptacle with a flat floor and disjoint wall colliders."""
        x, y, z = position
        hx, hy = interior
        self.rounded(
            f"{path}/Floor", (x, y, z + floor / 2), (2 * (hx + wall), 2 * (hy + wall), floor), material, 0.004, True
        )
        for i, sign in enumerate((-1, 1)):
            self.rounded(
                f"{path}/WallX{i}",
                (x + sign * (hx + wall / 2), y, z + (height - 0.004 + floor) / 2),
                (wall, 2 * hy, height - 0.004 - floor),
                material,
                0.003,
                True,
            )
            self.rounded(
                f"{path}/WallY{i}",
                (x, y + sign * (hy + wall / 2), z + (height - 0.004 + floor) / 2),
                (2 * (hx + wall), wall, height - 0.004 - floor),
                material,
                0.003,
                True,
            )
        self.rim(
            f"{path}/RolledLip",
            (x, y, z + height - 0.004),
            (hx + wall, hy + wall),
            (hx, hy),
            0.004,
            material,
        )


def textures():
    """Draw original display, product badge, sorting labels and subtle oak grain."""
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    badge = Image.new("RGB", (768, 200), (24, 27, 30))
    draw = ImageDraw.Draw(badge)
    draw.text((384, 71), "A R O M A", font=ImageFont.truetype(bold, 64), anchor="mm", fill=(210, 216, 220))
    draw.text((384, 151), "AUTOMATICA", font=ImageFont.truetype(font, 26), anchor="mm", fill=(155, 167, 175))
    badge.save(ASSETS / "aroma.png")
    screen = Image.new("RGB", (600, 800), (8, 20, 26))
    draw = ImageDraw.Draw(screen)
    draw.rounded_rectangle((8, 8, 592, 792), 34, outline=(41, 85, 100), width=5)
    draw.rounded_rectangle((25, 35, 575, 109), 10, fill=(43, 139, 164))
    draw.text((300, 73), "READY", font=ImageFont.truetype(bold, 30), anchor="mm", fill=(219, 250, 252))
    for i, title in enumerate(("ESPRESSO", "COFFEE", "LONG", "CAPPUCCINO")):
        cx, cy = 158 + (i % 2) * 285, 245 + (i // 2) * 190
        draw.rounded_rectangle((cx - 30, cy - 42, cx + 23, cy + 10), 8, outline=(175, 242, 249), width=5)
        draw.arc((cx + 14, cy - 33, cx + 42, cy), -100, 100, fill=(175, 242, 249), width=5)
        draw.line((cx - 35, cy + 17, cx + 37, cy + 17), fill=(175, 242, 249), width=4)
        draw.text((cx, cy + 58), title, font=ImageFont.truetype(font, 20), anchor="mm", fill=(194, 236, 241))
    for i in range(6):
        cx, cy = 150 + (i % 2) * 300, 622 + (i // 2) * 54
        draw.rounded_rectangle((cx - 59, cy - 17, cx + 59, cy + 17), 16, outline=(74, 117, 130), width=2)
        draw.text(
            (cx, cy),
            ("+", "−", "○", "◉", "MENU", "OK")[i],
            font=ImageFont.truetype(font, 20),
            anchor="mm",
            fill=(140, 177, 186),
        )
    screen.save(ASSETS / "display.png")
    for filename, title, color in [("organic", "ORGANIC", (139, 160, 94)), ("recycling", "RECYCLING", (111, 160, 181))]:
        label = Image.new("RGB", (512, 120), (40, 46, 46))
        draw = ImageDraw.Draw(label)
        draw.text((256, 60), title, font=ImageFont.truetype(bold, 48), anchor="mm", fill=color)
        label.save(ASSETS / f"{filename}.png")
    oak = Image.new("RGB", (1024, 256), (150, 110, 70))
    draw = ImageDraw.Draw(oak)
    for row in range(256):
        tone = int(7 * math.sin(row * 0.48) + 4 * math.sin(row * 1.88))
        points = []
        for x in range(0, 1025, 8):
            points.append((x, row + 2 * math.sin(x * 0.011 + row * 0.05)))
        draw.line(points, fill=(150 + tone, 110 + tone, 70 + tone), width=1)
    oak.save(ASSETS / "oak.png", optimize=True)


def build_machine():
    """Build the fixed machine, height-adjustable twin dispenser and complete sliding tray."""
    m = Model("coffee_machine")
    m.fixed_base(9)
    b = "/Asset/Base"
    m.rounded(f"{b}/ChassisFloor", (0, 0.04, 0.006), (0.310, 0.350, 0.012), "black", 0.004, True)
    m.rounded(f"{b}/UpperBody", (0, 0.025, 0.315), (0.292, 0.370, 0.290), "black", 0.010, True)
    for i, x in enumerate((-0.151, 0.151)):
        side = m.rounded(f"{b}/SilverSide{i}", (x, 0.025, 0.234), (0.018, 0.390, 0.444), "silver", 0.008, True)
        if i == 1:
            side.CreateVisibilityAttr("invisible")
            # Retain the original simple side collider, opening only the detailed tank window.
            for suffix, center, size in [
                ("Upper", (x, 0.025, 0.388), (0.018, 0.390, 0.136)),
                ("Lower", (x, 0.025, 0.036), (0.018, 0.390, 0.048)),
                ("Front", (x, -0.144, 0.190), (0.018, 0.052, 0.260)),
                ("Rear", (x, 0.198, 0.190), (0.018, 0.044, 0.260)),
            ]:
                m.rounded(f"{b}/TankFrame{suffix}", center, size, "silver", 0.006)
        for vent in range(25):
            m.box(f"{b}/Vent{i}_{vent}", (x * 1.063, -0.092 + vent * 0.010, 0.391), (0.0006, 0.003, 0.042), "rubber")
    m.rounded(f"{b}/BackWall", (0, 0.211, 0.226), (0.294, 0.018, 0.440), "black", collision=True)
    m.rounded(f"{b}/FrontFascia", (0, -0.165, 0.298), (0.279, 0.016, 0.250), "steel", 0.004, True)
    m.rounded(f"{b}/ControlBezel", (-0.020, -0.184, 0.353), (0.135, 0.030, 0.224), "silver", 0.009)
    m.rounded(f"{b}/ControlGlass", (-0.020, -0.201, 0.353), (0.121, 0.006, 0.211), "black", 0.004)
    m.group(f"{b}/Screen", (-0.020, -0.2041, 0.353), (90, 0, 0))
    m.label(f"{b}/Screen/Decal", "display.png", (0, 0, 0), 0.110, 0.197)
    shader = UsdShade.Shader.Get(m.stage, f"{b}/Screen/DecalMaterial/Shader")
    texture = UsdShade.Shader.Get(m.stage, f"{b}/Screen/DecalMaterial/Texture")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    m.group(f"{b}/Badge", (0.098, -0.174, 0.424), (90, 0, 0))
    m.label(f"{b}/Badge/Decal", "aroma.png", (0, 0, 0), 0.074, 0.025)
    m.rounded(f"{b}/BeanHopperLid", (-0.032, 0.052, 0.465), (0.208, 0.282, 0.012), "black", 0.005)
    m.rounded(f"{b}/BeanHopperInset", (-0.032, 0.052, 0.472), (0.184, 0.257, 0.002), "glass", 0.0008)
    m.rounded(f"{b}/TankBacking", (0.154, 0.025, 0.187), (0.004, 0.274, 0.246), "black", 0.001)
    m.rounded(f"{b}/Water", (0.158, 0.025, 0.150), (0.003, 0.263, 0.164), "water", 0.001)
    m.rounded(f"{b}/Tank", (0.162, 0.025, 0.187), (0.002, 0.272, 0.242), "glass", 0.0008)
    m.box(f"{b}/Waterline", (0.1632, 0.025, 0.232), (0.0005, 0.254, 0.0012), "silver")
    for index, z in enumerate((0.066, 0.310)):
        m.rounded(f"{b}/TankEdge{index}", (0.163, 0.025, z), (0.004, 0.276, 0.005), "black", 0.001)
    m.rounded(f"{b}/TankHandle", (0.164, 0.025, 0.294), (0.006, 0.090, 0.010), "steel", 0.002)
    for index in range(6):
        m.box(f"{b}/TankGraduation{index}", (0.1635, -0.047, 0.103 + index * 0.031), (0.0005, 0.023, 0.001), "silver")
    # An open lower cavity is essential: the tub top is at machine-local Z=0.136.
    m.box(f"{b}/CavityBack", (0, 0.174, 0.086), (0.268, 0.012, 0.140), "rubber", True)
    m.rounded(f"{b}/ServiceLatch", (-0.114, -0.155, 0.070), (0.020, 0.018, 0.018), "black", 0.004)
    m.slider("Spout", "spout_slide", (0, -0.209, 0.160), "Z", 0, 0.075, 0.30)
    s = "/Asset/Spout"
    m.rounded(f"{s}/Carriage", (0, 0, 0.055), (0.080, 0.045, 0.093), "black", 0.008, True)
    for i, x in enumerate((-0.029, 0.029)):
        m.cylinder(f"{s}/ChromeGuide{i}", (x, -0.026, 0.033), 0.008, 0.071, "silver")
        m.ring(f"{s}/Nozzle{i}", (x, -0.026, -0.010), 0.008, 0.0045, 0.018, "silver")
    m.rounded(f"{s}/ThumbGrip", (0, -0.013, 0.009), (0.062, 0.047, 0.010), "steel", 0.004, True)
    m.slider("Tray", "tray_slide", TRAY_IN_MACHINE, "Y", -0.42, 0, 0.65)
    t = "/Asset/Tray"
    m.rounded(f"{t}/Baseplate", (0, -0.045, 0.006), (0.264, 0.390, 0.012), "black", 0.005, True)
    m.rounded(f"{t}/FrontTrim", (0, -0.235, 0.018), (0.264, 0.010, 0.034), "silver", 0.004, True)
    # Slotted grate is one tray body; dark backing remains visible between metal strips.
    for i, x in enumerate((-0.125, 0.125)):
        m.rounded(f"{t}/GrateSide{i}", (x, -0.139, 0.024), (0.010, 0.190, 0.009), "silver", 0.003)
    for i, y in enumerate((-0.229, -0.048)):
        m.rounded(f"{t}/GrateEdge{i}", (0, y, 0.024), (0.242, 0.012, 0.009), "silver", 0.003)
    for i in range(25):
        x = -0.115 + i * 0.0096
        m.rounded(f"{t}/GrateBar{i}", (x, -0.139, 0.024), (0.0055, 0.170, 0.007), "silver", 0.002)
    for i, x in enumerate((-0.030, 0.030)):
        m.cylinder(f"{t}/CupPosition{i}", (x, -0.135, 0.028), 0.012, 0.001, "rubber")
    # Short cradle guides retain the cup without covering its opening or lift path.
    for i, x in enumerate((-0.078, 0.078)):
        m.rounded(f"{t}/CradleGuide{i}", (x, 0.055, 0.024), (0.008, 0.170, 0.024), "black", 0.003, True)
    for i, y in enumerate((-0.031, 0.141)):
        m.rounded(f"{t}/CradleStop{i}", (0, y, 0.021), (0.148, 0.008, 0.018), "black", 0.003, True)
    m.save()


def build_cup():
    """Build a hollow removable grounds container with a silver front and adhered residue."""
    m = Model("grounds_cup")
    m.body("/Asset", 0.24)
    m.root.SetDisplayName("grounds_container")
    width, depth, height = CUP_SIZE
    m.tub("/Asset/Tub", (0, 0, 0), (width / 2 - 0.006, depth / 2 - 0.006), height, 0.006, "black", 0.006)
    m.rounded("/Asset/SilverFascia", (0, -depth / 2 + 0.0015, 0.052), (width - 0.006, 0.003, 0.098), "silver", 0.001)
    for i, x in enumerate((-0.049, 0.049)):
        m.rounded(f"/Asset/GripRib{i}", (x, -0.078, 0.054), (0.007, 0.004, 0.062), "steel", 0.001)
    for i, x in enumerate((-0.069, 0.069)):
        m.rounded(f"/Asset/MouldedTab{i}", (x, 0.049, 0.106), (0.006, 0.034, 0.008), "black", 0.002)
    rng = random.Random(1729)
    for i in range(145):
        x, y = rng.uniform(-0.059, 0.059), rng.uniform(-0.065, 0.065)
        radius = rng.uniform(0.0005, 0.0017)
        m.sphere(
            f"/Asset/Residue/Speck{i}",
            (x, y, 0.0063),
            (radius, radius * 0.8, 0.00035),
            "coffee_light" if i % 5 == 0 else "coffee",
        )
    m.save()


def build_lid():
    """Build a removable gray molded lid with two recessed panels and a raised center grip."""
    m = Model("bin_lid")
    m.body("/Asset", 0.19)
    width, depth, height = LID_SIZE
    m.rounded("/Asset/Plate", (0, 0, height / 2), (width, depth, height), "lid", 0.005, True)
    for i, x in enumerate((-0.065, 0.065)):
        m.rounded(f"/Asset/Recess{i}", (x, 0, height + 0.0003), (0.058, 0.252, 0.002), "bin", 0.0008)
        m.rounded(f"/Asset/RecessFloor{i}", (x, 0, height + 0.001), (0.049, 0.238, 0.002), "lid", 0.0008)
    m.rounded("/Asset/Grip", (0, 0, height + 0.010), (0.031, 0.235, 0.020), "lid", 0.008, True)
    m.rounded("/Asset/GripCrown", (0, 0, height + 0.020), (0.023, 0.215, 0.002), "steel", 0.0008)
    m.save()


def build_cabinet():
    """Build an under-counter waste drawer with integrated hollow organic and recycling bins."""
    m = Model("waste_cabinet")
    m.fixed_base(24)
    b = "/Asset/Base"
    for i, x in enumerate((-0.353, 0.353)):
        m.rounded(f"{b}/Side{i}", (x, 0.008, 0.355), (0.018, 0.728, 0.710), "white", 0.004, True)
        m.box(f"{b}/Rail{i}", (x * 0.96, -0.02, 0.112), (0.013, 0.600, 0.044), "silver")
    m.rounded(f"{b}/Rear", (0, 0.365, 0.290), (0.688, 0.018, 0.580), "white", 0.004, True)
    m.rounded(f"{b}/Plinth", (0, 0.017, -0.035), (0.718, 0.66, 0.070), "black", 0.003, True)
    m.rounded(f"{b}/UpperFascia", (0, -0.349, 0.697), (0.684, 0.021, 0.086), "white", 0.004, True)
    m.slider("Drawer", "drawer_slide", (0, 0, 0), "Y", -0.57, 0, 7)
    d = "/Asset/Drawer"
    m.rounded(f"{d}/Floor", (0, -0.012, 0.014), (0.666, 0.625, 0.028), "white", 0.006, True)
    m.rounded(f"{d}/Front", (0, -0.340, 0.320), (0.680, 0.024, 0.640), "white", 0.004, True)
    m.rounded(f"{d}/TopRail", (0, -0.338, 0.637), (0.676, 0.033, 0.014), "black", 0.005, True)
    m.rounded(f"{d}/HandleUndercut", (0, -0.354, 0.616), (0.620, 0.006, 0.018), "rubber", 0.002)
    for i, x in enumerate((-0.322, 0.322)):
        m.rounded(f"{d}/Runner{i}", (x, -0.008, 0.092), (0.014, 0.590, 0.060), "steel", 0.003)
        m.box(f"{d}/RunnerInset{i}", (x, -0.022, 0.127), (0.020, 0.555, 0.004), "silver")
    # A steel bridge supports the raised organizer above the original drawer floor.
    bin_base = BIN_IN_DRAWER[2]
    platform_bottom = bin_base - 0.012
    support_height = platform_bottom - 0.028
    for index, x in enumerate((-0.308, 0.247)):
        m.rounded(
            f"{d}/OrganizerSupport/Leg{index}",
            (x, -0.02, 0.028 + support_height / 2),
            (0.014, 0.350, support_height),
            "steel",
            0.003,
            True,
        )
    m.rounded(
        f"{d}/OrganizerSupport/Platform",
        (-0.029, -0.02, bin_base - 0.006),
        (0.594, 0.350, 0.012),
        "steel",
        0.004,
        True,
    )
    for index, y in enumerate((-0.170, 0.130)):
        m.rounded(f"{d}/OrganizerSupport/Brace{index}", (-0.030, y, 0.086), (0.568, 0.009, 0.018), "silver", 0.003)
    m.tub(f"{d}/Organic", BIN_IN_DRAWER, BIN_HALF_INTERIOR, BIN_HEIGHT, 0.010)
    m.tub(f"{d}/Recycling", (-0.19, -0.02, bin_base), (0.126, 0.135), BIN_HEIGHT - 0.012, 0.010)
    # Broad rim provides real support for the detached lid at the prescribed parked pose.
    m.rim(
        f"{d}/Recycling/SupportRim",
        (-0.19, -0.02, bin_base + BIN_HEIGHT - 0.012),
        (0.145, 0.160),
        (0.126, 0.135),
        0.012,
        "lid",
    )
    for path, x, texture in [("Organic", 0.12, "organic"), ("Recycling", -0.19, "recycling")]:
        m.group(f"{d}/{path}/Mark", (x, -0.173, bin_base + 0.314), (90, 0, 0))
        m.label(f"{d}/{path}/Mark/Decal", f"{texture}.png", (0, 0, 0), 0.118, 0.028)
    # Low decorative scraps leave the receiving volume open for dynamic coffee clumps.
    for i in range(6):
        angle = i * 1.8
        x, y = 0.12 + 0.058 * math.sin(angle), -0.02 + 0.080 * math.cos(angle)
        leaf = m.sphere(f"{d}/Organic/Scrap{i}", (x, y, bin_base + 0.022), (0.021, 0.008, 0.002), "leaf")
        leaf.AddRotateZOp().Set(i * 37)
    # A folded blue liner and small packaging shapes reproduce the neighboring-bin context.
    for i, x in enumerate((-0.296, -0.084)):
        m.rounded(f"{d}/Recycling/LinerFold{i}", (x, -0.02, bin_base + 0.280), (0.011, 0.245, 0.142), "blue", 0.004)
    for i in range(4):
        m.cylinder(
            f"{d}/Recycling/Bottle{i}", (-0.26 + i * 0.045, -0.052, bin_base + 0.106 + i * 0.008), 0.018, 0.125, "white"
        )
        m.cylinder(
            f"{d}/Recycling/Cap{i}", (-0.26 + i * 0.045, -0.052, bin_base + 0.174 + i * 0.008), 0.012, 0.010, "blue"
        )
    m.save()


def build_kitchen():
    """Build the quartz worktop, black inset sink, faucet, cabinetry, window and oak floor."""
    m = Model("kitchen")
    # Four abutting worktop slabs leave an actual sink opening at X[-.55,-.09], Y[-.12,.34].
    for name, position, size in [
        ("Right", (0.480, 0.005, 0.878), (1.140, 1.290, 0.044)),
        ("Left", (-0.650, 0.005, 0.878), (0.200, 1.290, 0.044)),
        ("Front", (-0.320, -0.380, 0.878), (0.460, 0.520, 0.044)),
        ("Rear", (-0.320, 0.495, 0.878), (0.460, 0.310, 0.044)),
    ]:
        m.box(f"/Asset/Counter/{name}", position, size, "stone", True)
    m.tub("/Asset/Sink", (-0.32, 0.11, 0.714), (0.225, 0.225), 0.184, 0.008, "black", 0.008)
    m.rim("/Asset/Sink/Lip", (-0.32, 0.11, 0.898), (0.238, 0.238), (0.225, 0.225), 0.003, "black")
    m.ring("/Asset/Sink/Drain", (-0.32, 0.11, 0.724), 0.031, 0.019, 0.002, "silver")
    m.cylinder("/Asset/Sink/DrainWell", (-0.32, 0.11, 0.723), 0.019, 0.001, "rubber")
    for i in range(10):
        angle = i * math.tau / 10
        m.cylinder(
            f"/Asset/Sink/DrainHole{i}",
            (-0.32 + 0.025 * math.cos(angle), 0.11 + 0.025 * math.sin(angle), 0.7252),
            0.0023,
            0.0003,
            "rubber",
        )
    # Matte black gooseneck assembled from a smooth swept tube.
    centers = [(-0.32, 0.409, 0.90 + i * 0.018) for i in range(15)]
    for i in range(1, 25):
        angle = i * math.pi / 24
        centers.append((-0.32, 0.319 + 0.090 * math.cos(angle), 1.152 + 0.090 * math.sin(angle)))
    points, faces = [], []
    for index, (x, y, z) in enumerate(centers):
        before = centers[max(0, index - 1)]
        after = centers[min(len(centers) - 1, index + 1)]
        dy, dz = after[1] - before[1], after[2] - before[2]
        length = math.hypot(dy, dz)
        for i in range(16):
            angle = i * math.tau / 16
            radial = 0.012 * math.sin(angle)
            points.append((x + 0.012 * math.cos(angle), y - radial * dz / length, z + radial * dy / length))
    for row in range(len(centers) - 1):
        for i in range(16):
            j = (i + 1) % 16
            faces.append([row * 16 + i, row * 16 + j, (row + 1) * 16 + j, (row + 1) * 16 + i])
    m.mesh("/Asset/Faucet/Tube", points, faces, "black")
    m.cylinder("/Asset/Faucet/Base", (-0.32, 0.409, 0.916), 0.021, 0.032, "black")
    m.cylinder("/Asset/Faucet/Aerator", (-0.32, 0.229, 1.145), 0.012, 0.018, "steel")
    m.cylinder("/Asset/Faucet/Lever", (-0.276, 0.409, 0.980), 0.004, 0.068, "black")
    m.rounded("/Asset/Soap/Bottle", (-0.51, 0.400, 0.960), (0.053, 0.043, 0.120), "white", 0.014)
    m.cylinder("/Asset/Soap/Neck", (-0.51, 0.400, 1.027), 0.010, 0.015, "white")
    m.rounded("/Asset/Soap/Pump", (-0.51, 0.389, 1.039), (0.032, 0.044, 0.010), "white", 0.004)
    m.rounded("/Asset/Sponge", (-0.48, 0.22, 0.735), (0.070, 0.044, 0.022), "sponge", 0.008)
    m.rounded("/Asset/SpongeScrub", (-0.48, 0.22, 0.747), (0.070, 0.044, 0.003), "leaf", 0.001)
    # Right cabinet context stays behind the worktop front and outside the waste drawer.
    m.rounded("/Asset/RightCabinet/Body", (0.558, 0.04, 0.444), (0.976, 1.140, 0.804), "white", 0.004, True)
    for i, x in enumerate((0.313, 0.802)):
        m.rounded(f"/Asset/RightCabinet/Door{i}", (x, -0.542, 0.425), (0.480, 0.022, 0.755), "white", 0.004)
        m.box(f"/Asset/RightCabinet/Handle{i}", (x, -0.556, 0.783), (0.486, 0.008, 0.021), "black")
    m.box("/Asset/BackWall", (0, 0.715, 2.1), (10.0, 0.060, 4.2), "wall")
    m.box("/Asset/Skirting", (0, 0.671, 0.060), (10.0, 0.028, 0.120), "white")
    m.box("/Asset/SkirtingShadow", (0, 0.666, 0.009), (10.0, 0.031, 0.018), "black")
    m.box("/Asset/LeftWall", (-4.97, -2.64, 2.1), (0.06, 6.65, 4.2), "wall")
    m.box("/Asset/Window/Outside", (0.22, 0.678, 1.550), (1.55, 0.008, 0.990), "sky")
    for i, x in enumerate((-0.57, 0.215, 1.00)):
        m.rounded(f"/Asset/Window/Mullion{i}", (x, 0.645, 1.550), (0.032, 0.065, 1.055), "black", 0.005)
    for i, z in enumerate((1.03, 2.07)):
        m.rounded(f"/Asset/Window/Frame{i}", (0.215, 0.645, z), (1.602, 0.065, 0.035), "black", 0.006)
    for i in range(28):
        m.box(f"/Asset/Window/Blind{i}", (0.215, 0.635, 1.057 + i * 0.036), (1.545, 0.042, 0.013), "steel")
    for i, x in enumerate((-0.43, 0.06, 0.38, 0.86)):
        m.box(f"/Asset/Window/BlindCord{i}", (x, 0.610, 1.55), (0.0012, 0.0012, 1.0), "black")
    # Continuous room floor with a single shared grain material and clipped plank boundaries.
    m.box("/Asset/Floor/Underlay", (0, -2.625, -0.0079), (10.0, 6.75, 0.016), "wood")
    m.label("/Asset/Floor/Grain", "oak.png", (0, 0, 0.0003), 1, 1)
    floor_mesh = UsdGeom.Mesh.Get(m.stage, "/Asset/Floor/Grain")
    points, indices, coordinates = [], [], []
    for row in range(36):
        y0 = -6.0 + row * 0.19
        y1 = min(0.75, y0 + 0.188)
        for col in range(13):
            raw_x0 = -5.9 + col * 0.90 + (row % 3) * 0.30
            x0, x1 = max(-5.0, raw_x0), min(5.0, raw_x0 + 0.897)
            if x1 <= x0:
                continue
            start = len(points)
            points.extend([(x0, y0, 0.0003), (x1, y0, 0.0003), (x1, y1, 0.0003), (x0, y1, 0.0003)])
            indices.extend([start, start + 1, start + 2, start + 3])
            u0, u1 = (x0 - raw_x0) / 0.897, (x1 - raw_x0) / 0.897
            coordinates.extend([(u0, 0), (u1, 0), (u1, 1), (u0, 1)])
    floor_mesh.CreatePointsAttr(points)
    floor_mesh.CreateFaceVertexCountsAttr([4] * (len(points) // 4))
    floor_mesh.CreateFaceVertexIndicesAttr(indices)
    UsdGeom.PrimvarsAPI(floor_mesh).GetPrimvar("st").Set(coordinates)
    # Reference-inspired countertop context, outside the tray and robot working area.
    m.cylinder("/Asset/Kettle/Base", (0.941, 0.405, 0.906), 0.069, 0.012, "black")
    m.cylinder("/Asset/Kettle/Body", (0.941, 0.405, 1.001), 0.064, 0.178, "silver")
    m.sphere("/Asset/Kettle/Shoulder", (0.941, 0.405, 1.081), (0.063, 0.063, 0.023), "silver")
    m.cylinder("/Asset/Kettle/Lid", (0.941, 0.405, 1.103), 0.046, 0.007, "black")
    m.rounded("/Asset/Kettle/LidGrip", (0.941, 0.405, 1.112), (0.030, 0.015, 0.013), "black", 0.004)
    m.group("/Asset/Kettle/Handle", (0.999, 0.405, 1.005), (90, 0, 0))
    m.ring("/Asset/Kettle/Handle/Ring", (0, 0, 0), 0.054, 0.043, 0.014, "black")
    m.rounded("/Asset/Kettle/Spout", (0.882, 0.405, 1.058), (0.036, 0.029, 0.026), "silver", 0.005)
    m.rounded("/Asset/Kettle/SpoutOpening", (0.864, 0.405, 1.064), (0.001, 0.020, 0.011), "black", 0.0004)
    m.ring("/Asset/CeramicCup/Body", (0.948, 0.153, 0.941), 0.033, 0.029, 0.080, "white")
    m.cylinder("/Asset/CeramicCup/Bottom", (0.948, 0.153, 0.903), 0.032, 0.006, "white")
    m.group("/Asset/CeramicCup/Handle", (0.981, 0.153, 0.944), (90, 0, 0))
    m.ring("/Asset/CeramicCup/Handle/Ring", (0, 0, 0), 0.021, 0.015, 0.007, "white")
    m.save()


def build_grounds():
    """Build one irregular compressed grounds clump with a convex hull and granular surface."""
    m = Model("grounds")
    m.body("/Asset", 0.008)
    rng = random.Random(51)
    points, faces = list(GROUND_VERTICES), []
    rows, columns = GROUND_ROWS, GROUND_COLUMNS
    for row in range(rows):
        for col in range(columns):
            following = (col + 1) % columns
            faces.append([
                row * columns + col,
                row * columns + following,
                (row + 1) * columns + following,
                (row + 1) * columns + col,
            ])
    m.mesh("/Asset/Clump", points, faces, "coffee")
    collision = m.mesh("/Asset/CollisionHull", GROUND_COLLISION_VERTICES, GROUND_COLLISION_FACES, "coffee")
    collision.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexHull")
    for i in range(140):
        phi, theta = rng.uniform(0.1, math.pi - 0.1), rng.uniform(0, math.tau)
        radius = GROUND_RADIUS * (0.88 + 0.08 * math.sin(theta * 5 + phi * 7) + 0.035 * math.cos(theta * 9))
        position = (
            radius * math.sin(phi) * math.cos(theta),
            radius * math.sin(phi) * math.sin(theta),
            radius * math.cos(phi) * 0.83,
        )
        grain = rng.uniform(0.0003, 0.0008)
        m.sphere(
            f"/Asset/Grains/Grain{i}", position, (grain, grain, grain * 0.7), "coffee_light" if i % 3 else "coffee_dark"
        )
    m.save()


def validate():
    """Check exported joints, hollow receptacles, mass properties and default prims."""
    contracts = {
        "coffee_machine": {"spout_slide": ("Z", 0, 0.075), "tray_slide": ("Y", -0.42, 0)},
        "waste_cabinet": {"drawer_slide": ("Y", -0.57, 0)},
    }
    for filename in ("coffee_machine", "waste_cabinet", "grounds_cup", "bin_lid", "grounds", "kitchen"):
        stage = Usd.Stage.Open(str(ASSETS / f"{filename}.usda"))
        assert stage and stage.GetDefaultPrim(), filename
        assert UsdGeom.GetStageMetersPerUnit(stage) == 1
        colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
        assert colliders, f"Missing colliders: {filename}"
        for joint_name, (axis, lower, upper) in contracts.get(filename, {}).items():
            joint = UsdPhysics.PrismaticJoint.Get(stage, f"/Asset/{joint_name}")
            assert joint.GetAxisAttr().Get() == axis
            assert abs(joint.GetLowerLimitAttr().Get() - lower) < 1e-6
            assert abs(joint.GetUpperLimitAttr().Get() - upper) < 1e-6
            assert joint.GetBody0Rel().GetTargets() == [Sdf.Path("/Asset/Base")]
        if filename in contracts:
            assert stage.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
            assert UsdPhysics.FixedJoint.Get(stage, "/Asset/FixedBase")
        if filename == "grounds":
            assert [str(prim.GetPath()) for prim in colliders] == ["/Asset/CollisionHull"]
            hull = UsdGeom.Mesh.Get(stage, "/Asset/CollisionHull")
            points = hull.GetPointsAttr().Get()
            assert len(points) == len(GROUND_COLLISION_VERTICES) == 24
            assert len(hull.GetFaceVertexCountsAttr().Get()) == len(GROUND_COLLISION_FACES) == 26
            for point, expected in zip(points, GROUND_COLLISION_VERTICES):
                assert max(abs(point[index] - expected[index]) for index in range(3)) < 1e-8
            for face in GROUND_COLLISION_FACES:
                a, b, c = (Gf.Vec3d(*points[index]) for index in face[:3])
                normal = Gf.Cross(b - a, c - a).GetNormalized()
                distance = Gf.Dot(normal, a)
                assert distance > 0, "Collision hull face must point outward"
                for point in points:
                    assert Gf.Dot(normal, Gf.Vec3d(*point)) <= distance + 1e-8, "Hull must be convex"
                for point in GROUND_VERTICES:
                    assert Gf.Dot(normal, Gf.Vec3d(*point)) <= distance + 1e-8, "Hull must enclose render mesh"
        print(f"Validated {filename}: {len(colliders)} colliders")


def main():
    """Regenerate and validate all committed coffee-disposal assets."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    textures()
    build_machine()
    build_cup()
    build_lid()
    build_cabinet()
    build_kitchen()
    build_grounds()
    validate()


if __name__ == "__main__":
    main()
