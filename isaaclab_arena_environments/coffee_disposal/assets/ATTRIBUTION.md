# Coffee disposal asset provenance

All six USDA models and their PNG textures are original procedural artwork made
for Isaac Lab-Arena in 2026 and distributed under the repository's Apache-2.0
license. Regenerate them with `coffee_disposal/build_assets.py` using Isaac Sim's
Python runtime. No external model, photograph, video frame, downloaded material,
or extracted commercial product logo is included.

The user-provided `dispose_coffee.mp4` informed the visual arrangement and motions:
a silver/black automatic coffee machine, adjustable twin dispenser, removable
slotted tray and grounds tub, white kitchen, black sink/faucet, and two-bin waste
drawer with a detached molded lid. Dimensions, contact geometry, fictional AROMA
branding, textures, and mesh topology were authored for this simulation; they are
not measurements or a manufacturer CAD model. The source footage and contact
sheets remain outside the repository.

The builder reuses the repository's Apache-2.0 `pc_upgrade/build_assets.py`
primitive-authoring methods and rounded-rectangle helper. Its independent model
constructor never changes the PC builder's output directory or assets. All coffee
geometry, palette, decals and grain were authored here. Decal lettering is
rasterized with the system DejaVu Sans and DejaVu Sans Bold fonts (DejaVu Fonts
license, based on Bitstream Vera); no font binaries are redistributed. The screen
symbols, cup illustrations, oak grain and sorting labels are original Pillow
artwork.

## Physical contract

- `coffee_machine.usda`: fixed base with `Base`, `Spout`, and `Tray` bodies;
  `spout_slide` is Z [0, 0.075] m and `tray_slide` is Y [-0.42, 0] m.
- `waste_cabinet.usda`: fixed `Base`, sliding `Drawer`, and Y `drawer_slide`
  [-0.57, 0] m. Both open bins and their rims belong to the drawer body. A
  steel platform and two support legs raise their bottoms to drawer-local
  Z = 0.16 m, making the working height compatible with the countertop robot.
  The raised closed lid retains 16 mm of clearance beneath the inset sink.
- `grounds_cup.usda`: 0.24 kg open receptacle, origin at bottom center,
  display name `grounds_container`; adhered decorative residue has no collision.
- `bin_lid.usda`: 0.19 kg removable rigid lid, origin at plate bottom center;
  its raised molded grip remains part of the same rigid body.
- `grounds.usda`: one 0.008 kg irregular compressed clump with separate detailed
  render and contact meshes. The invisible contact mesh is a convex
  rhombicuboctahedron with 24 vertices and 26 planar faces, scaled to enclose all
  360 primary render vertices. It stays below both GPU convex cooking limits,
  so cooking does not need to reduce its vertices or faces. The task and asset
  builder share its exact coordinates through `GROUND_COLLISION_VERTICES`;
  decorative surface grains do not participate in collision. The environment
  creates the eight instances.
- `kitchen.usda`: static countertop at 0.90 m, open sink, contextual cabinets,
  faucet, window/blinds, continuous wooden room floor, and decorative kettle and
  ceramic cup. These contextual objects have no task actions or moving collisions.
  No geometry is embedded in other props.

Render bevels and fine detail are separate from simple contact primitives.
Receptacles use separate floor and wall contacts, preserving their open volume.
The visible botanical scraps and recycling packaging are decorative context;
coffee grounds are the only dynamic disposal contents. This is rigid-clump bulk
disposal, not a particle/fluid or appliance-electrical simulation.
