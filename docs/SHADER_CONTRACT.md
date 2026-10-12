# Voxel-mode shader contract

What `MCPassthrough.fx` (the ReShade effect from the friend kit) reads in **voxel mode** (`VoxActive = true`), where it
draws the blocks and Steve itself instead of compositing a frame rendered by Minecraft. The standalone core
(`core/`) produces exactly this data, so the existing effect draws a world with no Minecraft running.

Everything below is taken from the shader source; where the shader leaves something open, it says so.

## Coordinates

- World coordinates are Minecraft's: x east, y up, z south, 1 unit = 1 block = 1 metre. Distances in the shader are in
  metres (RL's depth is linearised with `HostPlanes` in metres).
- The voxel grid is fixed in the shader: cells `x in [-64, 64)`, `y in [48, 88)`, `z in [-64, 64)`, i.e. corner
  `(-64, 48, -64)`, size `128 x 40 x 128` (`trace_voxels`: `lo = (-64, 48, -64)`, `hi = (64, 88, 64)`, cells clamped
  to `(127, 39, 127)`).
- Camera space (for `HostCamRow*`): x right, y up, looking down -z. A pixel's ray is
  `((u*2-1) * tanHalfFovY * aspect, (1-v*2) * tanHalfFovY, -1)`.

## Textures (bound by the add-on via texture semantics)

### `MCVOX` - block per cell

- Size `128 x 5120` (width `size.x`, height `size.z * size.y`), single channel, read as `r * 65535` and rounded:
  a 16-bit UNORM texture (`R16_UNORM`) holding the block id.
- Texel for cell `c` relative to the corner: `(c.x, c.z + 128 * c.y)`.
  As a row-major array with width 128: `index = c.x + 128 * (c.z + 128 * c.y)`.
- `0` = nothing drawn. Any other value is a block id looked up in `MCTYPES`. Blocks that must not be drawn (the
  invisible arena barrier) are written as `0`.

### `MCHOLES` - arena cells

- Same size and texel layout as `MCVOX`, read from `.r` with point sampling.
- Classified as `r > 0.7`: **dug** (2), `r > 0.25`: **shell** (1), else **open** (0). The shader's comment gives the
  values as "0 open, ~0.47 shell, ~0.94 dug": as `R8_UNORM` that is `0`, `120` (0.4706), `240` (0.9412).
- **Shell**: the cell containing RL's surface (a wall, the floor, the ceiling). Minecraft blocks in a shell cell are
  behind RL's surface and hidden (unless the pixel looks into a hole). When the surface lies exactly on a cell
  boundary, the shell is the cell *behind* the surface (for the floor at `y = 64`: the cells at `y = 63`); the
  shader's hole test samples just behind the surface for the floor and just in front of it for walls that sit inside
  their cell.
- **Dug**: a shell cell Steve broke. RL's surface is gone there: the shader looks past RL's surface into the hole.

### `MCTYPES` - block types

- Size `256 x 384`, four channels, sampled at `((id % 256 + 0.5) / 256, (id / 256 + row * 128 + 0.5) / 384)`.
  So ids up to `256 * 128 - 1 = 32767`. Values are used as exact integers (atlas slots up to 4095, bit masks), so the
  texture must be **32-bit float** (`RGBA32F`); `RGBA16F` cannot hold odd integers above 2048.
- Row block `+0` (texel row `id / 256`): `x, y, z` = atlas slot of the top, side and bottom face (`-1`: no texture,
  draw the block's own colour); `w` = solid (1/0; not read by the shader, used for collision).
- Row block `+128`: `rgb` = tint (0..1) multiplied into tinted faces; `w` = tinted-faces bit mask (bit 0 top, bit 1
  side, bit 2 bottom), read as `int(w)`.
- Row block `+256`: `rgb` = the block's own colour (0..1), used for faces without a texture (noised per texel);
  `w` unused (written as 1).
- Face index: 0 top (+y face seen from above), 1 side (x and z faces), 2 bottom.

### Atlas: `mcrl_atlas.png` (file texture, `reshade-shaders\Textures`)

- `1024 x 1024` RGBA8, 64 x 64 slots of 16 x 16. Slot `s` is at pixel `((s % 64) * 16, (s / 64) * 16)`.
- Alpha `<= 0.5` is see-through (cutout: leaves, glass, flowers).
- `mcrl_atlas.txt` lists `"<slot> <name>"` per line (the shader does not read it; tools and the plugin do).

### Skin: `mcrl_steve.png`

- `64 x 64` RGBA8, Minecraft's modern skin layout (wide 4-pixel arms). The shader reads the left arm at `(32, 48)`
  and the left leg at `(16, 48)`, so a legacy `64 x 32` skin must be converted (left limbs mirrored from the right).

## Uniforms (voxel mode)

| Uniform | Meaning |
|---|---|
| `VoxActive` | true: draw the voxel world |
| `HostCamRow0..2` | rows of RL's camera-to-world rotation, in Minecraft coordinates (`world = R * cameraRay`) |
| `HostCamPos` | RL's camera position, Minecraft coordinates |
| `StevePos` | `xyz` feet position; `w` shown (> 0.5) |
| `SteveLook` | `x` body yaw, `y` head yaw, `z` pitch (degrees, Minecraft's convention: front = `(-sin yaw, 0, cos yaw)`); `w` crouching |
| `SteveAnim` | `x` walk position, `y` walk speed (limb swing amount), `z` arm swing 0..1 |
| `SteveItem` | `x` held item's atlas slot (-1 none), `y` held block's id (0 none) |

## Not in the contract (the plugin decides)

- The scale and axes between RL units and Minecraft blocks, and where RL's floor is in Minecraft's `y`. The plugin
  already converts RL's camera into `HostCamRow*` / `HostCamPos`; the arena generator in
  `games/rocketleague` takes the same mapping as a parameter and must be set to match it.

## Known issues in the shader

Found while porting the voxel path to Python for `tools/preview` (the preview copies the shader as written):

- **Steve's head pitch looks inverted.** `trace_steve` turns the head by `xRot = -headPitch`; with `SteveLook.z` in
  Minecraft's convention (positive looks down) that tilts his head up in the shader's y-up model frame. The arms and
  legs use Minecraft's own signs. Fix it in the shader (`+headPitch`) or negate the pitch the add-on sends; decide when
  the effect is in this repo.
- **Exact ties in the grid walk** step z before y before x (`trace_voxels`), the opposite of `vox::raycast`. Only rays
  through an exact edge or corner differ, which camera rays practically never are.

## Notes for the plugin

- Atlas names are now VoxeLibre file names (`default_stone`, `mcl_core_grass_block_top`, generated ones `mcrl_...`),
  not the friend kit's `minecraft:block/...`; the shader doesn't read them, but anything in the plugin that looked
  textures up by name must use the new ones.
- Shell cells under RL's floor hold visible ground (dirt) rather than the invisible barrier (`rl::ArenaOptions::
  shellTakesGround`): the shader hides blocks in shell cells unless the pixel looks into a hole, and this gives a dug
  hole real walls. Walls and the ceiling keep the barrier while nothing solid is outside them.
