# Voxel preview

`render.py` draws a scene the way `MCPassthrough.fx` draws it in voxel mode (`VoxActive = true`), without Rocket League,
so the data the voxel core exports (and later the whole pipeline) can be checked on any machine. The shader's voxel path
is ported line by line to numpy, vectorised over pixels and computed in float32 in the shader's order of operations.

```
python3 tools/preview/render.py <scene_dir> [--out file.png] [--debug cells] [--size WxH]
python3 tools/preview/render.py --demo tools/preview/out/demo
```

- `--out`: the PNG to write (default `<scene_dir>/preview.png`, or `preview_<debug>.png`).
- `--debug`: one of the shader's debug views instead of the composite: `host-depth` (DebugView 1, RL depth in 1 m
  bands), `vox-depth` (2), `depth-diff` (3), `cells` (4: RL's surface red where it is in a shell cell, green where it is
  in a dug one).
- `--size`: render at another resolution (the field of view stays vertical).
- `--demo <dir>`: writes a small synthetic scene to `<dir>` and renders it: RL's floor as a shell layer at `y = 63` over
  the pitch and goals (invisible, id 0, as the exporter writes the barrier), a few untextured blocks on the floor, one dug
  cell with dirt under it and dirt in the shell cells around it (the hole's sides), and Steve walking nearby. It needs
  no atlas (every slot is -1) and writes the plain skin as `mcrl_steve.png`.

Renders go to `tools/preview/out/` (git-ignored) or anywhere outside the repository. Python 3.10+, numpy and Pillow.
640 x 360 takes about a second.

## Scene directory

Written by the C++ exporter (`core/tools/vox_export.cpp`) or by `--demo`. Formats are those of
`docs/SHADER_CONTRACT.md`:

| File | Contents |
|---|---|
| `vox.r16` | MCVOX: raw little-endian uint16, 128 x 5120, row-major (cell `(x, y, z)` relative to `(-64, 48, -64)` at index `x + 128 * (z + 128 * y)`) |
| `cells.r8` | MCHOLES: raw uint8, same layout; 0 open, 120 shell, 240 dug (read as `/ 255` and classified like the shader: `> 0.7` dug, `> 0.25` shell) |
| `types.rgba32f` | MCTYPES: raw little-endian float32, 256 x 384 x 4 |
| `scene.json` | camera, Steve, arena, asset paths (below) |

```json
{"width": 640, "height": 360, "fovY": 70,
 "camera": {"pos": [x, y, z], "yaw": 0, "pitch": 12},
 "steve": {"pos": [x, y, z, shown], "look": [bodyYaw, headYaw, pitch, crouch], "anim": [walkPos, walkSpeed, armSwing, 0],
           "item": [slot, blockId, 0, 0]},
 "arena": {"uuPerBlock": 100, "floorY": 64, "halfWidth": 4096, "halfLength": 5120, "height": 2044, "cornerSum": 8064,
           "goalHalfWidth": 893, "goalHeight": 642.775, "goalDepth": 880},
 "atlas": "assets/mcrl_atlas.png", "skin": "assets/mcrl_steve.png"}
```

- `camera`: Minecraft's convention, degrees: yaw 0 looks toward +z, yaw 90 toward -x, pitch > 0 looks down. It
  becomes `HostCamRow0..2` (rows of the camera-to-world rotation; camera space x right, y up, looking down -z) and
  `HostCamPos`. `fovY` is the vertical field of view.
- `steve`: the `StevePos` / `SteveLook` / `SteveAnim` / `SteveItem` uniforms. Missing: Steve is not shown. The shader
  draws neither crouching nor the held item, so neither does the preview.
- `arena`: `rl::ArenaMapping` and `rl::SoccarDimensions` (defaults if missing).
- `atlas`, `skin`: absolute, or relative to the scene directory, the current directory or the repository root. Without
  an atlas every face is drawn untextured (the block's own noised colour); without a skin Steve gets a plain built-in
  one (face, hair, shirt, trousers). Either warns when a named file is missing, or when none is named but a drawn block
  has a textured face (atlas) or Steve is shown (skin). A file of the wrong size is resized as ReShade would, with a
  warning.

## What is ported

From `MCPassthrough.fx` (voxel mode), each as its own function with the shader's name:

- `vox_at`, `shell_at`, `type_row`, `atlas` (`atlas_texel`), `block_color`: texel addressing, the untextured colour
  `own * (0.8 + 0.35 * frac(sin(dot(px, (12.9898, 78.233))) * 43758.5453))` and the tint bits.
- `trace_voxels`: the grid walk with its exact stepping (entry, the `1e-4` nudge into the first cell, `tNext` / `tDelta`,
  tie-breaking toward z then y, the 320-cell limit, the origin's own cell never drawn), the face light (top 1.0, z sides
  0.8, x sides 0.6, bottom 0.5), the uv rules, cutout (`alpha > 0.5`) and the shell rule
  `!(!hole && shell_at(cell) == 1 && tSurf < t + 1.5)`.
- `steve_box`, `trace_steve`: the six boxes, their skin uvs, the quick reject, walk and arm swing, head yaw / pitch.
- `arena_cell` and `PS_Composite`'s voxel branch: the hole test at RL's surface (`zh + max(0.03, zh * 0.01)` behind it,
  `zh - max(0.02, zh * 0.005)` in front of it, only for `zh < 400`), `tMax`, `visible = hole || zm < zh + allow`, the
  shell hiding rule (`zh < zm + 1.5` and the point `zm + 0.03` along the ray in a shell cell), cover, and the four debug
  views. `allow` uses the effect's defaults: `DepthBias` 0.04, `SlopeBias` 0, `MaxBias` 3 (`ddy` is emulated per 2 x 2
  quad in case `SlopeBias` is changed in `render.py`).

HLSL semantics emulated: `tex2Dlod` with point sampling and CLAMP addressing (texel `floor(uv * size)`, clamped), `int()`
truncation toward zero (saturating), `frac(x) = x - floor(x)`, `floor`, `sign()`, `step()`, `saturate`, integer `/` and
`%`, the `abs(d) < 1e-6` epsilon trick (so `sign(dd)` is never 0 and `1 / dd` stays finite), R16_UNORM read as
`round(r * 65535)`, the R8_UNORM thresholds, the order of float32 operations as written, and the final UNORM rounding.

## What is not

- **RL's picture and depth buffer** don't exist offline. RL's surface is synthesised: per pixel, `zh` is the linear depth
  (camera -z) where the pixel's ray leaves the closed soccar volume (the pitch box cut by the 45-degree corners
  `|x| + |y| <= cornerSum`, united with the two goal boxes; RL units via `x_rl = x * uu`, `y_rl = z * uu`,
  `z_rl = (y - floorY) * uu`), or first enters it from a camera outside; no surface: `zh = 1e9`, as `host_linear`
  returns past the far plane. The curved wall ramps, the ball, cars and goal posts are left out. The host colour is a
  plain stand-in: green pitch with stripes, halfway line and centre circle, blue-grey walls, team-coloured goals
  (orange at +z, blue at -z), dark ceiling, one fixed light and a little distance fade.
- **PS_Light, relight, PS_Bright and PS_Final** are skipped: no relighting by RL's light, colour grade or haze (`lit` is
  the voxel colour as drawn), no glow, edge softening, contact shadows, shake or portal warp. The output is the
  composite pass, saturated and rounded to 8 bits.
- **Bit-exactness with a GPU** is not guaranteed: the GPU may fuse multiply-adds or reorder them, its `rcp`, `sin` and
  `cos` are approximate (the untextured noise, `frac(sin(large) * 43758.5)`, amplifies that, so its texel pattern
  differs while its range does not), and RL's real depth is quantised by the depth buffer. Everything else (which cell,
  which face, which texel, hidden or shown) matches except within float rounding of a boundary.

## Notes

- The shader turns Steve's head by `xRot = -pitch`, so with Minecraft's convention (pitch > 0 looks down) his head tilts
  up; the preview mirrors that.
- The exporter's world fills shell cells with the invisible barrier. Through a dug cell in RL's floor the hole's sides at
  the floor layer are therefore not drawn: rays pass through the invisible neighbours and see the ground layer under
  RL's floor, so a one-deep hole reads as a dirt patch with a ledge rather than a pit. The demo puts visible dirt in the
  shell cells around its hole, which the shell rule hides everywhere but through the hole.

## Tests

```
python3 -m unittest discover -s tools/preview -p 'test_*.py'
```

Texel addressing against the contract, the grid walk against a brute-force float64 reference on random rays (cells,
crossing times and first block hit), face light and uv for axis-aligned rays, tint and cutout, the camera basis, the
synthesised arena depth for each wall, the floor, the ceiling, the corners and the goals, Steve's skin uvs and
animation, the shell and hole rules on small hand-built scenes (floor and wall holes, a block in a shell cell, Steve in
a wall's shell), the scene files, the command line, and the speed of a 320 x 180 demo render.
