# Asset pipeline

`build_assets.py` turns VoxeLibre's texture folder into exactly the files the voxel-mode shader (`MCPassthrough.fx`)
and the core read. The data contract is [docs/SHADER_CONTRACT.md](../../docs/SHADER_CONTRACT.md); the block list
format is `vox::BlockRegistry::parse` (`core/include/vox/block_registry.hpp`).

```
python3 tools/assets/build_assets.py --textures <VoxeLibre textures folder> --out assets [--sheet sheet.png]
python3 -m unittest discover -s tools/assets -p 'test_*.py'
```

Needs Python 3.10+, Pillow and numpy. `--skin <png>` picks another skin (default `<textures>/character.png`);
`--sheet <png>` also writes a contact sheet of every block's top, side and bottom (8x, tinted as the shader tints,
see-through texels on a checkerboard) for checking the block list by eye.

## Outputs (`assets/`)

| File | Contents |
|---|---|
| `mcrl_atlas.png` | 1024 x 1024 RGBA, 64 x 64 slots of 16 x 16; slot `s` at `((s % 64) * 16, (s // 64) * 16)` |
| `mcrl_atlas.txt` | `<slot> <name>` per slot |
| `mcrl_blocks.txt` | `<id> <name> <top> <side> <bottom> <solid> <visible> <breakable> <tint_faces> <tint> <color>` |
| `mcrl_steve.png` | 64 x 64 RGBA skin, Minecraft's modern layout |

Running the build twice gives byte-identical files. Nothing is written unless every check passes.

### Atlas

1. **Generated textures** (`blocks.GENERATED`, named `mcrl_...`) take slots `0..k-1` in list order. The shader has
   one texture and one tint per face, so overlays VoxeLibre draws at run time are baked:
   - `mcrl_grass_block_side`: `default_dirt` with `mcl_core_grass_block_side_overlay` multiplied by the grass colour
     drawn over it. The grass block's side face is then *not* tinted (only its top is).
   - `mcrl_copper_ore`: `default_stone` under `mcl_copper_ore` (VoxeLibre's copper ore texture is only the ore
     spots, with soft alpha).
   - `mcrl_crying_obsidian`: `default_obsidian` under `mcl_core_crying_obsidian` (also only an overlay).
2. **Every PNG in the folder** (not its subfolders) whose width is 16 and height a positive multiple of 16, sorted
   by file name, named by the file name without `.png`. Animated textures are vertical strips: the first frame
   (top 16 x 16) is used. Other sizes and non-PNG entries are skipped and counted; more than 4096 textures is an
   error.

Images are expanded to RGBA as stored (palette transparency applied). The shader draws texels with alpha > 0.5
(128 and up), so glass and leaves are cutouts and semi-transparent ice and slime draw as opaque.

### Block list (`blocks.py`)

`blocks.BLOCKS` is the source of truth for ids: list order is the id, from 1 (0 is air, implicit). Saved worlds
store blocks by name, so only append. Names are Minecraft Java's ids. The first block is `barrier`: no textures,
solid, not visible (exported to the shader as air), breakable - the Rocket League arena shell. `bedrock` is the
only unbreakable block. Every texture name must exist in the atlas, or the build fails and lists all that are
missing.

- **Tint.** Colours are Minecraft's plains defaults: grass `#91BD59`, foliage `#77AB2F`, birch `#80A755`, spruce
  `#619961`. VoxeLibre ships per-biome palettes (`mcl_core_palette_grass.png`, `mcl_core_palette_foliage.png`), but
  which entry belongs to which biome is in VoxeLibre's Lua code, not in the texture folder, so they do not name a
  better default.
  A face is tinted only if its texture is greyscale: Minecraft tints greyscale textures, and tinting an
  already-coloured one would colour it twice. The build measures each requested tinted face's mean HSV saturation
  over its drawn texels and drops the tint at 0.25 or above (logged). With VoxeLibre's textures: the grass top
  (0.20), oak leaves (0.19) and jungle leaves (0.20) are tinted; birch (0.64, orange), spruce (0.32), acacia (0.35)
  and dark oak (0.37) leaves are already coloured and are not. `tint` is written as `#ffffff` when no face is tinted.
- **Colour** (used by the shader for faces without a texture) is the average colour of the side texture's drawn
  texels (alpha >= 128), as stored in the atlas, untinted: the shader multiplies the tint into untextured faces too.
  The barrier's is `#000000`.
- One texture covers all four sides, so the furnace and crafting table show their fronts on every side. The
  bookshelf's top and bottom are oak planks.

### Skin

A 64 x 64 skin is copied as is. A legacy 64 x 32 skin (VoxeLibre's `character.png`) goes onto the top half of a
transparent 64 x 64 image, and the left leg and arm are made from the right ones as Minecraft's legacy-skin
conversion does: twelve 4 x 4 / 4 x 12 face regions, each copied with a horizontal mirror
(`build_assets.LEGACY_LIMB_COPIES`). The build then walks every box `trace_steve()` draws (head, body, arms, legs,
with the shader's `steve_box()` UV layout; left arm at `(32, 48)`, left leg at `(16, 48)`) and fails if any face it
samples has see-through texels, since the shader would show holes in Steve.

## Licence

The outputs are derived from VoxeLibre's media (CC BY-SA); see [assets/LICENSE.md](../../assets/LICENSE.md) and
[CREDITS.md](../../CREDITS.md).
