# Credits

## Textures and skin (`assets/`)

`assets/mcrl_atlas.png`, `assets/mcrl_atlas.txt`, `assets/mcrl_blocks.txt` and `assets/mcrl_steve.png` are built by
`tools/assets/build_assets.py` from the texture folder of **VoxeLibre** (formerly MineClone 2), a free game for the
Luanti (formerly Minetest) engine: <https://git.minetest.land/VoxeLibre/VoxeLibre>. The texture folder's release
notes include screenshots up to VoxeLibre 0.92, so it is from that release or later.

- Most textures come from the **Pixel Perfection** texture pack by Hugh "XSSheep" Rutland, licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- VoxeLibre's other media are licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) unless the
  mod they belong to says otherwise; some are CC0.

The build changes them: textures are cropped to their first frame and rearranged into one atlas, three textures are
composited from two VoxeLibre textures each (the grass block side, with the overlay tinted; copper ore; crying
obsidian), block colours are averaged from textures, and the 64 x 32 skin is converted to the 64 x 64 layout. These
derived files are shared under the same terms; see [assets/LICENSE.md](assets/LICENSE.md).

**Before distribution:** the texture folder this was built from is a flat extract that lost VoxeLibre's `LEGAL.md`
and the per-mod READMEs / licence files that name each texture's author and licence (including the skin,
`character.png`). Add them from VoxeLibre's source (the same commit as the textures), check every texture used here
against them, and ship them with `assets/`. Until then this attribution is incomplete.

## Not affiliated

This project is not affiliated with Mojang Studios, Microsoft, Psyonix or Epic Games. Block names follow
Minecraft's ids for familiarity; no Minecraft files are used or shipped.
