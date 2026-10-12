"""The block list and the generated textures: the source of truth for block ids (list order = id, from 1).

Texture names are VoxeLibre file names without ".png" (build_assets.py fails if one is missing). Block names are
Minecraft Java's ids: saved worlds store blocks by name, so never rename or reorder existing entries; append new
blocks at the end.
"""

from dataclasses import dataclass

Rgb = tuple[int, int, int]

# Tint colours: Minecraft's plains defaults. VoxeLibre ships per-biome palettes (mcl_core_palette_grass.png,
# mcl_core_palette_foliage.png), but which entry is which biome lives in its Lua code, not in the texture folder,
# so the images alone do not name a better default.
GRASS: Rgb = (0x91, 0xBD, 0x59)
FOLIAGE: Rgb = (0x77, 0xAB, 0x2F)
BIRCH_FOLIAGE: Rgb = (0x80, 0xA7, 0x55)
SPRUCE_FOLIAGE: Rgb = (0x61, 0x99, 0x61)
WHITE: Rgb = (0xFF, 0xFF, 0xFF)
BLACK: Rgb = (0x00, 0x00, 0x00)

# The shader's face bits (MCTYPES row +128 w): bit 0 top, bit 1 side, bit 2 bottom.
TOP = 1
SIDE = 2
BOTTOM = 4
ALL = TOP | SIDE | BOTTOM


@dataclass(frozen=True)
class Generated:
	"""A texture composed at build time: `overlay` (multiplied by `tint` if set) drawn over `base` (both first
	frames). The shader has one texture and one tint per face, so overlays VoxeLibre draws at run time are baked."""

	name: str
	base: str
	overlay: str
	tint: Rgb | None = None


@dataclass(frozen=True)
class Block:
	"""One block type. Faces name atlas textures (None: no texture, drawn in `color`). A face in `tint_faces` is
	multiplied by `tint` only if its texture is greyscale: build_assets.py drops the bit for an already-coloured
	texture (Minecraft tints greyscale textures; tinting a coloured one would colour it twice)."""

	name: str
	top: str | None
	side: str | None
	bottom: str | None
	solid: bool = True
	visible: bool = True
	breakable: bool = True
	tint_faces: int = 0
	tint: Rgb = WHITE
	# None: the average opaque colour of the side texture as stored in the atlas (untinted: the shader multiplies
	# the tint into the colour as well).
	color: Rgb | None = None


def cube(name: str, texture: str, **kw) -> Block:
	return Block(name, texture, texture, texture, **kw)


def column(name: str, top: str, side: str, bottom: str | None = None, **kw) -> Block:
	return Block(name, top, side, bottom if bottom is not None else top, **kw)


# Generated textures take the first atlas slots, in this order.
GENERATED: list[Generated] = [
	# VoxeLibre draws the grass side as dirt with a grass-coloured overlay; the side face is then not tinted.
	Generated("mcrl_grass_block_side", "default_dirt", "mcl_core_grass_block_side_overlay", GRASS),
	# Ore and crying obsidian textures that are overlays (VoxeLibre: "default_stone.png^mcl_copper_ore.png").
	Generated("mcrl_copper_ore", "default_stone", "mcl_copper_ore"),
	Generated("mcrl_crying_obsidian", "default_obsidian", "mcl_core_crying_obsidian"),
]

# Minecraft's 16 colours in its order, with VoxeLibre's file suffixes for wool, concrete and terracotta.
COLOURS: list[tuple[str, str, str]] = [
	# (Minecraft colour, wool texture, concrete / terracotta suffix)
	("white", "wool_white", "white"),
	("orange", "wool_orange", "orange"),
	("magenta", "wool_magenta", "magenta"),
	("light_blue", "mcl_wool_light_blue", "light_blue"),
	("yellow", "wool_yellow", "yellow"),
	("lime", "mcl_wool_lime", "lime"),
	("pink", "wool_pink", "pink"),
	("gray", "wool_dark_grey", "grey"),
	("light_gray", "wool_grey", "silver"),
	("cyan", "wool_cyan", "cyan"),
	("purple", "wool_violet", "purple"),
	("blue", "wool_blue", "blue"),
	("brown", "wool_brown", "brown"),
	("green", "wool_dark_green", "green"),
	("red", "wool_red", "red"),
	("black", "wool_black", "black"),
]

BLOCKS: list[Block] = [
	# The Rocket League arena shell: collides, never drawn (exported to the shader as air), and digging is allowed.
	Block("barrier", None, None, None, solid=True, visible=False, breakable=True, color=BLACK),
	# stone and earth
	cube("stone", "default_stone"),
	cube("cobblestone", "default_cobble"),
	cube("mossy_cobblestone", "default_mossycobble"),
	cube("stone_bricks", "default_stone_brick"),
	cube("mossy_stone_bricks", "mcl_core_stonebrick_mossy"),
	cube("cracked_stone_bricks", "mcl_core_stonebrick_cracked"),
	cube("chiseled_stone_bricks", "mcl_core_stonebrick_carved"),
	cube("granite", "mcl_core_granite"),
	cube("polished_granite", "mcl_core_granite_smooth"),
	cube("diorite", "mcl_core_diorite"),
	cube("polished_diorite", "mcl_core_diorite_smooth"),
	cube("andesite", "mcl_core_andesite"),
	cube("polished_andesite", "mcl_core_andesite_smooth"),
	column("deepslate", "mcl_deepslate_top", "mcl_deepslate"),
	cube("cobbled_deepslate", "mcl_cobbled_deepslate"),
	cube("bricks", "default_brick"),
	cube("dirt", "default_dirt"),
	cube("coarse_dirt", "mcl_core_coarse_dirt"),
	Block("grass_block", "mcl_core_grass_block_top", "mcrl_grass_block_side", "default_dirt", tint_faces=TOP,
		tint=GRASS),
	column("podzol", "mcl_core_dirt_podzol_top", "mcl_core_dirt_podzol_side", "default_dirt"),
	column("mycelium", "mcl_core_mycelium_top", "mcl_core_mycelium_side", "default_dirt"),
	cube("sand", "default_sand"),
	cube("red_sand", "mcl_core_red_sand"),
	cube("gravel", "default_gravel"),
	cube("clay", "default_clay"),
	column("sandstone", "mcl_core_sandstone_top", "mcl_core_sandstone_normal", "mcl_core_sandstone_bottom"),
	column("chiseled_sandstone", "mcl_core_sandstone_top", "mcl_core_sandstone_carved"),
	column("red_sandstone", "mcl_core_red_sandstone_top", "mcl_core_red_sandstone_normal",
		"mcl_core_red_sandstone_bottom"),
	# wood
	column("oak_log", "default_tree_top", "default_tree"),
	column("birch_log", "mcl_core_log_birch_top", "mcl_core_log_birch"),
	column("spruce_log", "mcl_core_log_spruce_top", "mcl_core_log_spruce"),
	column("jungle_log", "default_jungletree_top", "default_jungletree"),
	column("acacia_log", "default_acacia_tree_top", "default_acacia_tree"),
	column("dark_oak_log", "mcl_core_log_big_oak_top", "mcl_core_log_big_oak"),
	cube("oak_planks", "default_wood"),
	cube("birch_planks", "mcl_core_planks_birch"),
	cube("spruce_planks", "mcl_core_planks_spruce"),
	cube("jungle_planks", "default_junglewood"),
	cube("acacia_planks", "default_acacia_wood"),
	cube("dark_oak_planks", "mcl_core_planks_big_oak"),
	# Leaves get Minecraft's foliage tints; VoxeLibre's birch, spruce, acacia and dark oak leaves are already
	# coloured, so the build keeps only the oak and jungle tints (see Block).
	cube("oak_leaves", "default_leaves", tint_faces=ALL, tint=FOLIAGE),
	cube("birch_leaves", "mcl_core_leaves_birch", tint_faces=ALL, tint=BIRCH_FOLIAGE),
	cube("spruce_leaves", "mcl_core_leaves_spruce", tint_faces=ALL, tint=SPRUCE_FOLIAGE),
	cube("jungle_leaves", "default_jungleleaves", tint_faces=ALL, tint=FOLIAGE),
	cube("acacia_leaves", "default_acacia_leaves", tint_faces=ALL, tint=FOLIAGE),
	cube("dark_oak_leaves", "mcl_core_leaves_big_oak", tint_faces=ALL, tint=FOLIAGE),
	# building and utility (one texture for all four sides: furnace shows its front so it reads as a furnace)
	cube("glass", "default_glass"),
	column("bookshelf", "default_wood", "default_bookshelf"),
	column("crafting_table", "crafting_workbench_top", "crafting_workbench_front", "default_wood"),
	column("furnace", "default_furnace_top", "default_furnace_front", "default_furnace_bottom"),
	column("tnt", "default_tnt_top", "default_tnt_side", "default_tnt_bottom"),
	cube("obsidian", "default_obsidian"),
	cube("crying_obsidian", "mcrl_crying_obsidian"),
	cube("bedrock", "mcl_core_bedrock", breakable=False),
	# ores and mineral blocks
	cube("coal_ore", "mcl_core_coal_ore"),
	cube("iron_ore", "mcl_core_iron_ore"),
	cube("gold_ore", "mcl_core_gold_ore"),
	cube("diamond_ore", "mcl_core_diamond_ore"),
	cube("redstone_ore", "mcl_core_redstone_ore"),
	cube("lapis_ore", "mcl_core_lapis_ore"),
	cube("emerald_ore", "mcl_core_emerald_ore"),
	cube("copper_ore", "mcrl_copper_ore"),
	cube("coal_block", "default_coal_block"),
	cube("iron_block", "default_steel_block"),
	cube("gold_block", "default_gold_block"),
	cube("diamond_block", "default_diamond_block"),
	cube("redstone_block", "redstone_redstone_block"),
	cube("lapis_block", "mcl_core_lapis_block"),
	cube("emerald_block", "mcl_core_emerald_block"),
	cube("copper_block", "mcl_copper_block"),
]

BLOCKS += [cube(colour + "_wool", wool) for colour, wool, _ in COLOURS]
BLOCKS += [cube(colour + "_concrete", "mcl_colorblocks_concrete_" + suffix) for colour, _, suffix in COLOURS]
BLOCKS.append(cube("terracotta", "hardened_clay"))
BLOCKS += [cube(colour + "_terracotta", "hardened_clay_stained_" + suffix) for colour, _, suffix in COLOURS]

BLOCKS += [
	# nether, end, ocean
	cube("glowstone", "mcl_nether_glowstone"),
	cube("netherrack", "mcl_nether_netherrack"),
	cube("nether_bricks", "mcl_nether_nether_brick"),
	cube("soul_sand", "mcl_nether_soul_sand"),
	column("quartz_block", "mcl_nether_quartz_block_top", "mcl_nether_quartz_block_side",
		"mcl_nether_quartz_block_bottom"),
	cube("end_stone", "mcl_end_end_stone"),
	cube("purpur_block", "mcl_end_purpur_block"),
	cube("prismarine", "mcl_ocean_prismarine_anim"),
	cube("prismarine_bricks", "mcl_ocean_prismarine_bricks"),
	cube("sea_lantern", "mcl_ocean_sea_lantern"),
	# snow and ice
	cube("snow_block", "default_snow"),
	column("snowy_grass_block", "default_snow", "mcl_core_grass_side_snowed", "default_dirt"),
	cube("ice", "default_ice"),
	cube("packed_ice", "mcl_core_ice_packed"),
	# plants and others
	column("pumpkin", "farming_pumpkin_top", "farming_pumpkin_side"),
	column("melon", "farming_melon_top", "farming_melon_side"),
	column("hay_block", "mcl_farming_hayblock_top", "mcl_farming_hayblock_side"),
	cube("sponge", "mcl_sponges_sponge"),
	cube("slime_block", "mcl_core_slime"),
	column("bone_block", "mcl_core_bone_block_top", "mcl_core_bone_block_side"),
	cube("amethyst_block", "mcl_amethyst_amethyst_block"),
	cube("redstone_lamp", "jeija_lightstone_gray_off"),
]
