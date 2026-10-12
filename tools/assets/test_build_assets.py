"""Tests for build_assets.py on synthetic textures.

Run: python3 -m unittest discover -s tools/assets -p 'test_*.py'
"""

import contextlib
import io
import os
import re
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocks  # noqa: E402
import build_assets as ba  # noqa: E402
from blocks import Block, Generated, column, cube  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def solid(rgb, width=16, height=16, alpha=255):
	image = np.zeros((height, width, 4), dtype=np.uint8)
	image[..., :3] = rgb
	image[..., 3] = alpha
	return image


def save(directory, name, rgba):
	path = os.path.join(directory, name + ".png")
	Image.fromarray(np.ascontiguousarray(rgba)).save(path)
	return path


def unique_skin(width=64, height=32):
	"""Every texel a different opaque colour, so copies can be traced back to their source."""
	image = np.zeros((height, width, 4), dtype=np.uint8)
	for y in range(height):
		for x in range(width):
			image[y, x] = (x * 4 % 256, y * 4 % 256, (x * 7 + y * 13) % 256, 255)
	return image


def quiet(*_args, **_kwargs):
	pass


def parse_blocks_file(text):
	"""Data lines of mcrl_blocks.txt as field lists (comments and blank lines skipped)."""
	rows = []
	for line in text.splitlines():
		stripped = line.strip()
		if not stripped or stripped.startswith("#"):
			continue
		rows.append(stripped.split())
	return rows


class TempDirTest(unittest.TestCase):
	def setUp(self):
		self._tmp = tempfile.TemporaryDirectory()
		self.dir = self._tmp.name
		self.textures = os.path.join(self.dir, "textures")
		os.makedirs(self.textures)

	def tearDown(self):
		self._tmp.cleanup()

	def texture(self, name, rgba):
		return save(self.textures, name, rgba)


class SlotOriginTest(unittest.TestCase):
	def test_slots_fill_rows_of_64(self):
		self.assertEqual(ba.slot_origin(0), (0, 0))
		self.assertEqual(ba.slot_origin(1), (16, 0))
		self.assertEqual(ba.slot_origin(63), (1008, 0))
		self.assertEqual(ba.slot_origin(64), (0, 16))
		self.assertEqual(ba.slot_origin(65), (16, 16))
		self.assertEqual(ba.slot_origin(4095), (1008, 1008))

	def test_atlas_constants_match_the_shader(self):
		self.assertEqual(ba.ATLAS_SIZE, 1024)
		self.assertEqual(ba.MAX_SLOTS, 4096)


class AtlasTest(TempDirTest):
	def test_sources_are_placed_by_sorted_file_name(self):
		self.texture("b", solid((0, 255, 0)))
		self.texture("c", solid((0, 0, 255)))
		self.texture("a", solid((255, 0, 0)))
		atlas = ba.build_atlas(self.textures, [])
		self.assertEqual(atlas.names, ["a", "b", "c"])
		self.assertEqual(atlas.slots, {"a": 0, "b": 1, "c": 2})
		self.assertEqual(atlas.pixels.shape, (1024, 1024, 4))
		self.assertEqual(atlas.pixels.dtype, np.uint8)
		for slot, rgb in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
			x, y = ba.slot_origin(slot)
			np.testing.assert_array_equal(atlas.pixels[y:y + 16, x:x + 16], solid(rgb))
		# everything after the last slot stays transparent black
		self.assertFalse(atlas.pixels[:16, 48:].any())
		self.assertFalse(atlas.pixels[16:].any())

	def test_texture_lands_on_its_slot_in_a_later_row(self):
		for i in range(70):
			self.texture(f"t{i:03d}", solid((i, 255 - i, 7)))
		atlas = ba.build_atlas(self.textures, [])
		self.assertEqual(atlas.slot("t065"), 65)
		np.testing.assert_array_equal(atlas.pixels[16:32, 16:32], solid((65, 190, 7)))
		np.testing.assert_array_equal(atlas.pixels[0:16, 1008:1024], solid((63, 192, 7)))

	def test_strip_contributes_its_first_frame_only(self):
		strip = np.concatenate([solid((255, 0, 0)), solid((0, 255, 0)), solid((0, 0, 255))], axis=0)
		strip[3, 5] = (1, 2, 3, 4)
		self.texture("anim", strip)
		self.texture("zz", solid((9, 9, 9)))
		atlas = ba.build_atlas(self.textures, [])
		self.assertEqual(atlas.names, ["anim", "zz"])
		np.testing.assert_array_equal(atlas.pixels[:16, :16], strip[:16])
		np.testing.assert_array_equal(atlas.frames["anim"], strip[:16])
		np.testing.assert_array_equal(atlas.pixels[:16, 16:32], solid((9, 9, 9)))
		self.assertFalse(atlas.pixels[16:].any())

	def test_other_sizes_and_non_png_entries_are_skipped(self):
		self.texture("big", solid((1, 1, 1), 32, 32))
		self.texture("odd_height", solid((1, 1, 1), 16, 20))
		self.texture("short", solid((1, 1, 1), 16, 8))
		self.texture("narrow", solid((1, 1, 1), 8, 16))
		self.texture("ok", solid((1, 1, 1)))
		with open(os.path.join(self.textures, "notes.txt"), "w") as f:
			f.write("not a texture")
		Image.fromarray(solid((1, 1, 1))[..., :3]).save(os.path.join(self.textures, "photo.jpg"))
		os.makedirs(os.path.join(self.textures, "folder.png"))
		atlas = ba.build_atlas(self.textures, [])
		self.assertEqual(atlas.names, ["ok"])
		self.assertEqual(atlas.skipped_size, 4)
		self.assertEqual(atlas.skipped_other, 3)

	def test_palette_and_grey_images_are_expanded_to_rgba(self):
		palette = Image.new("P", (16, 16), 0)
		palette.putpalette([10, 20, 30, 200, 100, 50] + [0] * 762)
		palette.putpixel((1, 0), 1)
		palette.info["transparency"] = 1
		palette.save(os.path.join(self.textures, "pal.png"), transparency=1)
		Image.new("LA", (16, 16), (90, 255)).save(os.path.join(self.textures, "grey.png"))
		atlas = ba.build_atlas(self.textures, [])
		self.assertEqual(atlas.names, ["grey", "pal"])
		self.assertEqual(tuple(atlas.frames["grey"][0, 0]), (90, 90, 90, 255))
		self.assertEqual(tuple(atlas.frames["pal"][0, 0]), (10, 20, 30, 255))
		self.assertEqual(atlas.frames["pal"][0, 1, 3], 0)

	def test_unreadable_png_is_an_error(self):
		with open(os.path.join(self.textures, "broken.png"), "wb") as f:
			f.write(b"\x89PNG not really")
		with self.assertRaises(ba.AssetError) as ctx:
			ba.build_atlas(self.textures, [])
		self.assertIn("broken.png", str(ctx.exception))

	def test_missing_texture_folder_is_an_error(self):
		with self.assertRaises(ba.AssetError):
			ba.build_atlas(os.path.join(self.dir, "nope"), [])

	def test_atlas_text_lists_slot_and_name(self):
		self.texture("b", solid((1, 1, 1)))
		self.texture("a", solid((1, 1, 1)))
		self.texture("base", solid((50, 50, 50)))
		self.texture("over", solid((50, 50, 50), alpha=0))
		atlas = ba.build_atlas(self.textures, [Generated("mcrl_gen", "base", "over")])
		self.assertEqual(ba.format_atlas_names(atlas), "0 mcrl_gen\n1 a\n2 b\n3 base\n4 over\n")


class GeneratedTest(TempDirTest):
	def test_generated_textures_come_first_in_list_order(self):
		self.texture("a", solid((10, 10, 10)))
		self.texture("base", solid((100, 50, 25)))
		self.texture("over", solid((200, 200, 200), alpha=0))
		recipes = [Generated("mcrl_two", "base", "over"), Generated("mcrl_one", "a", "over")]
		atlas = ba.build_atlas(self.textures, recipes)
		self.assertEqual(atlas.names, ["mcrl_two", "mcrl_one", "a", "base", "over"])
		self.assertEqual(atlas.generated, 2)
		np.testing.assert_array_equal(atlas.pixels[:16, :16], solid((100, 50, 25)))
		np.testing.assert_array_equal(atlas.pixels[:16, 16:32], solid((10, 10, 10)))
		np.testing.assert_array_equal(atlas.pixels[:16, 32:48], solid((10, 10, 10)))

	def test_tinted_overlay_is_composited_over_the_base(self):
		base = solid((120, 80, 40))
		overlay = solid((200, 100, 50), alpha=0)
		overlay[0:4] = (200, 100, 50, 255)  # opaque band
		overlay[4, 0] = (200, 100, 50, 128)  # half see-through texel
		self.texture("dirt", base)
		self.texture("grass", overlay)
		tint = (0x91, 0xBD, 0x59)
		atlas = ba.build_atlas(self.textures, [Generated("mcrl_side", "dirt", "grass", tint)])
		side = atlas.frames["mcrl_side"]
		tinted = [(c * t + 127) // 255 for c, t in zip((200, 100, 50), tint)]
		np.testing.assert_array_equal(side[0:4], solid(tinted)[0:4])
		blended = [(o * 128 + b * 127 + 127) // 255 for o, b in zip(tinted, (120, 80, 40))]
		self.assertEqual(tuple(side[4, 0]), tuple(blended) + (255,))
		np.testing.assert_array_equal(side[5:], base[5:])
		np.testing.assert_array_equal(side[4, 1:], base[4, 1:])

	def test_untinted_overlay_keeps_its_colours(self):
		overlay = solid((0, 0, 0), alpha=0)
		overlay[2, 3] = (180, 90, 40, 255)
		self.texture("stone", solid((128, 128, 128)))
		self.texture("ore", overlay)
		atlas = ba.build_atlas(self.textures, [Generated("mcrl_ore", "stone", "ore")])
		self.assertEqual(tuple(atlas.frames["mcrl_ore"][2, 3]), (180, 90, 40, 255))
		self.assertEqual(tuple(atlas.frames["mcrl_ore"][0, 0]), (128, 128, 128, 255))

	def test_generated_from_strips_uses_first_frames(self):
		self.texture("base", np.concatenate([solid((1, 2, 3)), solid((9, 9, 9))]))
		self.texture("over", np.concatenate([solid((0, 0, 0), alpha=0), solid((7, 7, 7))]))
		atlas = ba.build_atlas(self.textures, [Generated("mcrl_x", "base", "over")])
		np.testing.assert_array_equal(atlas.frames["mcrl_x"], solid((1, 2, 3)))

	def test_missing_recipe_source_is_an_error(self):
		self.texture("base", solid((1, 1, 1)))
		with self.assertRaises(ba.AssetError) as ctx:
			ba.build_atlas(self.textures, [Generated("mcrl_x", "base", "nothing")])
		self.assertIn("nothing.png", str(ctx.exception))

	def test_recipe_source_of_wrong_size_is_an_error(self):
		self.texture("base", solid((1, 1, 1), 32, 32))
		self.texture("over", solid((1, 1, 1)))
		with self.assertRaises(ba.AssetError):
			ba.build_atlas(self.textures, [Generated("mcrl_x", "base", "over")])

	def test_generated_names_need_the_mcrl_prefix(self):
		self.texture("base", solid((1, 1, 1)))
		with self.assertRaises(ba.AssetError):
			ba.build_atlas(self.textures, [Generated("side", "base", "base")])

	def test_name_clash_with_a_source_texture_is_an_error(self):
		self.texture("base", solid((1, 1, 1)))
		self.texture("mcrl_x", solid((1, 1, 1)))
		with self.assertRaises(ba.AssetError) as ctx:
			ba.build_atlas(self.textures, [Generated("mcrl_x", "base", "base")])
		self.assertIn("mcrl_x", str(ctx.exception))


class OverflowTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls._tmp = tempfile.TemporaryDirectory()
		cls.textures = cls._tmp.name
		for i in range(ba.MAX_SLOTS):
			save(cls.textures, f"t{i:04d}", solid((i % 256, i // 256, 77)))

	@classmethod
	def tearDownClass(cls):
		cls._tmp.cleanup()

	def test_exactly_4096_textures_fill_the_atlas(self):
		atlas = ba.build_atlas(self.textures, [])
		self.assertEqual(len(atlas.names), 4096)
		self.assertEqual(atlas.names[-1], "t4095")
		np.testing.assert_array_equal(atlas.pixels[1008:, 1008:], solid((4095 % 256, 4095 // 256, 77)))

	def test_one_more_slot_is_an_error(self):
		with self.assertRaises(ba.AssetError) as ctx:
			ba.build_atlas(self.textures, [Generated("mcrl_extra", "t0000", "t0001")])
		self.assertIn("4097", str(ctx.exception))
		self.assertIn("4096", str(ctx.exception))


class ColourTest(unittest.TestCase):
	def test_tint_multiplies_and_keeps_alpha(self):
		image = solid((255, 128, 0), alpha=77)
		out = ba.tint_rgba(image, (128, 255, 50))
		self.assertEqual(tuple(out[0, 0]), (128, 128, 0, 77))

	def test_saturation_of_grey_and_coloured(self):
		self.assertEqual(ba.saturation(solid((90, 90, 90))), 0.0)
		self.assertAlmostEqual(ba.saturation(solid((200, 100, 100))), 0.5)
		self.assertEqual(ba.saturation(solid((0, 0, 0))), 0.0)

	def test_saturation_ignores_see_through_texels(self):
		image = solid((100, 100, 100))
		image[:8] = (255, 0, 0, 127)
		self.assertEqual(ba.saturation(image), 0.0)
		self.assertEqual(ba.saturation(solid((255, 0, 0), alpha=0)), 0.0)

	def test_average_colour_of_opaque_texels_rounded_half_up(self):
		image = solid((1, 10, 200))
		image[:8] = (2, 11, 201, 128)  # alpha 128 is drawn by the shader
		image[0, 0] = (255, 255, 255, 127)  # alpha 127 is not
		# 127 texels of the second colour, 128 of the first
		self.assertEqual(ba.average_color(image, "x"), (1, 10, 200))
		image[0, 0] = (2, 11, 201, 255)
		self.assertEqual(ba.average_color(image, "x"), (2, 11, 201))

	def test_average_colour_of_nothing_is_an_error(self):
		with self.assertRaises(ba.AssetError):
			ba.average_color(solid((5, 5, 5), alpha=0), "empty")

	def test_hex_colour(self):
		self.assertEqual(ba.hex_color((0, 171, 255)), "#00abff")


class SkinTest(unittest.TestCase):
	def test_legacy_top_half_is_copied_unchanged(self):
		legacy = unique_skin()
		skin = ba.convert_skin(legacy)
		self.assertEqual(skin.shape, (64, 64, 4))
		np.testing.assert_array_equal(skin[:32], legacy)

	def test_left_limbs_are_mirrored_copies_of_the_right(self):
		legacy = unique_skin()
		skin = ba.convert_skin(legacy)
		self.assertEqual(len(ba.LEGACY_LIMB_COPIES), 12)
		for sx, sy, w, h, dx, dy in ba.LEGACY_LIMB_COPIES:
			for j in range(h):
				for i in range(w):
					self.assertEqual(tuple(skin[dy + j, dx + i]), tuple(legacy[sy + j, sx + w - 1 - i]),
						f"copy ({sx},{sy}) -> ({dx},{dy}) texel ({i},{j})")

	def test_spot_checks_of_the_mirror(self):
		legacy = unique_skin()
		skin = ba.convert_skin(legacy)
		# left leg top (20,48) is the right leg top (4..7,16) mirrored: its first column is source column 7
		self.assertEqual(tuple(skin[48, 20]), tuple(legacy[16, 7]))
		self.assertEqual(tuple(skin[48, 23]), tuple(legacy[16, 4]))
		# left leg's outer side (24,52) is the right leg's outer side (0..3,20) mirrored
		self.assertEqual(tuple(skin[52, 24]), tuple(legacy[20, 3]))
		self.assertEqual(tuple(skin[63, 27]), tuple(legacy[31, 0]))
		# left arm front (36,52) from the right arm front (44..47,20)
		self.assertEqual(tuple(skin[52, 36]), tuple(legacy[20, 47]))
		# left arm back (44,52) from the right arm back (52..55,20)
		self.assertEqual(tuple(skin[60, 47]), tuple(legacy[28, 52]))

	def test_rest_of_the_bottom_half_is_transparent(self):
		skin = ba.convert_skin(unique_skin())
		written = np.zeros((64, 64), dtype=bool)
		for _sx, _sy, w, h, dx, dy in ba.LEGACY_LIMB_COPIES:
			written[dy:dy + h, dx:dx + w] = True
		self.assertFalse(written[:32].any())
		self.assertFalse(skin[32:][~written[32:]].any())
		self.assertTrue((skin[32:][written[32:]][:, 3] == 255).all())

	def test_modern_skin_is_copied_as_is(self):
		modern = unique_skin(64, 64)
		modern[40, 40] = (1, 2, 3, 0)
		skin = ba.convert_skin(modern)
		np.testing.assert_array_equal(skin, modern)
		self.assertIsNot(skin, modern)

	def test_other_skin_sizes_are_errors(self):
		for width, height in ((32, 32), (64, 48), (128, 64)):
			with self.assertRaises(ba.AssetError):
				ba.convert_skin(unique_skin(width, height))

	def test_box_faces_follow_minecraft_layout(self):
		self.assertEqual(ba.steve_box_faces((0, 0), 8, 8, 8), {
			"top": (8, 0, 8, 8), "bottom": (16, 0, 8, 8), "right": (0, 8, 8, 8),
			"front": (8, 8, 8, 8), "left": (16, 8, 8, 8), "back": (24, 8, 8, 8)})
		self.assertEqual(ba.steve_box_faces((16, 16), 8, 12, 4), {
			"top": (20, 16, 8, 4), "bottom": (28, 16, 8, 4), "right": (16, 20, 4, 12),
			"front": (20, 20, 8, 12), "left": (28, 20, 4, 12), "back": (32, 20, 8, 12)})

	def test_shader_boxes_match_the_effect(self):
		boxes = {name: (uv0, w, h, d) for name, uv0, w, h, d in ba.STEVE_BOXES}
		self.assertEqual(boxes["left arm"], ((32, 48), 4, 12, 4))
		self.assertEqual(boxes["left leg"], ((16, 48), 4, 12, 4))
		self.assertEqual(boxes["right arm"], ((40, 16), 4, 12, 4))
		self.assertEqual(boxes["right leg"], ((0, 16), 4, 12, 4))

	def test_left_limb_faces_are_exactly_the_converted_regions(self):
		sampled = np.zeros((64, 64), dtype=bool)
		for name, uv0, w, h, d in ba.STEVE_BOXES:
			if name.startswith("left"):
				for x, y, fw, fh in ba.steve_box_faces(uv0, w, h, d).values():
					sampled[y:y + fh, x:x + fw] = True
		written = np.zeros((64, 64), dtype=bool)
		for _sx, _sy, w, h, dx, dy in ba.LEGACY_LIMB_COPIES:
			written[dy:dy + h, dx:dx + w] = True
		np.testing.assert_array_equal(sampled, written)

	def test_converted_opaque_skin_has_no_holes(self):
		self.assertEqual(ba.check_skin(ba.convert_skin(unique_skin())), [])

	def test_holes_in_the_right_arm_show_on_both_arms(self):
		legacy = unique_skin()
		legacy[20:32, 44:48, 3] = 0  # right arm front
		problems = ba.check_skin(ba.convert_skin(legacy))
		self.assertEqual(problems, ["right arm front: 48 of 48 texels see-through",
			"left arm front: 48 of 48 texels see-through"])

	def test_half_alpha_counts_as_opaque(self):
		legacy = unique_skin()
		legacy[..., 3] = 128
		self.assertEqual(ba.check_skin(ba.convert_skin(legacy)), [])
		legacy[0, 8, 3] = 127  # head top
		self.assertEqual(ba.check_skin(ba.convert_skin(legacy)), ["head top: 1 of 64 texels see-through"])

	def test_modern_skin_missing_left_leg_is_reported(self):
		modern = unique_skin(64, 64)
		modern[48:64, 16:32] = 0
		problems = ba.check_skin(modern)
		self.assertEqual(len(problems), 6)
		self.assertTrue(all(p.startswith("left leg ") for p in problems))


class BlockResolveTest(TempDirTest):
	def setUp(self):
		super().setUp()
		self.texture("grey", solid((128, 128, 128)))
		self.texture("red", solid((200, 30, 30)))
		self.texture("top", solid((10, 20, 30)))
		self.texture("side", solid((40, 50, 60)))
		self.texture("bottom", solid((70, 80, 90)))
		half = solid((60, 60, 60))
		half[8:] = (255, 0, 0, 0)
		self.texture("half", half)
		self.atlas = ba.build_atlas(self.textures, [])

	def resolve(self, block_list):
		return ba.resolve_blocks(block_list, self.atlas)

	def test_slots_flags_and_ids(self):
		resolved = self.resolve([
			Block("barrier", None, None, None, visible=False, color=(0, 0, 0)),
			column("log", "top", "side", "bottom"),
			cube("hard", "grey", breakable=False, solid=False),
		])
		self.assertEqual([b.id for b in resolved], [1, 2, 3])
		barrier, log, hard = resolved
		self.assertEqual((barrier.top, barrier.side, barrier.bottom), (-1, -1, -1))
		self.assertEqual((barrier.solid, barrier.visible, barrier.breakable), (True, False, True))
		self.assertEqual(barrier.color, (0, 0, 0))
		self.assertEqual((log.top, log.side, log.bottom),
			(self.atlas.slot("top"), self.atlas.slot("side"), self.atlas.slot("bottom")))
		self.assertEqual(log.color, (40, 50, 60))
		self.assertEqual((hard.solid, hard.visible, hard.breakable), (False, True, False))

	def test_column_bottom_defaults_to_top(self):
		block = column("x", "top", "side")
		self.assertEqual((block.top, block.side, block.bottom), ("top", "side", "top"))

	def test_greyscale_faces_keep_their_tint(self):
		(b,) = self.resolve([cube("leaves", "grey", tint_faces=blocks.ALL, tint=(0x77, 0xAB, 0x2F))])
		self.assertEqual(b.tint_faces, 7)
		self.assertEqual(b.tint, (0x77, 0xAB, 0x2F))
		self.assertEqual(len(b.tints_kept), 3)
		self.assertEqual(b.tints_dropped, [])

	def test_coloured_faces_are_never_tinted(self):
		(b,) = self.resolve([cube("leaves", "red", tint_faces=blocks.ALL, tint=(0x77, 0xAB, 0x2F))])
		self.assertEqual(b.tint_faces, 0)
		self.assertEqual(b.tint, (255, 255, 255))
		self.assertEqual([face for face, _, _ in b.tints_dropped], ["top", "side", "bottom"])

	def test_tint_decided_per_face(self):
		(b,) = self.resolve([Block("grass", "grey", "red", "grey", tint_faces=blocks.TOP | blocks.SIDE,
			tint=(1, 2, 3))])
		self.assertEqual(b.tint_faces, blocks.TOP)
		self.assertEqual(b.tint, (1, 2, 3))

	def test_tinted_face_without_texture_keeps_the_bit(self):
		(b,) = self.resolve([Block("flat", None, "grey", None, tint_faces=blocks.TOP, tint=(9, 9, 9))])
		self.assertEqual(b.tint_faces, blocks.TOP)

	def test_colour_is_the_untinted_side_average(self):
		(b,) = self.resolve([Block("x", "red", "half", "red", tint_faces=blocks.ALL, tint=(10, 10, 10))])
		self.assertEqual(b.color, (60, 60, 60))

	def test_explicit_colour_wins(self):
		(b,) = self.resolve([cube("x", "red", color=(1, 2, 3))])
		self.assertEqual(b.color, (1, 2, 3))

	def test_missing_textures_are_all_named(self):
		with self.assertRaises(ba.AssetError) as ctx:
			self.resolve([cube("a", "nope"), column("b", "grey", "gone")])
		message = str(ctx.exception)
		self.assertIn("nope (a top)", message)
		self.assertIn("gone (b side)", message)
		self.assertNotIn("grey", message)

	def test_texture_of_wrong_size_is_not_in_the_atlas(self):
		self.texture("wide", solid((1, 1, 1), 32, 16))
		atlas = ba.build_atlas(self.textures, [])
		with self.assertRaises(ba.AssetError) as ctx:
			ba.resolve_blocks([cube("w", "wide")], atlas)
		self.assertIn("wide", str(ctx.exception))

	def test_bad_and_duplicate_names_are_errors(self):
		for bad in ("Stone", "two words", "", "hash#", "dash-name"):
			with self.assertRaises(ba.AssetError, msg=bad):
				self.resolve([cube(bad, "grey")])
		with self.assertRaises(ba.AssetError):
			self.resolve([cube("a", "grey"), cube("a", "red")])

	def test_bad_tint_values_are_errors(self):
		with self.assertRaises(ba.AssetError):
			self.resolve([cube("a", "grey", tint_faces=8)])
		with self.assertRaises(ba.AssetError):
			self.resolve([cube("a", "grey", tint=(256, 0, 0))])
		with self.assertRaises(ba.AssetError):
			self.resolve([cube("a", "grey", color=(1, 2))])

	def test_no_side_and_no_colour_is_an_error(self):
		with self.assertRaises(ba.AssetError):
			self.resolve([Block("x", "grey", None, "grey")])

	def test_see_through_side_is_an_error(self):
		self.texture("clear", solid((1, 1, 1), alpha=0))
		atlas = ba.build_atlas(self.textures, [])
		with self.assertRaises(ba.AssetError):
			ba.resolve_blocks([cube("x", "clear")], atlas)


class BlocksFileTest(TempDirTest):
	def test_format(self):
		self.texture("grey", solid((128, 128, 128)))
		self.texture("red", solid((200, 30, 30)))
		atlas = ba.build_atlas(self.textures, [])
		resolved = ba.resolve_blocks([
			Block("barrier", None, None, None, visible=False, color=(0, 0, 0)),
			cube("leaves", "grey", tint_faces=blocks.ALL, tint=(0x77, 0xAB, 0x2F)),
			column("thing", "red", "grey", breakable=False),
		], atlas)
		text = ba.format_blocks(resolved)
		self.assertTrue(text.startswith("# mcrl blocks v1\n"))
		self.assertTrue(text.endswith("\n"))
		self.assertNotIn("\r", text)
		self.assertEqual(parse_blocks_file(text), [
			"1 barrier -1 -1 -1 1 0 1 0 #ffffff #000000".split(),
			"2 leaves 0 0 0 1 1 1 7 #77ab2f #808080".split(),
			"3 thing 1 0 1 1 1 0 0 #ffffff #808080".split(),
		])


def check_blocks_file(test, text, atlas_names):
	"""The checks BlockRegistry::parse makes, plus slots that exist in the atlas."""
	rows = parse_blocks_file(text)
	test.assertGreater(len(rows), 0)
	hex_colour = re.compile(r"^#[0-9a-f]{6}$")
	names = set()
	for expected_id, fields in enumerate(rows, start=1):
		test.assertEqual(len(fields), 11, fields)
		test.assertEqual(fields[0], str(expected_id))
		test.assertRegex(fields[1], r"^[a-z0-9_]+$")
		test.assertNotIn(fields[1], names)
		names.add(fields[1])
		for slot in fields[2:5]:
			test.assertRegex(slot, r"^-?\d+$")
			test.assertTrue(-1 <= int(slot) < min(4096, len(atlas_names)), fields)
		for flag in fields[5:8]:
			test.assertIn(flag, ("0", "1"))
		test.assertIn(fields[8], [str(i) for i in range(8)])
		test.assertRegex(fields[9], hex_colour)
		test.assertRegex(fields[10], hex_colour)
	return rows


class RealBlockListTest(TempDirTest):
	"""blocks.py, built against a synthetic folder holding exactly the textures it names."""

	def referenced(self):
		names = set()
		for recipe in blocks.GENERATED:
			names.update((recipe.base, recipe.overlay))
		generated = {recipe.name for recipe in blocks.GENERATED}
		for block in blocks.BLOCKS:
			names.update(t for t in (block.top, block.side, block.bottom) if t is not None and t not in generated)
		return sorted(names)

	def test_list_basics(self):
		self.assertEqual(blocks.BLOCKS[0], Block("barrier", None, None, None, solid=True, visible=False,
			breakable=True, color=(0, 0, 0)))
		names = [b.name for b in blocks.BLOCKS]
		self.assertEqual(len(names), len(set(names)))
		self.assertTrue(all(b.visible for b in blocks.BLOCKS[1:]))
		by_name = {b.name: b for b in blocks.BLOCKS}
		self.assertFalse(by_name["bedrock"].breakable)
		self.assertEqual(sum(1 for n in names if n.endswith("_wool")), 16)
		self.assertEqual(sum(1 for n in names if n.endswith("_concrete")), 16)
		self.assertEqual(sum(1 for n in names if n.endswith("_terracotta")), 16)
		grass = by_name["grass_block"]
		self.assertEqual((grass.top, grass.side, grass.bottom),
			("mcl_core_grass_block_top", "mcrl_grass_block_side", "default_dirt"))
		self.assertEqual(grass.tint_faces, blocks.TOP)
		self.assertEqual(blocks.GENERATED[0].name, "mcrl_grass_block_side")
		self.assertEqual(blocks.GENERATED[0].tint, blocks.GRASS)
		self.assertTrue(all(r.name.startswith("mcrl_") for r in blocks.GENERATED))

	def test_every_referenced_texture_resolves_to_its_slot(self):
		for name in self.referenced():
			self.texture(name, solid((100, 100, 100)))
		save(self.textures, "character", unique_skin())
		out = os.path.join(self.dir, "out")
		ba.build(self.textures, out, log=quiet)
		with open(os.path.join(out, "mcrl_atlas.txt"), encoding="utf-8") as f:
			atlas_names = [line.split(" ", 1)[1] for line in f.read().splitlines()]
		with open(os.path.join(out, "mcrl_blocks.txt"), encoding="utf-8") as f:
			rows = check_blocks_file(self, f.read(), atlas_names)
		self.assertEqual(len(rows), len(blocks.BLOCKS))
		self.assertEqual(atlas_names[:len(blocks.GENERATED)], [r.name for r in blocks.GENERATED])
		for block, fields in zip(blocks.BLOCKS, rows):
			self.assertEqual(fields[1], block.name)
			for texture, slot in zip((block.top, block.side, block.bottom), fields[2:5]):
				if texture is None:
					self.assertEqual(slot, "-1")
				else:
					self.assertEqual(atlas_names[int(slot)], texture, block.name)

	def test_a_missing_file_fails_loudly(self):
		names = self.referenced()
		for name in names[1:]:
			self.texture(name, solid((100, 100, 100)))
		save(self.textures, "character", unique_skin())
		out = os.path.join(self.dir, "out")
		with self.assertRaises(ba.AssetError) as ctx:
			ba.build(self.textures, out, log=quiet)
		self.assertIn(names[0], str(ctx.exception))
		self.assertFalse(os.path.exists(out))


@unittest.skipUnless(os.path.isfile(os.path.join(REPO, "assets", "mcrl_blocks.txt")), "assets/ not built")
class CommittedAssetsTest(unittest.TestCase):
	"""The built files in assets/ match blocks.py and the shader's sizes (rebuild after editing blocks.py)."""

	def read(self, name):
		with open(os.path.join(REPO, "assets", name), encoding="utf-8") as f:
			return f.read()

	def test_blocks_file_matches_the_block_list(self):
		atlas_names = [line.split(" ", 1)[1] for line in self.read("mcrl_atlas.txt").splitlines()]
		self.assertEqual(atlas_names[:len(blocks.GENERATED)], [r.name for r in blocks.GENERATED])
		rows = check_blocks_file(self, self.read("mcrl_blocks.txt"), atlas_names)
		self.assertEqual([fields[1] for fields in rows], [b.name for b in blocks.BLOCKS])
		for block, fields in zip(blocks.BLOCKS, rows):
			for texture, slot in zip((block.top, block.side, block.bottom), fields[2:5]):
				self.assertEqual(atlas_names[int(slot)] if texture is not None else None, texture, block.name)

	def test_atlas_text_numbers_every_slot(self):
		lines = self.read("mcrl_atlas.txt").splitlines()
		self.assertLessEqual(len(lines), 4096)
		for slot, line in enumerate(lines):
			number, name = line.split(" ")
			self.assertEqual(number, str(slot))
			self.assertTrue(name)

	def test_images_have_the_shader_sizes(self):
		with Image.open(os.path.join(REPO, "assets", "mcrl_atlas.png")) as atlas:
			self.assertEqual((atlas.size, atlas.mode), ((1024, 1024), "RGBA"))
		with Image.open(os.path.join(REPO, "assets", "mcrl_steve.png")) as skin:
			self.assertEqual((skin.size, skin.mode), ((64, 64), "RGBA"))
			self.assertEqual(ba.check_skin(np.array(skin)), [])


class BuildTest(TempDirTest):
	def setUp(self):
		super().setUp()
		self.texture("dirt", solid((120, 80, 40)))
		overlay = solid((150, 150, 150), alpha=0)
		overlay[:5] = (150, 150, 150, 255)
		self.texture("overlay", overlay)
		self.texture("grass_top", solid((140, 140, 140)))
		self.texture("stone", np.concatenate([solid((90, 90, 90)), solid((1, 1, 1))]))
		self.texture("big", solid((3, 3, 3), 32, 32))
		save(self.textures, "character", unique_skin())
		self.generated = [Generated("mcrl_side", "dirt", "overlay", (0x91, 0xBD, 0x59))]
		self.blocks = [
			Block("barrier", None, None, None, visible=False, color=(0, 0, 0)),
			cube("stone", "stone"),
			Block("grass_block", "grass_top", "mcrl_side", "dirt", tint_faces=blocks.TOP, tint=(0x91, 0xBD, 0x59)),
		]

	def build(self, out, **kw):
		return ba.build(self.textures, out, generated=self.generated, block_list=self.blocks, log=quiet, **kw)

	def test_outputs(self):
		out = os.path.join(self.dir, "out")
		self.build(out)
		self.assertEqual(sorted(os.listdir(out)),
			["mcrl_atlas.png", "mcrl_atlas.txt", "mcrl_blocks.txt", "mcrl_steve.png"])
		with Image.open(os.path.join(out, "mcrl_atlas.png")) as atlas:
			self.assertEqual((atlas.size, atlas.mode), ((1024, 1024), "RGBA"))
			pixels = np.array(atlas)
		with Image.open(os.path.join(out, "mcrl_steve.png")) as skin:
			self.assertEqual((skin.size, skin.mode), ((64, 64), "RGBA"))
		# character.png (64 x 32) and big.png (32 x 32) are not atlas textures
		with open(os.path.join(out, "mcrl_atlas.txt"), encoding="utf-8") as f:
			self.assertEqual(f.read(), "0 mcrl_side\n1 dirt\n2 grass_top\n3 overlay\n4 stone\n")
		with open(os.path.join(out, "mcrl_blocks.txt"), encoding="utf-8") as f:
			rows = parse_blocks_file(f.read())
		self.assertEqual(rows[0], "1 barrier -1 -1 -1 1 0 1 0 #ffffff #000000".split())
		self.assertEqual(rows[1], "2 stone 4 4 4 1 1 1 0 #ffffff #5a5a5a".split())
		# side colour: 80 texels of tinted overlay (85, 111, 52) and 176 of dirt (120, 80, 40), averaged
		self.assertEqual(rows[2], "3 grass_block 2 0 1 1 1 1 1 #91bd59 #6d5a2c".split())
		# the grass side's tint is baked: its top rows are the tinted overlay
		tinted = tuple((150 * t + 127) // 255 for t in (0x91, 0xBD, 0x59))
		self.assertEqual(tuple(pixels[0, 0, :3]), tinted)
		self.assertEqual(tuple(pixels[15, 15, :3]), (120, 80, 40))

	def test_running_twice_gives_identical_bytes(self):
		first = os.path.join(self.dir, "first")
		second = os.path.join(self.dir, "second")
		self.build(first, sheet=os.path.join(self.dir, "sheet1.png"))
		self.build(second, sheet=os.path.join(self.dir, "sheet2.png"))
		names = sorted(os.listdir(first))
		self.assertEqual(names, sorted(os.listdir(second)))
		for name in names:
			with open(os.path.join(first, name), "rb") as a, open(os.path.join(second, name), "rb") as b:
				self.assertEqual(a.read(), b.read(), name)
		with open(os.path.join(self.dir, "sheet1.png"), "rb") as a, open(os.path.join(self.dir, "sheet2.png"),
				"rb") as b:
			self.assertEqual(a.read(), b.read())

	def test_contact_sheet_is_written(self):
		sheet = os.path.join(self.dir, "sheets", "blocks.png")
		self.build(os.path.join(self.dir, "out"), sheet=sheet)
		with Image.open(sheet) as image:
			self.assertGreater(image.size[0], 3 * 128)

	def test_skin_option_and_bad_skin(self):
		modern = os.path.join(self.dir, "modern.png")
		Image.fromarray(unique_skin(64, 64)).save(modern)
		out = os.path.join(self.dir, "out")
		self.build(out, skin=modern)
		with Image.open(os.path.join(out, "mcrl_steve.png")) as skin:
			np.testing.assert_array_equal(np.array(skin), unique_skin(64, 64))
		holey = unique_skin()
		holey[16:32, 40:56, 3] = 0
		Image.fromarray(holey).save(os.path.join(self.dir, "holey.png"))
		with self.assertRaises(ba.AssetError):
			self.build(os.path.join(self.dir, "out2"), skin=os.path.join(self.dir, "holey.png"))
		with self.assertRaises(ba.AssetError):
			self.build(os.path.join(self.dir, "out3"), skin=os.path.join(self.dir, "missing.png"))

	def test_main_reports_errors(self):
		err = io.StringIO()
		with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
			code = ba.main(["--textures", os.path.join(self.dir, "nope"), "--out", os.path.join(self.dir, "o")])
		self.assertEqual(code, 1)
		self.assertIn("error:", err.getvalue())


if __name__ == "__main__":
	unittest.main()
