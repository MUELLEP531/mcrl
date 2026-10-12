"""Tests for render.py: python3 -m unittest discover -s tools/preview -p 'test_*.py'"""

import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import time
import unittest
import unittest.mock
import warnings

import numpy as np
from PIL import Image

# render.py sits next to this file; found even when Python does not put the test's folder on the path (-I).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

F32 = np.float32
S = 0.9375 / 16.0  # one Steve model pixel in blocks


def read_bytes(path):
	with open(path, "rb") as f:
		return bytearray(f.read())


def write_bytes(path, data):
	with open(path, "wb") as f:
		f.write(bytes(data))


def grid(world_cell):
	"""A world cell relative to the grid's corner (-64, 48, -64)."""
	return (world_cell[0] + 64, world_cell[1] - 48, world_cell[2] + 64)


def coord_atlas():
	"""An atlas whose texels say where they are: r = texel x in its slot, g = texel y, b = 60 * (1 + slot % 4)."""
	y, x = np.mgrid[0:1024, 0:1024]
	slot = (y // 16) * 64 + x // 16
	img = np.stack([(x % 16) * 16 + 8, (y % 16) * 16 + 8, 60 * (1 + slot % 4), np.full_like(x, 255)], axis=-1)
	return img.astype(F32) / F32(255.0)


def unique_skin():
	"""A skin whose texels say where they are: r = 4x, g = 4y, b = 255 (so b is the face's light)."""
	y, x = np.mgrid[0:64, 0:64]
	img = np.stack([x * 4, y * 4, np.full_like(x, 255), np.full_like(x, 255)], axis=-1)
	return img.astype(F32) / F32(255.0)


def skin_texel(color):
	"""(x, y, shade) of a unique_skin() colour."""
	shade = float(color[2])
	return int(round(color[0] / shade * 255.0 / 4.0)), int(round(color[1] / shade * 255.0 / 4.0)), round(shade, 5)


def trace_one(tex, o, d, t_max=1000.0, t_surf=1e9, hole=False):
	d = np.asarray(d, dtype=np.float64)
	d = (d / np.linalg.norm(d)).astype(F32)
	return render.trace_voxels(tex, np.asarray(o, F32), d[None, :], t_max, t_surf, hole)


def steve_one(skin, steve, o, d):
	d = np.asarray(d, dtype=np.float64)
	d = (d / np.linalg.norm(d)).astype(F32)
	h = render.trace_steve(skin, steve, np.asarray(o, F32), d[None, :])
	return bool(h.hit[0]), float(h.t[0]), h.color[0]


def reference_walk(o, d, t_max):
	"""Brute force: every cell the ray passes through (cells between consecutive plane crossings, found by sorting the
	crossings and taking the cell at each interval's midpoint), in float64. Also the crossing times (first: where the
	walk starts) and the smallest gap between consecutive events, below which the order is ambiguous in float32."""
	o = np.asarray(o, dtype=np.float64)
	d = np.asarray(d, dtype=np.float64)
	lo = np.array([-64.0, 48.0, -64.0])
	hi = np.array([64.0, 88.0, 64.0])
	dd = d + (np.abs(d) < 1e-6) * 1e-6
	ta, tb = (lo - o) / dd, (hi - o) / dd
	t_enter = np.minimum(ta, tb).max()
	t_end = min(np.maximum(ta, tb).min(), t_max)
	if t_end <= max(t_enter, 0.0):
		return [], [], np.inf, t_enter
	start = max(t_enter, 0.0)
	events = [start]
	for a in range(3):
		for k in range(int(lo[a]) + 1, int(hi[a])):  # interior planes; the grid's own faces bound the walk
			tk = (k - o[a]) / dd[a]
			if start < tk <= t_end:
				events.append(tk)
	events.sort()
	bounds = events + [t_end]
	cells = []
	for i in range(len(events)):
		mid = 0.5 * (bounds[i] + bounds[i + 1])
		cells.append(tuple(int(v) for v in np.floor(o + dd * mid) - lo))
	return cells, events, float(np.diff(bounds).min()) if len(bounds) > 1 else np.inf, t_enter


def walks(tex, origins, dirs, t_max):
	"""The cells trace_voxels tests per ray, in order, with their t."""
	record = []
	hits = render.trace_voxels(tex, origins, dirs, t_max, 1e9, False, record=record)
	cells = {i: [] for i in range(len(dirs))}
	times = {i: [] for i in range(len(dirs))}
	for idx, c, t in record:
		for k, i in enumerate(idx):
			cells[int(i)].append(tuple(int(v) for v in c[k]))
			times[int(i)].append(float(t[k]))
	return hits, cells, times


def flat_floor_scene(width, height, fov, cam_pos, yaw, pitch, tex=None, steve=None):
	return render.Scene(width, height, fov, cam_pos, yaw, pitch, tex or render.Textures.empty(),
		steve or render.Steve(), render.Arena())


def pixel_points(scene, plane_axis, plane_value):
	"""Per pixel, where its ray meets the world plane `axis = value` (independent of render_frame)."""
	ray = render.pixel_rays(scene.width, scene.height, scene.fov_y).astype(np.float64)
	rows = render.camera_rows(scene.cam_yaw, scene.cam_pitch).astype(np.float64)
	d = ray @ rows.T
	o = np.asarray(scene.cam_pos, dtype=np.float64)
	t = (plane_value - o[plane_axis]) / d[..., plane_axis]
	return o + d * t[..., None]


def pixels_in_cell(points, axes, cell, margin=0.15):
	"""Pixels whose plane points lie inside cell[axes] by at least `margin`."""
	mask = np.ones(points.shape[:2], dtype=bool)
	for a in axes:
		f = points[..., a] - cell[a]
		mask &= (f > margin) & (f < 1.0 - margin)
	return mask


class TexelAddressingTest(unittest.TestCase):
	def test_index_formula(self):
		self.assertEqual(render.texel_index(0, 0, 0), 0)
		self.assertEqual(render.texel_index(5, 7, 9), 5 + 128 * (9 + 128 * 7))
		self.assertEqual(render.texel_index(127, 39, 127), 128 * 5120 - 1)
		self.assertEqual(render.texel_index(0, 1, 0), 128 * 128)

	def test_vox_file_is_little_endian_row_major_width_128(self):
		with tempfile.TemporaryDirectory() as tmp:
			tex = render.Textures.empty()
			render.write_textures(tmp, tex)
			data = read_bytes(os.path.join(tmp, "vox.r16"))
			i = render.texel_index(5, 7, 9)
			data[2 * i:2 * i + 2] = struct.pack("<H", 0xBEEF)
			write_bytes(os.path.join(tmp, "vox.r16"), data)
			loaded = render.load_textures(tmp)
		self.assertEqual(loaded.vox[9 + 128 * 7, 5], 0xBEEF)
		cells = np.array([[5, 7, 9], [6, 7, 9], [5, 8, 9], [5, 7, 10], [9, 7, 5]], dtype=np.int32)
		self.assertEqual(render.vox_at(loaded, cells).tolist(), [0xBEEF, 0, 0, 0, 0])

	def test_walks_texel_lookup_equals_vox_at(self):
		# trace_voxels reads MCVOX through texel_index(); vox_at() samples it as the shader does
		tex = render.Textures.empty()
		tex.vox[:] = np.arange(5120 * 128, dtype=np.uint32).reshape(5120, 128) % 65521
		y, z, x = np.mgrid[0:40, 0:128, 0:128]
		cells = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.int32)
		np.testing.assert_array_equal(render.vox_at(tex, cells),
			tex.vox.reshape(-1)[render.texel_index(cells[:, 0], cells[:, 1], cells[:, 2])])
		np.testing.assert_array_equal(render.vox_at(tex, cells), tex.vox.reshape(-1))

	def test_set_block_uses_world_cells(self):
		tex = render.Textures.empty()
		render.set_block(tex, (-64, 48, -64), 7)
		render.set_block(tex, (63, 87, 63), 9)
		render.set_block(tex, (1, 50, -3), 11)
		flat = tex.vox.reshape(-1)
		self.assertEqual(flat[0], 7)
		self.assertEqual(flat[-1], 9)
		self.assertEqual(flat[render.texel_index(65, 2, 61)], 11)
		with self.assertRaises(ValueError):
			render.set_block(tex, (64, 60, 0), 1)
		with self.assertRaises(ValueError):
			render.set_block(tex, (0, 47, 0), 1)

	def test_cells_classification(self):
		tex = render.Textures.empty()
		values = [0, 63, 64, 120, 178, 179, 240, 255]
		for i, v in enumerate(values):
			tex.cells[0, i] = v
		cells = np.array([[i, 0, 0] for i in range(len(values))], dtype=np.int32)
		self.assertEqual(render.shell_at(tex, cells).tolist(), [0, 0, 1, 1, 1, 2, 2, 2])
		state_cells = [(0, 60, 0), (1, 60, 0), (2, 60, 0)]
		for state, cell in enumerate(state_cells):
			render.set_cell_state(tex, cell, state)
		self.assertEqual(render.shell_at(tex, np.array([grid(c) for c in state_cells])).tolist(), [0, 1, 2])
		self.assertEqual([tex.cells.reshape(-1)[render.texel_index(*grid(c))] for c in state_cells], [0, 120, 240])

	def test_types_file_layout(self):
		with tempfile.TemporaryDirectory() as tmp:
			tex = render.Textures.empty()
			render.write_textures(tmp, tex)
			data = read_bytes(os.path.join(tmp, "types.rgba32f"))
			block = 300
			for row in range(3):
				texel = (block // 256 + 128 * row) * 256 + block % 256
				data[16 * texel:16 * texel + 16] = struct.pack("<4f", row + 0.25, 2047.0 + row, -1.0, 4095.0)
			write_bytes(os.path.join(tmp, "types.rgba32f"), data)
			loaded = render.load_textures(tmp)
		for row in range(3):
			self.assertEqual(render.type_row(loaded, np.array([block]), row)[0].tolist(),
				[row + 0.25, 2047.0 + row, -1.0, 4095.0])
		# ids past 32767 read past the texture's last row: the sampler clamps
		loaded.types[383, 5] = (9.0, 9.0, 9.0, 9.0)
		self.assertEqual(render.type_row(loaded, np.array([65535 - 250]), 2)[0].tolist(), [9.0, 9.0, 9.0, 9.0])

	def test_wrong_file_size_is_an_error(self):
		with tempfile.TemporaryDirectory() as tmp:
			render.write_textures(tmp, render.Textures.empty())
			with open(os.path.join(tmp, "cells.r8"), "ab") as f:
				f.write(b"\0")
			with self.assertRaisesRegex(ValueError, "cells.r8"):
				render.load_textures(tmp)


class DdaTest(unittest.TestCase):
	def check_against_reference(self, origins, dirs, t_max, tol, min_kept):
		tex = render.Textures.empty()
		_, cells, times = walks(tex, origins, dirs, t_max)
		kept = 0
		for i in range(len(dirs)):
			ref, events, gap, _ = reference_walk(origins[i], dirs[i], float(t_max[i]))
			if gap < tol:
				continue  # two crossings too close to order reliably in float32
			kept += 1
			self.assertEqual(cells[i], ref, f"ray {i}: o={origins[i].tolist()} d={dirs[i].tolist()}")
			np.testing.assert_allclose(times[i], events, rtol=0, atol=tol / 2)
		self.assertGreaterEqual(kept, min_kept)

	def test_matches_brute_force_from_inside(self):
		rng = np.random.default_rng(1)
		n = 300
		origins = rng.uniform(render.GRID_LO + 0.5, render.GRID_HI - 0.5, (n, 3)).astype(F32)
		dirs = rng.normal(size=(n, 3))
		dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(F32)
		t_max = rng.uniform(5.0, 60.0, n).astype(F32)
		self.check_against_reference(origins, dirs, t_max, 2e-3, 150)

	def test_matches_brute_force_from_outside(self):
		rng = np.random.default_rng(2)
		n = 400
		centre = np.array([0.0, 68.0, 0.0])
		away = rng.normal(size=(n, 3))
		origins = (centre + 150.0 * away / np.linalg.norm(away, axis=1, keepdims=True)).astype(F32)
		targets = rng.uniform(render.GRID_LO, render.GRID_HI, (n, 3))
		dirs = targets - origins
		dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(F32)
		self.check_against_reference(origins, dirs, np.full(n, 400.0, F32), 2e-3, 100)

	def test_first_block_matches_brute_force(self):
		rng = np.random.default_rng(3)
		tex = render.Textures.empty()
		render.define_type(tex, 1, (0.5, 0.5, 0.5))
		solid = rng.random((40, 128, 128)) < 0.02  # [y, z, x]
		tex.vox[:] = solid.reshape(5120, 128).astype(np.uint16)
		n = 300
		origins = rng.uniform(render.GRID_LO + 0.5, render.GRID_HI - 0.5, (n, 3)).astype(F32)
		dirs = rng.normal(size=(n, 3))
		dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(F32)
		hits = render.trace_voxels(tex, origins, dirs, 300.0, 1e9, False)
		checked = 0
		for i in range(n):
			ref, events, gap, _ = reference_walk(origins[i], dirs[i], 300.0)
			if gap < 2e-3:
				continue
			checked += 1
			# the cell holding the origin is never drawn
			first = next((k for k in range(1, len(ref)) if solid[ref[k][1], ref[k][2], ref[k][0]]), None)
			if first is None:
				self.assertFalse(hits.hit[i])
			else:
				self.assertTrue(hits.hit[i])
				self.assertEqual(tuple(hits.cell[i]), ref[first])
				self.assertAlmostEqual(float(hits.t[i]), events[first], delta=1e-3)
		self.assertGreater(checked, 100)

	def test_axis_aligned_ray_walks_one_axis(self):
		tex = render.Textures.empty()
		origins = np.array([[0.5, 70.5, 0.5]], F32)
		_, cells, times = walks(tex, origins, np.array([[1.0, 0.0, 0.0]], F32), np.array([1000.0], F32))
		self.assertEqual(cells[0], [(x, 22, 64) for x in range(64, 128)])
		np.testing.assert_allclose(times[0], [0.0] + [x - 0.5 for x in range(1, 64)], atol=1e-4)
		_, cells, _ = walks(tex, origins, np.array([[0.0, -1.0, 0.0]], F32), np.array([1000.0], F32))
		self.assertEqual(cells[0], [(64, y, 64) for y in range(22, -1, -1)])

	def test_ties_step_z_then_y(self):
		tex = render.Textures.empty()
		o = np.array([[0.5, 60.5, 0.5]] * 3, F32)
		d = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
		d = (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(F32)
		_, cells, _ = walks(tex, o, d, np.full(3, 2.5, F32))
		self.assertEqual(cells[0][:5], [(64, 12, 64), (64, 13, 64), (65, 13, 64), (65, 14, 64), (66, 14, 64)])
		self.assertEqual(cells[1][:5], [(64, 12, 64), (64, 12, 65), (65, 12, 65), (65, 12, 66), (66, 12, 66)])
		self.assertEqual(cells[2][:4], [(64, 12, 64), (64, 12, 65), (64, 13, 65), (65, 13, 65)])

	def test_step_limit(self):
		tex = render.Textures.empty()
		render.define_type(tex, 1, (1.0, 1.0, 1.0))
		with unittest.mock.patch.object(render, "MAX_STEPS", 5):
			render.set_block(tex, (0, 70, 4), 1)  # the 5th cell tested
			self.assertTrue(trace_one(tex, (0.5, 70.5, 0.5), (0, 0, 1)).hit[0])
			render.set_block(tex, (0, 70, 4), 0)
			render.set_block(tex, (0, 70, 5), 1)
			self.assertFalse(trace_one(tex, (0.5, 70.5, 0.5), (0, 0, 1)).hit[0])
		self.assertTrue(trace_one(tex, (0.5, 70.5, 0.5), (0, 0, 1)).hit[0])

	def test_no_rays(self):
		h = render.trace_voxels(render.Textures.empty(), np.zeros(3, F32), np.zeros((0, 3), F32), 1.0, 1e9, False)
		self.assertEqual(h.hit.shape, (0,))
		self.assertEqual(h.color.shape, (0, 4))

	def test_ray_missing_the_grid_tests_nothing(self):
		tex = render.Textures.empty()
		render.define_type(tex, 1, (1.0, 1.0, 1.0))
		tex.vox[:] = 1
		hits, cells, _ = walks(tex, np.array([[0.0, 100.0, 0.0], [0.0, 100.0, 0.0]], F32),
			np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], F32), np.array([1000.0, 1000.0], F32))
		self.assertEqual(cells, {0: [], 1: []})
		self.assertFalse(hits.hit.any())

	def test_camera_cell_is_never_drawn(self):
		tex = render.Textures.empty()
		render.define_type(tex, 1, (1.0, 1.0, 1.0))
		render.set_block(tex, (0, 70, 0), 1)
		render.set_block(tex, (0, 70, 3), 1)
		h = trace_one(tex, (0.5, 70.5, 0.5), (0, 0, 1))
		self.assertTrue(h.hit[0])
		self.assertEqual(tuple(h.cell[0]), grid((0, 70, 3)))
		self.assertAlmostEqual(float(h.t[0]), 2.5, places=4)

	def test_t_max_ends_the_walk(self):
		tex = render.Textures.empty()
		render.define_type(tex, 1, (1.0, 1.0, 1.0))
		render.set_block(tex, (0, 70, 3), 1)
		self.assertFalse(trace_one(tex, (0.5, 70.5, 0.5), (0, 0, 1), t_max=2.49).hit[0])
		self.assertTrue(trace_one(tex, (0.5, 70.5, 0.5), (0, 0, 1), t_max=2.51).hit[0])

	def test_entering_from_outside_draws_the_first_cell(self):
		tex = render.Textures.empty()
		render.define_type(tex, 1, (1.0, 1.0, 1.0))
		render.set_block(tex, (-64, 60, 0), 1)
		h = trace_one(tex, (-70.0, 60.5, 0.5), (1, 0, 0))
		self.assertTrue(h.hit[0])
		self.assertAlmostEqual(float(h.t[0]), 6.0, places=4)
		self.assertEqual(int(h.face[0]), 1)
		self.assertAlmostEqual(float(h.shade[0]), 0.6, places=6)


class FaceTest(unittest.TestCase):
	def setUp(self):
		self.tex = render.Textures.empty()
		self.tex.atlas = coord_atlas()
		render.define_type(self.tex, 1, (0.5, 0.5, 0.5), slots=(1, 2, 3))
		render.set_block(self.tex, (0, 64, 0), 1)

	def check(self, o, d, face, shade, uv):
		h = trace_one(self.tex, o, d)
		self.assertTrue(h.hit[0])
		self.assertAlmostEqual(float(h.t[0]), 5.0 if face != 2 else 4.0, places=4)
		self.assertEqual(int(h.face[0]), face)
		self.assertAlmostEqual(float(h.shade[0]), shade, places=6)
		np.testing.assert_allclose(h.uv[0], uv, atol=1e-4)
		# the colour is the atlas texel of this face's slot at floor(uv * 15.999), times the light
		slot = (1, 2, 3)[face]
		tx, ty = (int(math.floor(x * 15.999)) for x in uv)
		expected = np.array([tx * 16 + 8, ty * 16 + 8, 60 * (1 + slot % 4)]) / 255.0 * shade
		np.testing.assert_allclose(h.color[0, :3], expected, atol=1e-6)
		self.assertEqual(float(h.color[0, 3]), 1.0)

	def test_top(self):
		self.check((0.3, 70.0, 0.7), (0, -1, 0), 0, 1.0, (0.3, 0.7))

	def test_bottom(self):
		self.check((0.3, 60.0, 0.7), (0, 1, 0), 2, 0.5, (0.3, 0.7))

	def test_x_sides(self):
		self.check((-5.0, 64.25, 0.7), (1, 0, 0), 1, 0.6, (0.7, 0.75))
		self.check((6.0, 64.25, 0.7), (-1, 0, 0), 1, 0.6, (0.3, 0.75))

	def test_z_sides(self):
		self.check((0.3, 64.25, -5.0), (0, 0, 1), 1, 0.8, (0.7, 0.75))
		self.check((0.3, 64.25, 6.0), (0, 0, -1), 1, 0.8, (0.3, 0.75))

	def test_tinted_faces(self):
		render.define_type(self.tex, 1, (0.5, 0.5, 0.5), slots=(1, 2, 3), tint=(0.5, 1.0, 0.25), tint_faces=1)
		top = trace_one(self.tex, (0.3, 70.0, 0.7), (0, -1, 0)).color[0]
		side = trace_one(self.tex, (-5.0, 64.25, 0.7), (1, 0, 0)).color[0]
		np.testing.assert_allclose(top[:3], np.array([4 * 16 + 8, 11 * 16 + 8, 120]) / 255.0 * (0.5, 1.0, 0.25),
			atol=1e-6)
		np.testing.assert_allclose(side[:3], np.array([11 * 16 + 8, 11 * 16 + 8, 180]) / 255.0 * 0.6, atol=1e-6)
		render.define_type(self.tex, 1, (0.5, 0.5, 0.5), slots=(1, 2, 3), tint=(0.5, 1.0, 0.25), tint_faces=6)
		top = trace_one(self.tex, (0.3, 70.0, 0.7), (0, -1, 0)).color[0]
		bottom = trace_one(self.tex, (0.3, 60.0, 0.7), (0, 1, 0)).color[0]
		side = trace_one(self.tex, (-5.0, 64.25, 0.7), (1, 0, 0)).color[0]
		np.testing.assert_allclose(top[:3], np.array([72, 184, 120]) / 255.0, atol=1e-6)
		np.testing.assert_allclose(bottom[:3], np.array([72, 184, 240]) / 255.0 * 0.5 * (0.5, 1.0, 0.25), atol=1e-6)
		np.testing.assert_allclose(side[:3], np.array([184, 184, 180]) / 255.0 * 0.6 * (0.5, 1.0, 0.25), atol=1e-6)

	def test_untextured_colour_is_the_blocks_own_noised_per_texel(self):
		render.define_type(self.tex, 2, (0.6, 0.4, 0.2))
		render.set_block(self.tex, (0, 64, 0), 2)
		n = 64
		xs = (np.arange(n) + 0.5) / n
		o = np.stack([xs, np.full(n, 70.0), np.full(n, 0.5)], axis=1).astype(F32)
		colors = np.array([trace_one(self.tex, o[i], (0, -1, 0)).color[0] for i in range(n)])
		k = colors[:, :3] / np.array([0.6, 0.4, 0.2])
		np.testing.assert_allclose(k, k[:, :1].repeat(3, axis=1), atol=1e-5)  # one factor for r, g and b
		self.assertTrue(((k >= 0.8 - 1e-6) & (k <= 1.15 + 1e-6)).all())
		# constant within a texel (4 rays per texel here), varying between texels
		self.assertTrue(np.allclose(k[0::4, 0], k[3::4, 0]))
		self.assertGreater(len(np.unique(np.round(k[::4, 0], 5))), 8)
		# the formula, for the first texel: cell (64, 16, 64), texel (0, 7) (floor(0.5 * 15.999) = 7)
		px = np.array([0 + 64 * 16 + 16 * 7, 7 + 64 * 16], dtype=F32)
		noise = np.sin(px[0] * F32(12.9898) + px[1] * F32(78.233)) * F32(43758.5453)
		noise = noise - np.floor(noise)
		self.assertAlmostEqual(float(k[0, 0]), float(F32(0.8) + F32(0.35) * noise), places=5)

	def test_no_atlas_draws_textured_faces_untextured(self):
		self.tex.atlas = None
		render.define_type(self.tex, 1, (0.6, 0.4, 0.2), slots=(1, 2, 3))
		c = trace_one(self.tex, (0.3, 70.0, 0.7), (0, -1, 0)).color[0]
		k = c[:3] / np.array([0.6, 0.4, 0.2])
		self.assertTrue(np.allclose(k, k[0]) and 0.8 - 1e-6 <= k[0] <= 1.15 + 1e-6)

	def test_cutout_texels_are_see_through(self):
		atlas = self.tex.atlas
		atlas[0:16, 16:32, 3] = 0.5  # slot 1 (the top): alpha 0.5 is not > 0.5
		render.define_type(self.tex, 3, (0.2, 0.2, 0.9))
		render.set_block(self.tex, (0, 62, 0), 3)
		h = trace_one(self.tex, (0.3, 70.0, 0.7), (0, -1, 0))
		self.assertTrue(h.hit[0])
		self.assertEqual(tuple(h.cell[0]), grid((0, 62, 0)))
		self.assertAlmostEqual(float(h.t[0]), 7.0, places=4)
		atlas[0:16, 16:32, 3] = 129.0 / 255.0
		self.assertEqual(tuple(trace_one(self.tex, (0.3, 70.0, 0.7), (0, -1, 0)).cell[0]), grid((0, 64, 0)))


class ShellRuleTest(unittest.TestCase):
	def setUp(self):
		self.tex = render.Textures.empty()
		render.define_type(self.tex, 1, (0.5, 0.5, 0.5))
		render.set_block(self.tex, (0, 64, 0), 1)

	def hit(self, t_surf, hole):
		return bool(trace_one(self.tex, (0.5, 70.0, 0.5), (0, -1, 0), t_surf=t_surf, hole=hole).hit[0])

	def test_shell_block_hidden_near_rls_surface(self):
		render.set_cell_state(self.tex, (0, 64, 0), 1)
		# the block's cell is entered at t = 5: hidden while RL's surface is less than 1.5 behind that
		self.assertFalse(self.hit(5.0, False))
		self.assertFalse(self.hit(6.49, False))
		self.assertTrue(self.hit(6.51, False))
		self.assertFalse(self.hit(1.0, False))

	def test_hole_shows_shell_blocks(self):
		render.set_cell_state(self.tex, (0, 64, 0), 1)
		self.assertTrue(self.hit(5.0, True))

	def test_open_and_dug_cells_are_never_hidden(self):
		for state in (0, 2):
			render.set_cell_state(self.tex, (0, 64, 0), state)
			self.assertTrue(self.hit(5.0, False))


class CameraTest(unittest.TestCase):
	def forward(self, yaw, pitch):
		return render.camera_rows(yaw, pitch) @ np.array([0.0, 0.0, -1.0])

	def test_yaw_and_pitch(self):
		np.testing.assert_allclose(self.forward(0, 0), (0, 0, 1), atol=1e-6)
		np.testing.assert_allclose(self.forward(90, 0), (-1, 0, 0), atol=1e-6)
		np.testing.assert_allclose(self.forward(180, 0), (0, 0, -1), atol=1e-6)
		np.testing.assert_allclose(self.forward(-90, 0), (1, 0, 0), atol=1e-6)
		np.testing.assert_allclose(self.forward(0, 90), (0, -1, 0), atol=1e-6)
		np.testing.assert_allclose(self.forward(37, -90), (0, 1, 0), atol=1e-6)
		np.testing.assert_allclose(self.forward(0, 30), (0, -0.5, math.sqrt(0.75)), atol=1e-6)

	def test_right_and_up(self):
		r = render.camera_rows(0, 0)
		np.testing.assert_allclose(r @ (1, 0, 0), (-1, 0, 0), atol=1e-6)  # facing +z, -x is on the right
		np.testing.assert_allclose(r @ (0, 1, 0), (0, 1, 0), atol=1e-6)
		np.testing.assert_allclose(render.camera_rows(0, 90) @ (0, 1, 0), (0, 0, 1), atol=1e-6)
		for yaw, pitch in ((0, 0), (33, 12), (-120, -45), (270, 80)):
			r = render.camera_rows(yaw, pitch).astype(np.float64)
			np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-6)
			self.assertAlmostEqual(np.linalg.det(r), 1.0, places=5)
			self.assertAlmostEqual((r @ (1, 0, 0))[1], 0.0, places=6)  # no roll

	def test_pixel_rays(self):
		ray = render.pixel_rays(5, 3, 90.0)
		self.assertEqual(ray.shape, (3, 5, 3))
		np.testing.assert_allclose(ray[1, 2], (0, 0, -1), atol=1e-7)
		# ((u * 2 - 1) * tanHalfFovY * aspect, (1 - v * 2) * tanHalfFovY, -1) at pixel centres
		np.testing.assert_allclose(ray[0, 0], ((0.1 * 2 - 1) * 5 / 3, 1 - 2 / 6, -1), atol=1e-6)
		np.testing.assert_allclose(ray[2, 4], ((0.9 * 2 - 1) * 5 / 3, 1 - 10 / 6, -1), atol=1e-6)


class ArenaTest(unittest.TestCase):
	def exit(self, origin, direction):
		d = np.asarray(direction, dtype=np.float64)
		d = (d / np.linalg.norm(d))[None]
		t, kind, normal = render.arena_surface(render.Arena(), np.asarray(origin, np.float64), d)
		return float(t[0]), int(kind[0]), normal[0]

	def test_walls_floor_ceiling(self):
		c = (0.0, 69.0, 0.0)  # 5 blocks above RL's floor (y = 64)
		cases = [((1, 0, 0), 40.96, render.KIND_WALL, (-1, 0, 0)), ((-1, 0, 0), 40.96, render.KIND_WALL, (1, 0, 0)),
			((0, 1, 0), 20.44 - 5.0, render.KIND_CEILING, (0, -1, 0)),
			((0, -1, 0), 5.0, render.KIND_FLOOR, (0, 1, 0))]
		for d, dist, kind, normal in cases:
			t, k, n = self.exit(c, d)
			self.assertAlmostEqual(t, dist, places=6)
			self.assertEqual(k, kind)
			np.testing.assert_allclose(n, normal, atol=1e-9)

	def test_back_walls_beside_and_above_the_goals(self):
		t, k, n = self.exit((20.0, 66.0, 0.0), (0, 0, 1))
		self.assertAlmostEqual(t, 51.2, places=6)
		self.assertEqual(k, render.KIND_WALL)
		np.testing.assert_allclose(n, (0, 0, -1), atol=1e-9)
		t, k, _ = self.exit((0.0, 72.0, 0.0), (0, 0, -1))  # above the crossbar (y = 70.43)
		self.assertAlmostEqual(t, 51.2, places=6)
		self.assertEqual(k, render.KIND_WALL)

	def test_corners(self):
		t, k, n = self.exit((0.0, 66.0, 0.0), (1, 0, 1))
		self.assertAlmostEqual(t, 40.32 * math.sqrt(2.0), places=6)
		self.assertEqual(k, render.KIND_WALL)
		np.testing.assert_allclose(n, (-math.sqrt(0.5), 0, -math.sqrt(0.5)), atol=1e-9)
		t, _, _ = self.exit((-30.0, 66.0, 45.0), (-1, 0, 0))
		self.assertAlmostEqual(t, 80.64 - 45.0 - 30.0, places=6)

	def test_goals(self):
		t, k, n = self.exit((0.0, 66.0, 0.0), (0, 0, 1))
		self.assertAlmostEqual(t, 60.0, places=6)
		self.assertEqual(k, render.KIND_GOAL_ORANGE)
		np.testing.assert_allclose(n, (0, 0, -1), atol=1e-9)
		t, k, _ = self.exit((0.0, 66.0, 0.0), (0, 0, -1))
		self.assertAlmostEqual(t, 60.0, places=6)
		self.assertEqual(k, render.KIND_GOAL_BLUE)
		t, k, n = self.exit((0.0, 66.0, 55.0), (0, 1, 0))  # under the goal's roof
		self.assertAlmostEqual(t, 6.42775 - 2.0, places=6)
		self.assertEqual(k, render.KIND_GOAL_ORANGE)
		np.testing.assert_allclose(n, (0, -1, 0), atol=1e-9)
		t, k, _ = self.exit((0.0, 66.0, 55.0), (1, 0, 0))
		self.assertAlmostEqual(t, 8.93, places=6)
		t, k, _ = self.exit((0.0, 66.0, 55.0), (0, -1, 0))
		self.assertAlmostEqual(t, 2.0, places=6)
		self.assertEqual(k, render.KIND_FLOOR)
		# from inside the goal out through its mouth, across the pitch, to the other goal's back wall
		t, k, _ = self.exit((0.0, 66.0, 55.0), (0, 0, -1))
		self.assertAlmostEqual(t, 115.0, places=6)
		self.assertEqual(k, render.KIND_GOAL_BLUE)

	def test_from_outside_the_first_entry(self):
		t, k, n = self.exit((0.0, 100.0, 0.0), (0, -1, 0))
		self.assertAlmostEqual(t, 100.0 - 84.44, places=6)
		self.assertEqual(k, render.KIND_CEILING)
		np.testing.assert_allclose(n, (0, 1, 0), atol=1e-9)
		t, k, _ = self.exit((0.0, 100.0, 0.0), (0, 1, 0))
		self.assertEqual(t, math.inf)
		self.assertEqual(k, render.KIND_NONE)

	def test_contains_and_mapping(self):
		a = render.Arena()
		q = a.to_rl(np.array([[1.0, 65.0, 2.0], [40.96, 64.0, 0.0], [41.0, 64.0, 0.0], [0.0, 63.99, 0.0],
			[0.0, 64.0, 59.0], [9.0, 64.5, 55.0], [40.0, 64.0, 41.0]]))
		np.testing.assert_allclose(q[0], (100.0, 200.0, 100.0))
		self.assertEqual(a.contains_rl(q).tolist(), [True, True, False, False, True, False, False])
		np.testing.assert_allclose(a.to_world(q), [[1.0, 65.0, 2.0], [40.96, 64.0, 0.0], [41.0, 64.0, 0.0],
			[0.0, 63.99, 0.0], [0.0, 64.0, 59.0], [9.0, 64.5, 55.0], [40.0, 64.0, 41.0]], atol=1e-9)

	def test_frame_depth_is_linear(self):
		scene = flat_floor_scene(9, 9, 60.0, (0.5, 70.0, 0.5), 0.0, 90.0)
		frame = render.render_frame(scene)
		# looking straight down at the floor 6 below: linear depth 6 in every pixel
		np.testing.assert_allclose(frame.zh, 6.0, atol=1e-5)
		scene = flat_floor_scene(9, 9, 60.0, (30.5, 66.5, 0.5), -90.0, 0.0)
		np.testing.assert_allclose(render.render_frame(scene).zh[3:6, 2:7], 40.96 - 30.5, atol=1e-5)


class SteveTest(unittest.TestCase):
	def setUp(self):
		self.skin = unique_skin()

	def test_ray_at_his_face(self):
		steve = render.Steve((0.5, 64.0, 0.5, 1.0), (180.0, 180.0, 0.0, 0.0))  # facing -z
		hit, t, c = steve_one(self.skin, steve, (0.5 + 1.5 * S, 64.0 + 28.3 * S, -5.0), (0, 0, 1))
		self.assertTrue(hit)
		self.assertAlmostEqual(t, 5.0 + 0.5 - 4.0 * S, places=4)
		# the front of the head (8..16, 8..16); +x is his right, the face texture's left half
		self.assertEqual(skin_texel(c), (10, 11, 0.8))
		self.assertEqual(float(c[3]), 1.0)

	def test_hidden_when_not_shown(self):
		steve = render.Steve((0.5, 64.0, 0.5, 0.4), (180.0, 180.0, 0.0, 0.0))
		self.assertFalse(steve_one(self.skin, steve, (0.5, 64.0 + 28.3 * S, -5.0), (0, 0, 1))[0])

	def test_back_and_top_of_his_head(self):
		steve = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.0))  # facing +z, his left is +x
		hit, t, c = steve_one(self.skin, steve, (0.5 + 1.5 * S, 64.0 + 28.3 * S, -5.0), (0, 0, 1))
		self.assertTrue(hit)
		self.assertEqual(skin_texel(c), (26, 11, 0.8))  # back (24..32, 8..16), his left on the texture's left
		hit, t, c = steve_one(self.skin, steve, (0.5 + 1.5 * S, 70.0, 0.5 + 2.5 * S), (0, -1, 0))
		self.assertTrue(hit)
		self.assertAlmostEqual(t, 6.0 - 32.0 * S, places=4)
		self.assertEqual(skin_texel(c), (13, 6, 1.0))  # top (8..16, 0..8), his front at the bottom

	def test_arms_from_each_side(self):
		steve = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.0))  # facing +z, his left is +x
		y, z = 64.0 + 18.5 * S, 0.5 + 0.5 * S
		hit, t, c = steve_one(self.skin, steve, (5.0, y, z), (-1, 0, 0))
		self.assertTrue(hit)
		self.assertAlmostEqual(t, 4.5 - 8.0 * S, places=4)
		self.assertEqual(skin_texel(c), (41, 57, 0.6))  # left arm (32, 48): its outer side
		hit, t, c = steve_one(self.skin, steve, (-4.0, y, z), (1, 0, 0))
		self.assertTrue(hit)
		self.assertAlmostEqual(t, 4.5 - 8.0 * S, places=4)
		self.assertEqual(skin_texel(c), (42, 25, 0.6))  # right arm (40, 16): its outer side

	def test_legs_from_the_front(self):
		steve = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.0))  # facing +z
		hit, t, c = steve_one(self.skin, steve, (0.5 - 1.5 * S, 64.0 + 6.5 * S, 5.0), (0, 0, -1))
		self.assertTrue(hit)
		self.assertAlmostEqual(t, 4.5 - 2.0 * S, places=4)
		self.assertEqual(skin_texel(c), (6, 25, 0.8))  # right leg (0, 16), front: x 4..8, y 20..32
		hit, t, c = steve_one(self.skin, steve, (0.5 + 1.5 * S, 64.0 + 6.5 * S, 5.0), (0, 0, -1))
		self.assertEqual(skin_texel(c), (21, 57, 0.8))  # left leg (16, 48), front: x 20..24, y 52..64

	def test_head_yaw_turns_his_face(self):
		o, d = (-5.0, 64.0 + 28.3 * S, 0.5 + 1.5 * S), (1, 0, 0)
		ahead = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.0))
		self.assertEqual(skin_texel(steve_one(self.skin, ahead, o, d)[2]), (5, 11, 0.6))  # his right side
		turned = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 90.0, 0.0, 0.0))  # head yaw 90: looks toward -x
		self.assertEqual(skin_texel(steve_one(self.skin, turned, o, d)[2]), (13, 11, 0.8))  # his face

	def test_arm_swing_raises_his_right_arm(self):
		o, d = (0.5 - 6.0 * S, 70.0, 0.5 + 6.0 * S), (0, -1, 0)  # above a spot in front of his right shoulder
		rest = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
		self.assertFalse(steve_one(self.skin, rest, o, d)[0])
		swing = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.5, 0.0))
		hit, t, c = steve_one(self.skin, swing, o, d)
		self.assertTrue(hit)
		x, y, _ = skin_texel(c)
		self.assertTrue(40 <= x < 56 and 16 <= y < 32, (x, y))

	def test_walking_swings_the_legs_apart(self):
		o, d = (0.5 - 2.0 * S, 64.0 + 1.0 * S, 5.0), (0, 0, -1)  # at his right foot, from the front
		still = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
		walking = render.Steve((0.5, 64.0, 0.5, 1.0), (0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
		self.assertTrue(steve_one(self.skin, still, o, d)[0])
		# legR = 1.4 rad: his right foot swings back, out of this ray
		self.assertFalse(steve_one(self.skin, walking, o, d)[0])

	def test_clear_skin_texels_let_the_ray_through(self):
		skin = self.skin.copy()
		skin[8:16, 8:16, 3] = 0.0  # no face
		steve = render.Steve((0.5, 64.0, 0.5, 1.0), (180.0, 180.0, 0.0, 0.0))
		# steve_box tests only where the ray enters a box: through a clear texel the whole head is skipped
		self.assertFalse(steve_one(skin, steve, (0.5 + 1.5 * S, 64.0 + 28.3 * S, -5.0), (0, 0, 1))[0])
		# the body under it still shows
		self.assertTrue(steve_one(skin, steve, (0.5, 64.0 + 18.0 * S, -5.0), (0, 0, 1))[0])


class CompositeTest(unittest.TestCase):
	def floor_scene(self):
		"""Looking straight down from 6 above RL's floor at a shell layer with a dug cell, dirt under it, a block in
		a shell cell, a block in an open cell at the same height, and a block on the floor."""
		tex = render.Textures.empty()
		for i, color in enumerate(((0.5, 0.35, 0.2), (0.45, 0.45, 0.45), (0.95, 0.8, 0.25)), start=1):
			render.define_type(tex, i, color)
		for x in range(-5, 6):
			for z in range(-5, 6):
				render.set_cell_state(tex, (x, 63, z), 1)
		render.set_cell_state(tex, (0, 63, 0), 2)
		render.set_block(tex, (0, 62, 0), 1)
		render.set_block(tex, (2, 63, 0), 2)
		render.set_cell_state(tex, (-2, 63, 0), 0)
		render.set_block(tex, (-2, 63, 0), 2)
		render.set_block(tex, (0, 64, 2), 3)
		return flat_floor_scene(81, 81, 70.0, (0.5, 70.0, 0.5), 0.0, 90.0, tex)

	def test_hole_in_the_floor(self):
		scene = self.floor_scene()
		frame = render.render_frame(scene)
		points = pixel_points(scene, 1, 64.0)
		hole = pixels_in_cell(points, (0, 2), (0, 0, 0))
		self.assertGreater(hole.sum(), 20)
		self.assertTrue(frame.hole[hole].all())
		self.assertTrue(frame.visible[hole].all())
		np.testing.assert_allclose(frame.zm[hole], 7.0, atol=1e-4)  # the dirt's top, a block under RL's floor
		np.testing.assert_allclose(frame.image[hole], frame.world[hole][:, :3])
		self.assertTrue((np.abs(frame.image[hole] - frame.host[hole]).max(axis=1) > 0.05).all())
		# pixels not over the dug cell: RL's floor
		plain = pixels_in_cell(points, (0, 2), (-1, 0, -2)) | pixels_in_cell(points, (0, 2), (1, 0, -1))
		self.assertFalse(frame.hole[plain].any())
		np.testing.assert_array_equal(frame.image[plain], frame.host[plain])

	def test_blocks_in_shell_cells_hide_behind_rls_floor(self):
		scene = self.floor_scene()
		frame = render.render_frame(scene)
		points = pixel_points(scene, 1, 64.0)
		shell = pixels_in_cell(points, (0, 2), (2, 0, 0))
		self.assertGreater(shell.sum(), 20)
		self.assertFalse(frame.visible[shell].any())
		np.testing.assert_array_equal(frame.image[shell], frame.host[shell])
		# the same block in an open cell at the same height shows (its top is level with RL's floor)
		open_ = pixels_in_cell(points, (0, 2), (-2, 0, 0))
		self.assertTrue(frame.visible[open_].all())
		np.testing.assert_allclose(frame.zm[open_], frame.zh[open_], atol=1e-4)
		np.testing.assert_allclose(frame.image[open_], frame.world[open_][:, :3])

	def test_block_on_the_floor_shows(self):
		scene = self.floor_scene()
		frame = render.render_frame(scene)
		top = pixels_in_cell(pixel_points(scene, 1, 65.0), (0, 2), (0, 0, 2))
		self.assertGreater(top.sum(), 20)
		self.assertTrue(frame.visible[top].all())
		np.testing.assert_allclose(frame.zm[top], 5.0, atol=1e-4)
		np.testing.assert_allclose(frame.image[top] / np.array([0.95, 0.8, 0.25]),
			np.repeat((frame.image[top] / np.array([0.95, 0.8, 0.25]))[:, :1], 3, axis=1), atol=1e-5)

	def test_cells_debug_view(self):
		scene = self.floor_scene()
		frame = render.render_frame(scene, debug="cells")
		points = pixel_points(scene, 1, 64.0)
		dug =pixels_in_cell(points, (0, 2), (0, 0, 0))
		shell = pixels_in_cell(points, (0, 2), (1, 0, 1))
		open_ = pixels_in_cell(points, (0, 2), (-2, 0, 0))
		np.testing.assert_allclose(frame.image[dug], np.tile([0.1, 1.0, 0.1], (dug.sum(), 1)), atol=1e-6)
		np.testing.assert_allclose(frame.image[shell], np.tile([1.0, 0.15, 0.1], (shell.sum(), 1)), atol=1e-6)
		np.testing.assert_allclose(frame.image[open_], frame.host[open_] * 0.3, atol=1e-6)

	def wall_scene(self, dug, steve=None):
		"""Looking at RL's side wall (x = 40.96, inside the shell cells x = 40) with a gold block behind it."""
		tex = render.Textures.empty()
		render.define_type(tex, 1, (0.95, 0.8, 0.25))
		for y in range(60, 76):
			for z in range(-6, 7):
				render.set_cell_state(tex, (40, y, z), 1)
		if dug:
			render.set_cell_state(tex, (40, 66, 0), 2)
		render.set_block(tex, (42, 66, 0), 1)
		return flat_floor_scene(61, 61, 20.0, (30.5, 66.5, 0.5), -90.0, 0.0, tex, steve)

	def test_hole_in_a_wall(self):
		for dug in (True, False):
			scene = self.wall_scene(dug)
			frame = render.render_frame(scene)
			points = pixel_points(scene, 0, 40.96)
			cell = pixels_in_cell(points, (1, 2), (0, 66, 0))
			self.assertGreater(cell.sum(), 20)
			self.assertEqual(bool(frame.hole[cell].all()), dug)
			self.assertEqual(bool(frame.visible[cell].all()), dug)
			np.testing.assert_allclose(frame.zh[cell], 40.96 - 30.5, atol=1e-4)
			if dug:  # the gold block's -x face, 11.5 away, through the hole
				np.testing.assert_allclose(frame.zm[cell], 11.5, atol=1e-4)
				self.assertTrue(np.allclose(frame.image[cell][:, 2] / frame.image[cell][:, 0], 0.25 / 0.95, atol=1e-4))
			else:
				np.testing.assert_array_equal(frame.image[cell], frame.host[cell])
			beside = pixels_in_cell(points, (1, 2), (0, 67, 0)) | pixels_in_cell(points, (1, 2), (0, 66, 1))
			self.assertFalse(frame.hole[beside].any())
			np.testing.assert_array_equal(frame.image[beside], frame.host[beside])

	def test_steve_inside_the_wall_shell_is_hidden(self):
		inside = render.Steve((40.45, 64.0, 0.5, 1.0), (90.0, 90.0, 0.0, 0.0))  # facing -x, in the shell cells
		frame = render.render_frame(self.wall_scene(False, inside))
		self.assertGreater(frame.steve.sum(), 20)
		self.assertTrue((frame.zm[frame.steve] < frame.zh[frame.steve]).all())  # in front of RL's wall...
		self.assertFalse(frame.visible[frame.steve].any())  # ...but in its shell: hidden
		np.testing.assert_array_equal(frame.image[frame.steve], frame.host[frame.steve])
		outside = render.Steve((39.45, 64.0, 0.5, 1.0), (90.0, 90.0, 0.0, 0.0))
		frame = render.render_frame(self.wall_scene(False, outside))
		self.assertGreater(frame.steve.sum(), 20)
		self.assertTrue(frame.visible[frame.steve].all())
		np.testing.assert_allclose(frame.image[frame.steve], frame.world[frame.steve][:, :3])

	def test_unknown_debug_view(self):
		with self.assertRaises(ValueError):
			render.render_frame(self.floor_scene(), debug="nope")

	def test_other_debug_views_run(self):
		scene = self.floor_scene()
		for view in ("host-depth", "vox-depth", "depth-diff"):
			image = render.render_frame(scene, debug=view).image
			self.assertEqual(image.shape, (81, 81, 3))
			self.assertTrue(np.isfinite(image).all())


class SceneFileTest(unittest.TestCase):
	def test_demo_round_trip(self):
		with tempfile.TemporaryDirectory() as tmp:
			render.build_demo(tmp, 64, 36)
			for name in ("vox.r16", "cells.r8", "types.rgba32f", "scene.json", "mcrl_steve.png"):
				self.assertTrue(os.path.isfile(os.path.join(tmp, name)), name)
			self.assertEqual(os.path.getsize(os.path.join(tmp, "vox.r16")), 128 * 5120 * 2)
			self.assertEqual(os.path.getsize(os.path.join(tmp, "types.rgba32f")), 256 * 384 * 16)
			with warnings.catch_warnings():
				warnings.simplefilter("error", render.PreviewWarning)
				scene = render.load_scene(tmp)
			js = json.loads(read_bytes(os.path.join(tmp, "scene.json")).decode("utf-8"))
		self.assertEqual((scene.width, scene.height), (64, 36))
		self.assertEqual(scene.steve.pos, tuple(float(v) for v in js["steve"]["pos"]))
		self.assertEqual(scene.arena, render.Arena())
		np.testing.assert_array_equal(scene.tex.skin, render.plain_skin().astype(F32) / 255.0)
		expected = render.build_demo_textures()
		np.testing.assert_array_equal(scene.tex.vox, expected.vox)
		np.testing.assert_array_equal(scene.tex.cells, expected.cells)
		np.testing.assert_array_equal(scene.tex.types, expected.types)
		layer = scene.tex.cells.reshape(40, 128, 128)[63 - 48]  # [z, x]
		self.assertEqual(int((layer == 240).sum()), 1)
		self.assertEqual(layer[-8 + 64, 0 + 64], 240)
		# the floor's shell covers the pitch (|x| <= 40.96, |z| <= 51.2) and the goals, nothing else
		self.assertEqual(layer[0 + 64, 40 + 64], 120)
		self.assertEqual(layer[0 + 64, 41 + 64], 0)
		self.assertEqual(layer[51 + 64, 20 + 64], 120)
		self.assertEqual(layer[52 + 64, 20 + 64], 0)
		self.assertEqual(layer[59 + 64, 0 + 64], 120)
		self.assertEqual(layer[60 + 64, 0 + 64], 120)  # the goal's back wall is on z = 60: the cells behind it
		self.assertEqual(layer[61 + 64, 0 + 64], 0)
		self.assertEqual(layer[50 + 64, 40 + 64], 0)  # cut off by the corner
		self.assertFalse((scene.tex.cells.reshape(40, 128, 128)[:63 - 48] != 0).any())

	def load_warnings(self, directory):
		with warnings.catch_warnings(record=True) as caught:
			warnings.simplefilter("always")
			scene = render.load_scene(directory)
		return scene, [str(w.message) for w in caught]

	def test_missing_atlas_and_skin(self):
		with tempfile.TemporaryDirectory() as tmp:
			tex = render.Textures.empty()
			render.define_type(tex, 1, (0.5, 0.5, 0.5), slots=(4, 5, 6))
			render.define_type(tex, 2, (0.5, 0.5, 0.5))
			render.set_block(tex, (0, 64, 0), 2)
			render.write_textures(tmp, tex)
			js = {"width": 8, "height": 8, "fovY": 60, "camera": {"pos": [0, 70, 0], "yaw": 0, "pitch": 0}}

			def write(**extra):
				with open(os.path.join(tmp, "scene.json"), "w") as f:
					json.dump(dict(js, **extra), f)

			# nothing named, nothing drawn needs an atlas, Steve not shown: quiet
			write(atlas="")
			scene, messages = self.load_warnings(tmp)
			self.assertEqual(messages, [])
			self.assertEqual(scene.steve, render.Steve())
			self.assertIsNone(scene.tex.atlas)
			# named but missing: warns even when nothing needs it
			write(atlas="nowhere/mcrl_atlas.png", skin="nowhere/mcrl_steve.png")
			scene, messages = self.load_warnings(tmp)
			self.assertEqual(len(messages), 2, messages)
			self.assertIn("nowhere/mcrl_atlas.png", messages[0])
			self.assertIn("untextured", messages[0])
			self.assertIn("nowhere/mcrl_steve.png", messages[1])
			np.testing.assert_array_equal(scene.tex.skin, render.plain_skin().astype(F32) / 255.0)
			# not named but needed: a drawn block with a textured face, Steve shown
			render.set_block(tex, (1, 64, 0), 1)
			render.write_textures(tmp, tex)
			write(steve={"pos": [0, 64, 5, 1]})
			scene, messages = self.load_warnings(tmp)
			self.assertEqual(len(messages), 2, messages)
			self.assertIn("atlas None", messages[0])
			self.assertIn("skin None", messages[1])
			self.assertIsNone(scene.tex.atlas)

	def test_atlas_and_skin_paths(self):
		with tempfile.TemporaryDirectory() as tmp:
			render.write_textures(tmp, render.Textures.empty())
			atlas = np.zeros((1024, 1024, 4), np.uint8)
			atlas[0, 16] = (10, 20, 30, 255)
			Image.fromarray(atlas, "RGBA").save(os.path.join(tmp, "atlas.png"))
			Image.fromarray(np.full((32, 64, 4), 200, np.uint8), "RGBA").save(os.path.join(tmp, "legacy.png"))
			js = {"width": 8, "height": 8, "fovY": 60, "camera": {"pos": [0, 70, 0], "yaw": 0, "pitch": 0},
				"atlas": "atlas.png", "skin": os.path.join(tmp, "legacy.png")}
			with open(os.path.join(tmp, "scene.json"), "w") as f:
				json.dump(js, f)
			with warnings.catch_warnings(record=True) as caught:
				warnings.simplefilter("always")
				scene = render.load_scene(tmp, (16, 9))
		self.assertEqual((scene.width, scene.height), (16, 9))
		np.testing.assert_allclose(scene.tex.atlas[0, 16], np.array([10, 20, 30, 255]) / 255.0)
		self.assertEqual(scene.tex.skin.shape, (64, 64, 4))
		self.assertEqual(len(caught), 1)
		self.assertIn("64 x 32", str(caught[0].message))

	def test_plain_skin_covers_every_face_steve_box_reads(self):
		skin = render.plain_skin()
		self.assertEqual(skin.shape, (64, 64, 4))
		boxes = (((0, 0), 8, 8, 8), ((16, 16), 8, 12, 4), ((40, 16), 4, 12, 4), ((32, 48), 4, 12, 4),
			((0, 16), 4, 12, 4), ((16, 48), 4, 12, 4))
		for (u, v), w, h, d in boxes:
			for x, y, fw, fh in render._box_faces(u, v, w, h, d).values():
				self.assertTrue((skin[y:y + fh, x:x + fw, 3] == 255).all(), (u, v, x, y))


class CommandLineTest(unittest.TestCase):
	def run_main(self, args):
		out, err = io.StringIO(), io.StringIO()
		with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
			code = render.main(args)
		return code, out.getvalue(), err.getvalue()

	def test_demo_and_debug(self):
		with tempfile.TemporaryDirectory() as tmp:
			code, out, err = self.run_main(["--demo", tmp, "--size", "96x54"])
			self.assertEqual(code, 0, err)
			self.assertEqual(err, "")
			self.assertIn("preview.png", out)
			with Image.open(os.path.join(tmp, "preview.png")) as im:
				self.assertEqual(im.size, (96, 54))
			png = os.path.join(tmp, "sub", "cells.png")
			code, out, err = self.run_main([tmp, "--debug", "cells", "--size", "64x36", "--out", png])
			self.assertEqual(code, 0, err)
			with Image.open(png) as im:
				pixels = np.asarray(im.convert("RGB"))
			self.assertEqual(pixels.shape, (36, 64, 3))
			self.assertTrue(((pixels == (255, 38, 26)).all(axis=-1)).any())  # shell (red)

	def test_missing_scene(self):
		with tempfile.TemporaryDirectory() as tmp:
			code, out, err = self.run_main([os.path.join(tmp, "nothing")])
		self.assertEqual(code, 1)
		self.assertIn("scene.json", err)

	def test_bad_scene_json(self):
		with tempfile.TemporaryDirectory() as tmp:
			render.write_textures(tmp, render.Textures.empty())
			good = {"width": 8, "height": 8, "fovY": 60, "camera": {"pos": [0, 70, 0], "yaw": 0, "pitch": 0}}
			for js, message in (({k: v for k, v in good.items() if k != "camera"}, "missing 'camera'"),
					(dict(good, camera={"yaw": 0, "pitch": 0}), "pos"), (dict(good, fovY=180), "fovY"),
					(dict(good, width=0), "positive"), (dict(good, steve={"pos": [1, 2]}), "pos")):
				with open(os.path.join(tmp, "scene.json"), "w") as f:
					json.dump(js, f)
				code, out, err = self.run_main([tmp])
				self.assertEqual(code, 1, js)
				self.assertIn(message, err)

	def test_needs_a_scene_or_demo(self):
		with contextlib.redirect_stderr(io.StringIO()):
			with self.assertRaises(SystemExit):
				render.main([])

	def test_demo_render_is_fast(self):
		with tempfile.TemporaryDirectory() as tmp:
			render.build_demo(tmp, 320, 180)
			scene = render.load_scene(tmp)
			started = time.perf_counter()
			frame = render.render_frame(scene)
			elapsed = time.perf_counter() - started
		self.assertLess(elapsed, 20.0)
		self.assertEqual(frame.image.shape, (180, 320, 3))
		self.assertTrue(frame.hole.any())
		self.assertTrue(frame.steve.any() and frame.visible[frame.steve].all())


if __name__ == "__main__":
	unittest.main()
