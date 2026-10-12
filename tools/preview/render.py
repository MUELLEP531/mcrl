#!/usr/bin/env python3
"""Offline preview of MCPassthrough.fx in voxel mode: draws what the voxel core exports (MCVOX, MCHOLES, MCTYPES, the
atlas and Steve's skin) the way the effect draws it, without Rocket League.

    render.py <scene_dir> [--out file.png] [--debug cells] [--size WxH]
    render.py --demo <out_dir>

The voxel path (vox_at / shell_at / type_row / atlas / block_color, trace_voxels, steve_box / trace_steve and
PS_Composite's voxel branch) is a line-by-line port to numpy, vectorised over pixels and computed in float32 in the
shader's order of operations. RL's picture and depth buffer don't exist offline, so RL's surface is synthesised from
the soccar arena's closed volume. See README.md for the scene format and what is left out.
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

F32 = np.float32

# The grid MCPassthrough.fx hard-codes: cells x in [-64, 64), y in [48, 88), z in [-64, 64) (trace_voxels' lo / hi).
GRID_LO = np.array([-64.0, 48.0, -64.0], dtype=F32)
GRID_HI = np.array([64.0, 88.0, 64.0], dtype=F32)
GRID_MAX_CELL = np.array([127.0, 39.0, 127.0], dtype=F32)
VOX_WIDTH = 128
VOX_HEIGHT = 128 * 40
TYPES_WIDTH = 256
TYPES_HEIGHT = 384
ATLAS_SIZE = 1024
SKIN_SIZE = 64
MAX_STEPS = 320

# The effect's defaults for its depth test (uniforms DepthBias, SlopeBias, MaxBias).
DEPTH_BIAS = 0.04
SLOPE_BIAS = 0.0
MAX_BIAS = 3.0
# trace_steve's own pi, not math.pi.
SHADER_PI = 3.14159265
# What host_linear returns where there is no surface, and the trace functions' "no hit".
FAR = 1e9

# --debug names for the shader's DebugView 1..4.
DEBUG_VIEWS = ("host-depth", "vox-depth", "depth-diff", "cells")

# The synthesised RL surface a pixel shows.
KIND_NONE, KIND_FLOOR, KIND_WALL, KIND_CEILING, KIND_GOAL_ORANGE, KIND_GOAL_BLUE = range(6)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PreviewWarning(UserWarning):
	"""A missing or odd input the preview works around."""


def warn(message):
	warnings.warn(message, PreviewWarning, stacklevel=2)


# ---- textures --------------------------------------------------------------------------------------------------------

@dataclass
class Textures:
	"""The effect's textures as arrays indexed [row, column] (texel (x, y) is [y, x])."""

	vox: np.ndarray    # MCVOX, (5120, 128) uint16: block ids
	cells: np.ndarray  # MCHOLES, (5120, 128) uint8: 0 open, 120 shell, 240 dug
	types: np.ndarray  # MCTYPES, (384, 256, 4) float32
	atlas: object = None  # mcrl_atlas.png, (1024, 1024, 4) float32 in 0..1; None: every face drawn untextured
	skin: object = None   # mcrl_steve.png, (64, 64, 4) float32 in 0..1

	@classmethod
	def empty(cls):
		"""All air, all open; every type untextured (slots -1) and black, as exportTypes writes unknown ids."""
		types = np.zeros((TYPES_HEIGHT, TYPES_WIDTH, 4), dtype=F32)
		types[:128, :, :3] = -1.0
		return cls(np.zeros((VOX_HEIGHT, VOX_WIDTH), np.uint16), np.zeros((VOX_HEIGHT, VOX_WIDTH), np.uint8), types,
			None, plain_skin().astype(F32) / F32(255.0))


def texel_index(x, y, z):
	"""Row-major index (width 128) of grid cell (x, y, z), relative to the grid's corner, in MCVOX / MCHOLES."""
	return x + VOX_WIDTH * (z + 128 * y)


def _grid_cell(world_cell):
	x, y, z = (int(world_cell[i]) - int(GRID_LO[i]) for i in range(3))
	if not (0 <= x < 128 and 0 <= y < 40 and 0 <= z < 128):
		raise ValueError(f"cell {tuple(world_cell)} is outside the grid")
	return x, y, z


def set_block(tex, world_cell, block_id):
	"""Writes a block id into MCVOX at a world cell (Minecraft coordinates)."""
	x, y, z = _grid_cell(world_cell)
	tex.vox.reshape(-1)[texel_index(x, y, z)] = block_id


def set_cell_state(tex, world_cell, state):
	"""Writes an arena state (0 open, 1 shell, 2 dug) into MCHOLES at a world cell, as texel values 0 / 120 / 240."""
	x, y, z = _grid_cell(world_cell)
	tex.cells.reshape(-1)[texel_index(x, y, z)] = (0, 120, 240)[state]


def define_type(tex, block_id, color, slots=(-1, -1, -1), solid=1.0, tint=(1.0, 1.0, 1.0), tint_faces=0):
	"""Writes a block type into MCTYPES' three row blocks (docs/SHADER_CONTRACT.md)."""
	col, row = block_id % 256, block_id // 256
	tex.types[row, col] = (slots[0], slots[1], slots[2], solid)
	tex.types[row + 128, col] = (tint[0], tint[1], tint[2], tint_faces)
	tex.types[row + 256, col] = (color[0], color[1], color[2], 1.0)


def _sample_point(tex, u, v):
	"""tex2Dlod with POINT filtering and CLAMP addressing: texel (floor(u * width), floor(v * height)), clamped."""
	h, w = tex.shape[:2]
	x = np.clip(np.floor(u * F32(w)), 0, w - 1).astype(np.intp)
	y = np.clip(np.floor(v * F32(h)), 0, h - 1).astype(np.intp)
	return tex[y, x]


def _classify(r):
	"""The shader's reading of an MCHOLES texel (r in 0..1): 2 dug, 1 shell, 0 open."""
	return np.where(r > F32(0.7), 2, np.where(r > F32(0.25), 1, 0)).astype(np.int32)


def vox_at(tex, cell):
	"""vox_at(): the block id in grid cells (N, 3) (relative to the corner). R16_UNORM read as round(r * 65535), which
	gives back the stored integer for every 16-bit value."""
	c = cell.astype(F32)
	u = (c[:, 0] + F32(0.5)) / F32(128.0)
	v = (c[:, 2] + F32(128.0) * c[:, 1] + F32(0.5)) / F32(5120.0)
	return _sample_point(tex.vox, u, v).astype(np.int32)


def shell_at(tex, cell):
	"""shell_at(): the arena state of grid cells (N, 3)."""
	c = cell.astype(F32)
	u = (c[:, 0] + F32(0.5)) / F32(128.0)
	v = (c[:, 2] + F32(128.0) * c[:, 1] + F32(0.5)) / F32(5120.0)
	return _classify(_sample_point(tex.cells, u, v).astype(F32) / F32(255.0))


def type_row(tex, block, row):
	"""type_row(): MCTYPES' texel (id % 256, id / 256 + row * 128) for ids >= 0 (HLSL's int / and % truncate toward
	zero; ids from MCVOX are never negative). Rows past the texture clamp, as the sampler does."""
	block = np.asarray(block, dtype=np.int64)
	u = ((block % 256).astype(F32) + F32(0.5)) / F32(256.0)
	v = ((block // 256 + row * 128).astype(F32) + F32(0.5)) / F32(384.0)
	return _sample_point(tex.types, u, v)


def atlas_texel(atlas, slot, uv):
	"""atlas(): the texel of atlas slot `slot` (float) at uv (16 x 16 slots, 64 to a row)."""
	row = np.floor(slot / F32(64.0))
	base_x = (slot - F32(64.0) * row) * F32(16.0)
	base_y = row * F32(16.0)
	fl = np.floor(np.clip(uv, F32(0.0), F32(1.0)) * F32(15.999))
	u = (base_x + fl[:, 0] + F32(0.5)) / F32(1024.0)
	v = (base_y + fl[:, 1] + F32(0.5)) / F32(1024.0)
	return _sample_point(atlas, u, v)


def block_color(tex, block, face, uv, cell, shade):
	"""block_color(): a face's colour (face 0 top, 1 side, 2 bottom) times its light `shade`. Without an atlas texture
	(slot < 0, or no atlas loaded at all) the block's own colour noised per texel; tinted faces times the tint."""
	m = block.shape[0]
	f = type_row(tex, block, 0)
	tn = type_row(tex, block, 1)
	av = type_row(tex, block, 2)
	slot = np.where(face == 0, f[:, 0], np.where(face == 1, f[:, 1], f[:, 2]))
	if tex.atlas is None:
		slot = np.full_like(slot, -1.0)
	c = np.empty((m, 4), dtype=F32)
	textured = slot >= 0
	if textured.any():
		c[textured] = atlas_texel(tex.atlas, slot[textured], uv[textured])
	plain = ~textured
	if plain.any():
		fl = np.floor(np.clip(uv[plain], F32(0.0), F32(1.0)) * F32(15.999))
		cp = cell[plain]
		px_x = fl[:, 0] + (cp[:, 0] * 16 + cp[:, 1] * 7).astype(F32)
		px_y = fl[:, 1] + (cp[:, 2] * 16).astype(F32)
		noise = np.sin(px_x * F32(12.9898) + px_y * F32(78.233)) * F32(43758.5453)
		noise = noise - np.floor(noise)  # frac
		c[plain, :3] = av[plain, :3] * (F32(0.8) + F32(0.35) * noise)[:, None]
		c[plain, 3] = 1.0
	# int(tn.w): truncation toward zero (saturating, NaN -> 0, as D3D's float-to-int conversion)
	w = np.nan_to_num(tn[:, 3].astype(np.float64), nan=0.0)
	bits = np.clip(np.trunc(w), -2.0 ** 31, 2.0 ** 31 - 1).astype(np.int32)
	tinted = ((bits >> face.astype(np.int32)) & 1) != 0
	c[tinted, :3] *= tn[tinted, :3]
	out = np.empty_like(c)
	out[:, :3] = c[:, :3] * shade[:, None]
	out[:, 3] = c[:, 3]
	return out


# ---- trace_voxels ----------------------------------------------------------------------------------------------------

@dataclass
class VoxelHits:
	hit: np.ndarray    # (N,) bool
	t: np.ndarray      # (N,) float32 distance along the ray, FAR where no hit
	color: np.ndarray  # (N, 4) float32, alpha 1 where hit
	face: np.ndarray   # (N,) int8: 0 top, 1 side, 2 bottom; -1 where no hit
	uv: np.ndarray     # (N, 2) float32
	shade: np.ndarray  # (N,) float32: 1.0 top, 0.8 z sides, 0.6 x sides, 0.5 bottom
	cell: np.ndarray   # (N, 3) int32 grid cell (relative to the corner); -1 where no hit


def _face_of(axis, stp, f):
	"""trace_voxels' face, uv and light for the face entered across `axis` stepping `stp`, at fractional point f."""
	m = axis.shape[0]
	face = np.ones(m, dtype=np.int8)
	uv = np.empty((m, 2), dtype=F32)
	shade = np.empty(m, dtype=F32)
	on_y = axis == 1
	on_x = axis == 0
	on_z = ~on_y & ~on_x
	up = stp[:, 1] > 0
	face[on_y] = np.where(up[on_y], 2, 0)
	uv[on_y, 0] = f[on_y, 0]
	uv[on_y, 1] = f[on_y, 2]
	shade[on_y] = np.where(up[on_y], F32(0.5), F32(1.0))
	uv[on_x, 0] = np.where(stp[on_x, 0] > 0, f[on_x, 2], F32(1.0) - f[on_x, 2])
	uv[on_x, 1] = F32(1.0) - f[on_x, 1]
	shade[on_x] = F32(0.6)
	uv[on_z, 0] = np.where(stp[on_z, 2] > 0, F32(1.0) - f[on_z, 0], f[on_z, 0])
	uv[on_z, 1] = F32(1.0) - f[on_z, 1]
	shade[on_z] = F32(0.8)
	return face, uv, shade


def trace_voxels(tex, o, d, t_max, t_surf, hole, record=None):
	"""trace_voxels(): the first block face along world rays (origins o (3,) or (N, 3), unit directions d (N, 3)) up to
	t_max, walking the grid cell by cell as the shader does (float32, at most 320 cells). A block in a shell cell is
	skipped where RL's surface (t_surf along the ray) is less than 1.5 behind the cell's entry, unless the ray looks
	into a hole. The cell holding the origin is never drawn. `record`, if a list, gets (ray indices, cells, t) for the
	cells each step tests, in order."""
	d = np.asarray(d, dtype=F32).reshape(-1, 3)
	n = d.shape[0]
	o = np.broadcast_to(np.asarray(o, dtype=F32), (n, 3))
	t_max = np.broadcast_to(np.asarray(t_max, dtype=F32), (n,))
	t_surf = np.broadcast_to(np.asarray(t_surf, dtype=F32), (n,))
	hole = np.broadcast_to(np.asarray(hole, dtype=bool), (n,))
	out = VoxelHits(np.zeros(n, bool), np.full(n, FAR, F32), np.zeros((n, 4), F32), np.full(n, -1, np.int8),
		np.zeros((n, 2), F32), np.zeros(n, F32), np.full((n, 3), -1, np.int32))

	# dd: components too small to divide by nudged by +1e-6 (so their sign() is +1 and 1 / dd is finite)
	dd = d + (np.abs(d) < F32(1e-6)).astype(F32) * F32(1e-6)
	inv = F32(1.0) / dd
	ta = (GRID_LO - o) * inv
	tb = (GRID_HI - o) * inv
	t0 = np.minimum(ta, tb)
	t1 = np.maximum(ta, tb)
	t_enter = np.maximum(np.maximum(t0[:, 0], t0[:, 1]), t0[:, 2])
	t_end = np.minimum(np.minimum(np.minimum(t1[:, 0], t1[:, 1]), t1[:, 2]), t_max)
	idx = np.nonzero(~(t_end <= np.maximum(t_enter, F32(0.0))))[0]

	te = t_enter[idx]
	t = np.maximum(te, F32(0.0))
	t0i = t0[idx]
	# the face the ray came in by (-1: it starts inside the grid)
	axis = np.where(te > 0, np.where((t0i[:, 0] >= t0i[:, 1]) & (t0i[:, 0] >= t0i[:, 2]), 0,
		np.where(t0i[:, 1] >= t0i[:, 2], 1, 2)), -1).astype(np.int8)
	ddi = dd[idx]
	invi = inv[idx]
	oi = o[idx]
	# int3(floor(...)) then clamp; clamping the float first gives the same cell and never overflows the conversion
	start = np.floor(oi + ddi * (t + F32(1e-4))[:, None] - GRID_LO)
	cell = np.clip(start, F32(0.0), GRID_MAX_CELL).astype(np.int32)
	t_next = (cell.astype(F32) + GRID_LO + (ddi >= 0).astype(F32) - oi) * invi  # step(0, dd) = dd >= 0
	# The walk's state, one array per component. A finished walk stays in the arrays, marked dead, until enough have
	# finished to be worth compacting; dead walks keep stepping but are never read.
	cx, cy, cz = (np.ascontiguousarray(cell[:, a]) for a in range(3))
	tnx, tny, tnz = (np.ascontiguousarray(t_next[:, a]) for a in range(3))
	tdx, tdy, tdz = (np.abs(invi[:, a]) for a in range(3))
	sx, sy, sz = (np.sign(ddi[:, a]).astype(np.int32) for a in range(3))
	state = [idx, t, axis, cx, cy, cz, tnx, tny, tnz, tdx, tdy, tdz, sx, sy, sz, t_end[idx]]
	alive = np.ones(idx.size, dtype=bool)
	vox = tex.vox.reshape(-1)

	for _ in range(MAX_STEPS):
		idx, t, axis, cx, cy, cz, tnx, tny, tnz, tdx, tdy, tdz, sx, sy, sz, t_end_i = state
		# past t_end (checked at the loop's top) or stepped out of the grid (checked right after stepping)
		alive &= ~(t > t_end_i) & (cx >= 0) & (cx <= 127) & (cy >= 0) & (cy <= 39) & (cz >= 0) & (cz <= 127)
		count = np.count_nonzero(alive)
		if count == 0:
			break
		if count < alive.size * 3 // 4:
			state = [a[alive] for a in state]
			idx, t, axis, cx, cy, cz, tnx, tny, tnz, tdx, tdy, tdz, sx, sy, sz, t_end_i = state
			alive = np.ones(count, dtype=bool)
		if record is not None:
			record.append((idx[alive], np.stack([cx, cy, cz], axis=1)[alive], t[alive]))
		# vox_at(): for cells in the grid its texel is exactly texel_index(cell)
		block = vox.take(texel_index(cx, cy, cz), mode="clip").astype(np.int32)
		cand = np.nonzero(alive & (block > 0) & (axis >= 0))[0]
		if cand.size:
			ray = idx[cand]
			cell = np.stack([cx[cand], cy[cand], cz[cand]], axis=1)
			shell = shell_at(tex, cell)
			drawn = ~(~hole[ray] & (shell == 1) & (t_surf[ray] < t[cand] + F32(1.5)))
			cand, ray, cell = cand[drawn], ray[drawn], cell[drawn]
		if cand.size:
			p = o[ray] + dd[ray] * t[cand][:, None]
			stp = np.stack([sx[cand], sy[cand], sz[cand]], axis=1)
			face, uv, shade = _face_of(axis[cand], stp, p - np.floor(p))
			c = block_color(tex, block[cand], face, uv, cell, shade)
			opaque = c[:, 3] > F32(0.5)  # cutout textures: see through the clear texels
			if opaque.any():
				r = ray[opaque]
				out.hit[r] = True
				out.t[r] = t[cand[opaque]]
				out.color[r, :3] = c[opaque, :3]
				out.color[r, 3] = 1.0
				out.face[r] = face[opaque]
				out.uv[r] = uv[opaque]
				out.shade[r] = shade[opaque]
				out.cell[r] = cell[opaque]
				alive[cand[opaque]] = False
		# step into the next cell across the nearest boundary (ties go to z, then y, as the shader's comparisons do)
		on_x = (tnx < tny) & (tnx < tnz)
		on_y = ~on_x & (tny < tnz)
		on_z = ~on_x & ~on_y
		state[1] = np.where(on_x, tnx, np.where(on_y, tny, tnz))
		state[2] = np.where(on_x, 0, np.where(on_y, 1, 2)).astype(np.int8)
		np.add(tnx, tdx, out=tnx, where=on_x)
		np.add(tny, tdy, out=tny, where=on_y)
		np.add(tnz, tdz, out=tnz, where=on_z)
		np.add(cx, sx, out=cx, where=on_x)
		np.add(cy, sy, out=cy, where=on_y)
		np.add(cz, sz, out=cz, where=on_z)
	return out


# ---- trace_steve -----------------------------------------------------------------------------------------------------

@dataclass
class Steve:
	"""The Steve uniforms."""

	pos: tuple = (0.0, 0.0, 0.0, 0.0)    # StevePos: feet x, y, z; w shown (> 0.5)
	look: tuple = (0.0, 0.0, 0.0, 0.0)   # SteveLook: body yaw, head yaw, pitch (degrees); w crouching (not drawn)
	anim: tuple = (0.0, 0.0, 0.0, 0.0)   # SteveAnim: walk position, walk speed, arm swing 0..1
	item: tuple = (-1.0, 0.0, 0.0, 0.0)  # SteveItem: atlas slot, block id (the shader doesn't draw the item)


@dataclass
class SteveHits:
	hit: np.ndarray    # (N,) bool
	t: np.ndarray      # (N,) float32, FAR where no hit
	color: np.ndarray  # (N, 4) float32


def _dot3(a, b):
	return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _steve_box(o, d, lo, hi, pivot, x_rot, y_rot, uv0, skin, t_best, color):
	"""steve_box(): one box of Steve's model (model pixels; o the model-space origin shared by all rays, d (N, 3)),
	turned about `pivot` by x_rot then y_rot. Where it is hit nearer than t_best on an opaque texel, updates t_best and
	color in place."""
	lo = np.asarray(lo, dtype=F32)
	hi = np.asarray(hi, dtype=F32)
	pivot = np.asarray(pivot, dtype=F32)
	# into the box's frame: undo y_rot (about y), then x_rot (about x)
	cy, sy = np.cos(-y_rot), np.sin(-y_rot)
	cx, sx = np.cos(-x_rot), np.sin(-x_rot)
	p = o - pivot
	p = np.array([cy * p[0] + sy * p[2], p[1], -sy * p[0] + cy * p[2]], dtype=F32)
	v = np.stack([cy * d[:, 0] + sy * d[:, 2], d[:, 1], -sy * d[:, 0] + cy * d[:, 2]], axis=1)
	p = np.array([p[0], cx * p[1] - sx * p[2], sx * p[1] + cx * p[2]], dtype=F32)
	v = np.stack([v[:, 0], cx * v[:, 1] - sx * v[:, 2], sx * v[:, 1] + cx * v[:, 2]], axis=1)
	p = p + pivot
	vv = v + (np.abs(v) < F32(1e-6)).astype(F32) * F32(1e-6)
	ta = (lo - p) / vv
	tb = (hi - p) / vv
	t0 = np.minimum(ta, tb)
	t1 = np.maximum(ta, tb)
	tn = np.maximum(np.maximum(t0[:, 0], t0[:, 1]), t0[:, 2])
	tf = np.minimum(np.minimum(t1[:, 0], t1[:, 1]), t1[:, 2])
	k = np.nonzero(~((tn > tf) | (tn <= 0) | (tn >= t_best)))[0]
	if k.size == 0:
		return
	tn = tn[k]
	t0 = t0[k]
	vv = vv[k]
	h = p + vv * tn[:, None]
	w, dp = hi[0] - lo[0], hi[2] - lo[2]
	u = np.empty(k.size, dtype=F32)
	v = np.empty(k.size, dtype=F32)
	shade = np.empty(k.size, dtype=F32)
	on_y = tn == t0[:, 1]
	on_x = ~on_y & (tn == t0[:, 0])
	on_z = ~on_y & ~on_x
	top = vv[:, 1] < 0
	u[on_y] = (uv0[0] + (dp + np.where(top, F32(0.0), w) + (h[:, 0] - lo[0])))[on_y]
	v[on_y] = (uv0[1] + (h[:, 2] - lo[2]))[on_y]
	shade[on_y] = np.where(top, F32(1.0), F32(0.5))[on_y]
	left = vv[:, 0] < 0  # hit his left side (+x)
	u[on_x] = (uv0[0] + np.where(left, dp + w + (hi[2] - h[:, 2]), h[:, 2] - lo[2]))[on_x]
	v[on_x] = (uv0[1] + (dp + (hi[1] - h[:, 1])))[on_x]
	shade[on_x] = F32(0.6)
	front = vv[:, 2] < 0
	u[on_z] = (uv0[0] + np.where(front, dp + (h[:, 0] - lo[0]), F32(2.0) * dp + w + (hi[0] - h[:, 0])))[on_z]
	v[on_z] = (uv0[1] + (dp + (hi[1] - h[:, 1])))[on_z]
	shade[on_z] = F32(0.8)
	c = _sample_point(skin, (np.floor(u) + F32(0.5)) / F32(64.0), (np.floor(v) + F32(0.5)) / F32(64.0))
	opaque = ~(c[:, 3] < F32(0.5))
	k = k[opaque]
	t_best[k] = tn[opaque]
	color[k, :3] = c[opaque, :3] * shade[opaque, None]
	color[k, 3] = 1.0


def trace_steve(skin, steve, o, d):
	"""trace_steve(): Steve (Minecraft's player model with its walk and arm swing) along world rays (origin o, unit
	directions d (N, 3)): distance and colour where hit."""
	d = np.asarray(d, dtype=F32).reshape(-1, 3)
	n = d.shape[0]
	hits = SteveHits(np.zeros(n, bool), np.full(n, FAR, F32), np.zeros((n, 4), F32))
	pos = np.asarray(steve.pos, dtype=F32)
	look = np.asarray(steve.look, dtype=F32)
	anim = np.asarray(steve.anim, dtype=F32)
	if pos[3] < 0.5:
		return hits
	if skin is None:
		skin = plain_skin().astype(F32) / F32(255.0)
	# model space: 1 unit = one model pixel (1/16 block, scaled 0.9375 as Minecraft draws players)
	s = F32(0.9375) / F32(16.0)
	d2r = F32(SHADER_PI) / F32(180.0)
	yaw = look[0] * d2r
	front = np.array([-np.sin(yaw), 0.0, np.cos(yaw)], dtype=F32)
	left = np.array([np.cos(yaw), 0.0, np.sin(yaw)], dtype=F32)
	q = np.asarray(o, dtype=F32) - pos[:3]
	om = np.array([_dot3(q, left), q[1], _dot3(q, front)], dtype=F32) / s
	dm = np.stack([_dot3(d, left), d[:, 1], _dot3(d, front)], axis=1) / s
	# quick reject: the ray must pass within his height and reach
	dxz = dm[:, 0] * dm[:, 0] + dm[:, 2] * dm[:, 2]
	tc = np.maximum(-(om[0] * dm[:, 0] + om[2] * dm[:, 2]) / np.maximum(dxz, F32(1e-9)), F32(0.0))
	cx = om[0] + dm[:, 0] * tc
	cz = om[2] + dm[:, 2] * tc
	near = np.nonzero(~(np.sqrt(cx * cx + cz * cz) > F32(12.0)))[0]
	if near.size == 0:
		return hits
	dm = dm[near]
	swing, amount, pi = anim[0] * F32(0.6662), anim[1], F32(SHADER_PI)
	leg_r = np.cos(swing) * F32(1.4) * amount
	leg_l = np.cos(swing + pi) * F32(1.4) * amount
	arm_r = np.cos(swing + pi) * amount - np.sin(anim[2] * pi) * F32(1.4)
	arm_l = np.cos(swing) * amount
	head_yaw = (look[1] - look[0]) * d2r
	head_pitch = look[2] * d2r
	zero = F32(0.0)
	t = np.full(near.size, F32(1e9))
	c = np.zeros((near.size, 4), dtype=F32)
	_steve_box(om, dm, (-4, 24, -4), (4, 32, 4), (0, 24, 0), -head_pitch, -head_yaw, (0, 0), skin, t, c)  # head
	_steve_box(om, dm, (-4, 12, -2), (4, 24, 2), (0, 24, 0), zero, zero, (16, 16), skin, t, c)            # body
	_steve_box(om, dm, (-8, 12, -2), (-4, 24, 2), (-5, 22, 0), arm_r, zero, (40, 16), skin, t, c)         # right arm
	_steve_box(om, dm, (4, 12, -2), (8, 24, 2), (5, 22, 0), arm_l, zero, (32, 48), skin, t, c)            # left arm
	_steve_box(om, dm, (-4, 0, -2), (0, 12, 2), (-2, 12, 0), leg_r, zero, (0, 16), skin, t, c)            # right leg
	_steve_box(om, dm, (0, 0, -2), (4, 12, 2), (2, 12, 0), leg_l, zero, (16, 48), skin, t, c)             # left leg
	got = ~(t > F32(1e8))
	r = near[got]
	hits.hit[r] = True
	hits.t[r] = t[got]
	hits.color[r] = c[got]
	return hits


# ---- the synthesised RL surface --------------------------------------------------------------------------------------

@dataclass
class Arena:
	"""rl::ArenaMapping + rl::SoccarDimensions (uu): x_rl = x * uu, y_rl = z * uu, z_rl = (y - floorY) * uu."""

	uu_per_block: float = 100.0
	floor_y: float = 64.0
	half_width: float = 4096.0
	half_length: float = 5120.0
	height: float = 2044.0
	corner_sum: float = 8064.0
	goal_half_width: float = 893.0
	goal_height: float = 642.775
	goal_depth: float = 880.0

	KEYS = (("uu_per_block", "uuPerBlock"), ("floor_y", "floorY"), ("half_width", "halfWidth"),
		("half_length", "halfLength"), ("height", "height"), ("corner_sum", "cornerSum"),
		("goal_half_width", "goalHalfWidth"), ("goal_height", "goalHeight"), ("goal_depth", "goalDepth"))

	@classmethod
	def from_json(cls, js):
		arena = cls()
		for attr, key in cls.KEYS:
			if key in js:
				setattr(arena, attr, float(js[key]))
		return arena

	def to_json(self):
		return {key: getattr(self, attr) for attr, key in self.KEYS}

	def to_rl(self, p):
		p = np.asarray(p, dtype=np.float64)
		uu = self.uu_per_block
		return np.stack([p[..., 0] * uu, p[..., 2] * uu, (p[..., 1] - self.floor_y) * uu], axis=-1)

	def to_world(self, q):
		q = np.asarray(q, dtype=np.float64)
		uu = self.uu_per_block
		return np.stack([q[..., 0] / uu, q[..., 2] / uu + self.floor_y, q[..., 1] / uu], axis=-1)

	def parts(self):
		"""The closed playable volume as convex parts (the pitch with its 45-degree corners, then the orange goal at
		+y and the blue one at -y), each a list of half-spaces (outward normal, c, surface kind): normal . q <= c."""
		hw, hl, h, cs = self.half_width, self.half_length, self.height, self.corner_sum
		pitch = [((1, 0, 0), hw, KIND_WALL), ((-1, 0, 0), hw, KIND_WALL), ((0, 1, 0), hl, KIND_WALL),
			((0, -1, 0), hl, KIND_WALL), ((0, 0, 1), h, KIND_CEILING), ((0, 0, -1), 0.0, KIND_FLOOR),
			((1, 1, 0), cs, KIND_WALL), ((1, -1, 0), cs, KIND_WALL), ((-1, 1, 0), cs, KIND_WALL),
			((-1, -1, 0), cs, KIND_WALL)]
		parts = [pitch]
		for sign, kind in ((1, KIND_GOAL_ORANGE), (-1, KIND_GOAL_BLUE)):
			parts.append([((1, 0, 0), self.goal_half_width, kind), ((-1, 0, 0), self.goal_half_width, kind),
				((0, sign, 0), hl + self.goal_depth, kind), ((0, -sign, 0), -hl, kind),
				((0, 0, 1), self.goal_height, kind), ((0, 0, -1), 0.0, KIND_FLOOR)])
		return parts

	def contains_rl(self, q, eps=1e-6):
		"""Whether RL points (..., 3) are in the closed volume (rl::insideSoccar, without the curved ramps)."""
		q = np.asarray(q, dtype=np.float64)
		inside = np.zeros(q.shape[:-1], dtype=bool)
		for planes in self.parts():
			part = np.ones(q.shape[:-1], dtype=bool)
			for normal, c, _ in planes:
				part &= q @ np.asarray(normal, dtype=np.float64) <= c + eps
			inside |= part
		return inside


def arena_surface(arena, origin, directions, chunk=1 << 16):
	"""Where world rays (origin, unit directions (N, 3)) meet RL's surface, the boundary of the closed arena volume:
	where they leave it when the origin is inside, where they first enter it otherwise. Returns the distance (inf if
	never), the surface kind and the surface's unit world normal on the origin's side."""
	o = arena.to_rl(origin)
	directions = np.asarray(directions, dtype=np.float64)
	n = directions.shape[0]
	parts = arena.parts()
	planes = [(i, normal, c, kind) for i, part in enumerate(parts) for normal, c, kind in part]
	part_of = np.array([p[0] for p in planes])
	normals = np.array([p[1] for p in planes], dtype=np.float64)
	kinds = np.array([p[3] for p in planes], dtype=np.int8)
	num = np.array([p[2] for p in planes], dtype=np.float64) - normals @ o
	unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
	unit = np.stack([unit[:, 0], unit[:, 2], unit[:, 1]], axis=1)  # RL (x, y, z) -> world (x, z, y)
	columns = [np.nonzero(part_of == i)[0] for i in range(len(parts))]
	t_out = np.empty(n)
	plane_out = np.zeros(n, dtype=np.intp)
	inside_out = np.zeros(n, dtype=bool)
	eps = 1e-6
	for s0 in range(0, n, chunk):
		d = directions[s0:s0 + chunk]
		m = d.shape[0]
		rows = np.arange(m)
		den = np.stack([d[:, 0], d[:, 2], d[:, 1]], axis=1) * arena.uu_per_block @ normals.T  # t in world units
		with np.errstate(divide="ignore", invalid="ignore"):
			tk = num / den
		leaving = np.where(den > 0, tk, np.inf)
		entering = np.where(den < 0, tk, -np.inf)
		blocked = (den == 0) & (num < 0)  # parallel to a plane, on its outside
		enter, leave, enter_k, leave_k, valid = [], [], [], [], []
		for cols in columns:
			bk = cols[leaving[:, cols].argmin(axis=1)]
			ak = cols[entering[:, cols].argmax(axis=1)]
			b = leaving[rows, bk]
			a = entering[rows, ak]
			enter.append(a)
			leave.append(b)
			enter_k.append(ak)
			leave_k.append(bk)
			valid.append(~blocked[:, cols].any(axis=1) & (a <= b))
		enter, leave, valid = np.array(enter), np.array(leave), np.array(valid)
		enter_k, leave_k = np.array(enter_k), np.array(leave_k)
		# origin inside: follow the union of the parts' intervals from t = 0 to where it ends
		inside = valid & (enter <= eps) & (leave >= -eps)
		origin_in = inside.any(axis=0)
		end = np.where(inside, leave, -np.inf)
		part = end.argmax(axis=0)
		end = end[part, rows]
		for _ in range(len(parts)):
			for i in range(len(parts)):
				ext = origin_in & valid[i] & (enter[i] <= end + eps) & (leave[i] > end)
				end[ext] = leave[i][ext]
				part[ext] = i
		# origin outside: the first entry
		entries = np.where(valid & (enter > 0), enter, np.inf)
		epart = entries.argmin(axis=0)
		t_out[s0:s0 + m] = np.where(origin_in, end, entries[epart, rows])
		plane_out[s0:s0 + m] = np.where(origin_in, leave_k[part, rows], enter_k[epart, rows])
		inside_out[s0:s0 + m] = origin_in
	found = np.isfinite(t_out)
	kind = np.where(found, kinds[plane_out], KIND_NONE).astype(np.int8)
	normal = unit[plane_out] * np.where(inside_out, -1.0, 1.0)[:, None]
	return np.where(found, t_out, np.inf), kind, normal


def host_picture(arena, origin, directions, t, kind, normal):
	"""A plain stand-in for RL's picture: green pitch with stripes and the halfway line and centre circle, blue-grey
	walls, team-coloured goals, dark ceiling; lit by one fixed light and faded with distance."""
	n = kind.shape[0]
	colors = np.array([(0.05, 0.06, 0.09), (0.22, 0.42, 0.18), (0.40, 0.45, 0.53), (0.10, 0.11, 0.14),
		(0.70, 0.45, 0.22), (0.22, 0.40, 0.72)])
	c = colors[kind]
	tt = np.where(np.isfinite(t), t, 0.0)
	q = arena.to_rl(np.asarray(origin, dtype=np.float64)[None, :] + np.asarray(directions, np.float64) * tt[:, None])
	floor = kind == KIND_FLOOR
	stripe = np.where(np.floor(q[:, 1] / 1024.0) % 2 == 0, 1.0, 0.9)
	line = (np.abs(q[:, 1]) < 25.0) | (np.abs(np.hypot(q[:, 0], q[:, 1]) - 915.0) < 25.0)
	c = np.where(floor[:, None], c * stripe[:, None], c)
	c = np.where((floor & line)[:, None], np.array([0.80, 0.84, 0.80]), c)
	light = np.array([0.35, 1.0, 0.25])
	light /= np.linalg.norm(light)
	c = c * (0.55 + 0.45 * np.clip(normal @ light, 0.0, 1.0))[:, None]
	fade = 0.3 * (1.0 - np.exp(-tt / 250.0))
	c = c * (1.0 - fade)[:, None] + np.array([0.55, 0.60, 0.68]) * fade[:, None]
	c[kind == KIND_NONE] = colors[KIND_NONE]
	return np.clip(c, 0.0, 1.0).astype(F32).reshape(n, 3)


# ---- PS_Composite (voxel branch) -------------------------------------------------------------------------------------

def camera_rows(yaw, pitch):
	"""HostCamRow0..2: rows of the camera-to-world rotation for Minecraft's yaw / pitch in degrees (yaw 0 looks toward
	+z, yaw 90 toward -x, pitch > 0 looks down). Camera space: x right, y up, looking down -z."""
	y, p = math.radians(yaw), math.radians(pitch)
	forward = np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])
	right = np.array([-math.cos(y), 0.0, -math.sin(y)])
	up = np.cross(right, forward)
	return np.column_stack([right, up, -forward]).astype(F32)


def pixel_rays(width, height, fov_y):
	"""Each pixel's view ray in camera space, (H, W, 3): ((u * 2 - 1) * tanHalfFovY * aspect, (1 - v * 2) *
	tanHalfFovY, -1) at the pixel's centre, as PS_Composite builds it (aspect = BUFFER_WIDTH * BUFFER_RCP_HEIGHT)."""
	u = (np.arange(width, dtype=F32) + F32(0.5)) / F32(width)
	v = (np.arange(height, dtype=F32) + F32(0.5)) / F32(height)
	tan_half = F32(math.tan(math.radians(fov_y) / 2.0))
	rx = (u * F32(2.0) - F32(1.0)) * tan_half * F32(width) * F32(1.0 / height)
	ry = (F32(1.0) - v * F32(2.0)) * tan_half
	ray = np.empty((height, width, 3), dtype=F32)
	ray[:, :, 0] = rx[None, :]
	ray[:, :, 1] = ry[:, None]
	ray[:, :, 2] = -1.0
	return ray


def world_of(rows, cam_pos, p):
	"""world_of() in voxel mode: camera-space points (N, 3) in the world."""
	return np.stack([_dot3(rows[0], p), _dot3(rows[1], p), _dot3(rows[2], p)], axis=1) + cam_pos


def arena_cell(tex, rows, cam_pos, p):
	"""arena_cell(): the arena state (0 open, 1 shell, 2 dug) of the cells holding camera-space points (N, 3)."""
	c = np.floor(world_of(rows, cam_pos, p)) - GRID_LO
	outside = (c < 0).any(axis=1) | (c[:, 0] >= 128) | (c[:, 1] >= 40) | (c[:, 2] >= 128)
	c = np.where(outside[:, None], F32(0.0), c)
	u = (c[:, 0] + F32(0.5)) / F32(128.0)
	v = (c[:, 2] + F32(128.0) * c[:, 1] + F32(0.5)) / F32(5120.0)
	state = _classify(_sample_point(tex.cells, u, v).astype(F32) / F32(255.0))
	return np.where(outside, 0, state)


def _ddy(z):
	"""ddy() per 2 x 2 quad (fine): the lower row minus the upper one, for both rows. Unpaired last row: 0."""
	out = np.zeros_like(z)
	h = z.shape[0] // 2 * 2
	diff = z[1:h:2] - z[0:h:2]
	out[0:h:2] = diff
	out[1:h:2] = diff
	return out


def _bands(z):
	"""bands(): 1 m depth bands for the debug views."""
	t = z - np.floor(z)
	c = np.where((t >= 0.5)[:, None], np.array([1.0, 0.85, 0.3], F32), np.array([0.1, 0.1, 0.1], F32))
	return c * np.clip(F32(1.5) - z / F32(100.0), 0.0, 1.0)[:, None]


@dataclass
class Scene:
	width: int
	height: int
	fov_y: float
	cam_pos: tuple
	cam_yaw: float
	cam_pitch: float
	tex: Textures
	steve: Steve = field(default_factory=Steve)
	arena: Arena = field(default_factory=Arena)


@dataclass
class Frame:
	"""A rendered frame and the composite's per-pixel inputs, all (H, W, ...)."""

	image: np.ndarray    # float32 RGB: the composite, or the debug view asked for
	host: np.ndarray     # the synthesised RL picture
	zh: np.ndarray       # RL's linear depth (FAR: no surface)
	zm: np.ndarray       # the voxel world's linear depth (FAR: nothing hit)
	hole: np.ndarray     # bool: the pixel looks into a dug cell
	visible: np.ndarray  # bool: the voxel world shows here
	world: np.ndarray    # RGBA the voxel world drew (alpha 1 where hit)
	steve: np.ndarray    # bool: what the voxel world drew is Steve


def render_frame(scene, debug=None):
	"""PS_Composite in voxel mode for every pixel, with the synthesised RL surface standing in for RL's back buffer and
	depth buffer. `debug`: one of DEBUG_VIEWS for the shader's DebugView 1..4."""
	if debug is not None and debug not in DEBUG_VIEWS:
		raise ValueError(f"unknown debug view {debug!r} (one of {', '.join(DEBUG_VIEWS)})")
	w, h, tex = scene.width, scene.height, scene.tex
	ray = pixel_rays(w, h, scene.fov_y).reshape(-1, 3)
	rows = camera_rows(scene.cam_yaw, scene.cam_pitch)
	cam = np.asarray(scene.cam_pos, dtype=F32)
	ray_len = np.sqrt(_dot3(ray, ray))
	d = np.stack([_dot3(rows[0], ray), _dot3(rows[1], ray), _dot3(rows[2], ray)], axis=1) / ray_len[:, None]

	# RL's depth buffer and picture, synthesised: linear depth = distance along the unit ray / |ray|
	t_rl, kind, normal = arena_surface(scene.arena, cam.astype(np.float64), d.astype(np.float64))
	zh = np.where(np.isfinite(t_rl), t_rl / ray_len, FAR).astype(F32)
	host = host_picture(scene.arena, cam.astype(np.float64), d.astype(np.float64), t_rl, kind, normal)

	ddy = _ddy(zh.reshape(h, w)).reshape(-1)
	allow = np.minimum(F32(DEPTH_BIAS) + F32(SLOPE_BIAS) * np.abs(ddy), F32(max(MAX_BIAS, DEPTH_BIAS)))
	# RL's surface just behind itself (the floor) / just in front (walls inside their cell): in a dug cell?
	behind = arena_cell(tex, rows, cam, ray * (zh + np.maximum(F32(0.03), zh * F32(0.01)))[:, None])
	front = arena_cell(tex, rows, cam, ray * np.maximum(zh - np.maximum(F32(0.02), zh * F32(0.005)), F32(0.0))[:, None])
	hole = (zh < F32(400.0)) & ((behind == 2) | (front == 2))

	t_surf = zh * ray_len
	t_max = np.where(hole, t_surf + F32(30.0), np.minimum(np.maximum(zh + allow, F32(0.5)), F32(400.0)) * ray_len)
	vh = trace_voxels(tex, cam, d, t_max, t_surf, hole)
	sh = trace_steve(tex.skin, scene.steve, cam, d)
	hit_s = sh.hit & (sh.t < t_max)
	inside = vh.hit | hit_s
	use_s = hit_s & (~vh.hit | (sh.t < vh.t))
	world = np.where(inside[:, None], np.where(use_s[:, None], sh.color, vh.color), F32(0.0))
	zm = np.where(inside, np.minimum(np.where(hit_s, sh.t, F32(FAR)), np.where(vh.hit, vh.t, F32(FAR))) / ray_len,
		F32(FAR)).astype(F32)

	visible = hole | (zm < zh + allow)
	# voxels inside the arena's shell, at most 1.5 m in front of RL's surface, are behind it (no hole: hidden)
	shell = arena_cell(tex, rows, cam, ray * (zm + F32(0.03))[:, None]) == 1
	visible &= ~(~hole & (zm < F32(1e8)) & (zh < zm + F32(1.5)) & shell)
	cover = world[:, 3] * visible
	# relight / grade / haze, bloom and PS_Final's effects are left out (README.md): lit = world.rgb
	image = world[:, :3] * visible[:, None] + host * (F32(1.0) - cover)[:, None]
	if debug == "host-depth":
		image = _bands(zh)
	elif debug == "vox-depth":
		image = np.where((world[:, 3] > 0)[:, None], _bands(zm), host * F32(0.3))
	elif debug == "depth-diff":
		diff = zh - zm
		dc = np.stack([np.clip(diff * F32(0.5) + F32(0.5), 0, 1), np.clip(-diff * F32(0.5) + F32(0.5), 0, 1),
			np.zeros_like(diff)], axis=1)
		image = np.where((world[:, 3] > 0)[:, None], dc, host * F32(0.3))
	elif debug == "cells":
		c = np.maximum(behind, front)
		image = np.where((c == 2)[:, None], np.array([0.1, 1.0, 0.1], F32),
			np.where((c == 1)[:, None], np.array([1.0, 0.15, 0.1], F32), host * F32(0.3)))

	def grid(a, *rest):
		return a.reshape((h, w) + rest)

	return Frame(grid(image.astype(F32), 3), grid(host, 3), grid(zh), grid(zm), grid(hole), grid(visible),
		grid(world, 4), grid(inside & use_s))


def to_rgb8(image):
	"""The finished picture as 8 bits (PS_Final's saturate, UNORM rounding)."""
	return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---- scene files -----------------------------------------------------------------------------------------------------

def _read_raw(path, dtype, shape):
	expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
	with open(path, "rb") as f:
		data = f.read()
	if len(data) != expected:
		raise ValueError(f"{path}: {len(data)} bytes, expected {expected} ({' x '.join(map(str, shape))} {dtype})")
	return np.frombuffer(data, dtype=dtype).reshape(shape).astype(np.dtype(dtype).newbyteorder("="))


def load_textures(directory):
	"""MCVOX / MCHOLES / MCTYPES from a scene directory (little-endian raw files); no atlas, the plain skin."""
	return Textures(
		_read_raw(os.path.join(directory, "vox.r16"), "<u2", (VOX_HEIGHT, VOX_WIDTH)),
		_read_raw(os.path.join(directory, "cells.r8"), "u1", (VOX_HEIGHT, VOX_WIDTH)),
		_read_raw(os.path.join(directory, "types.rgba32f"), "<f4", (TYPES_HEIGHT, TYPES_WIDTH, 4)),
		None, plain_skin().astype(F32) / F32(255.0))


def write_textures(directory, tex):
	os.makedirs(directory, exist_ok=True)
	with open(os.path.join(directory, "vox.r16"), "wb") as f:
		f.write(np.ascontiguousarray(tex.vox, dtype="<u2").tobytes())
	with open(os.path.join(directory, "cells.r8"), "wb") as f:
		f.write(np.ascontiguousarray(tex.cells, dtype=np.uint8).tobytes())
	with open(os.path.join(directory, "types.rgba32f"), "wb") as f:
		f.write(np.ascontiguousarray(tex.types, dtype="<f4").tobytes())


def resolve_asset(path, scene_dir):
	"""An atlas / skin path from scene.json: absolute, or relative to the scene directory, the current directory or the
	repository root (the exporter writes paths like assets/mcrl_atlas.png). None if missing."""
	if not path:
		return None
	candidates = [path] if os.path.isabs(path) else [os.path.join(scene_dir, path), path, os.path.join(REPO_ROOT, path)]
	for candidate in candidates:
		if os.path.isfile(candidate):
			return candidate
	return None


def load_rgba(path, size, what):
	"""An RGBA8 file texture as float32 0..1. ReShade resizes a file to the texture's declared size; so does this."""
	img = Image.open(path).convert("RGBA")
	if img.size != (size, size):
		warn(f"{what} {path} is {img.size[0]} x {img.size[1]}, not {size} x {size}: resized (the shader expects "
			f"{size} x {size})")
		img = img.resize((size, size), Image.NEAREST)
	return np.asarray(img).astype(F32) / F32(255.0)


def textured_faces_used(tex):
	"""Whether any block in MCVOX has a face with an atlas slot (so the atlas matters)."""
	ids = np.unique(tex.vox)
	ids = ids[ids > 0].astype(np.int64)
	if ids.size == 0:
		return False
	slots = tex.types[np.minimum(ids // 256, TYPES_HEIGHT - 1), ids % 256, :3]
	return bool((slots >= 0).any())


def _vec(js, key, n, default):
	value = js.get(key, default)
	if not isinstance(value, (list, tuple)) or len(value) != n:
		raise ValueError(f"scene.json: {key} must be a list of {n} numbers")
	return tuple(float(x) for x in value)


def load_scene(directory, size=None):
	"""A scene directory: vox.r16, cells.r8, types.rgba32f and scene.json (README.md). `size`: (width, height) to
	render at instead of the scene's."""
	with open(os.path.join(directory, "scene.json"), "r", encoding="utf-8") as f:
		js = json.load(f)
	tex = load_textures(directory)
	sjs = js.get("steve", {})
	steve = Steve(_vec(sjs, "pos", 4, (0, 0, 0, 0)), _vec(sjs, "look", 4, (0, 0, 0, 0)),
		_vec(sjs, "anim", 4, (0, 0, 0, 0)), _vec(sjs, "item", 4, (-1, 0, 0, 0)))
	# a named file that is missing always warns; no file named only where something drawn needs it
	name = js.get("atlas")
	atlas = resolve_asset(name, directory)
	if atlas:
		tex.atlas = load_rgba(atlas, ATLAS_SIZE, "atlas")
	elif name or textured_faces_used(tex):
		warn(f"atlas {name!r} not found: drawing every block face untextured (its own colour)")
	name = js.get("skin")
	skin = resolve_asset(name, directory)
	if skin:
		tex.skin = load_rgba(skin, SKIN_SIZE, "skin")
	elif name or steve.pos[3] >= 0.5:
		warn(f"skin {name!r} not found: drawing Steve with a plain built-in skin")
	try:
		cam = js["camera"]
		width, height = (int(js["width"]), int(js["height"])) if size is None else size
		fov_y, yaw, pitch = float(js["fovY"]), float(cam["yaw"]), float(cam["pitch"])
	except KeyError as e:
		raise ValueError(f"scene.json: missing {e.args[0]!r}") from None
	if width < 1 or height < 1:
		raise ValueError("scene.json: width and height must be positive")
	if not 0.0 < fov_y < 180.0:
		raise ValueError("scene.json: fovY must be between 0 and 180 degrees")
	return Scene(width, height, fov_y, _vec(cam, "pos", 3, None), yaw, pitch, tex, steve,
		Arena.from_json(js.get("arena", {})))


# ---- the plain skin and the demo scene -------------------------------------------------------------------------------

def _box_faces(u, v, w, h, d):
	"""Where steve_box() samples a box's faces in the skin: Minecraft's layout, (x, y, width, height) per face."""
	return {"top": (u + d, v, w, d), "bottom": (u + d + w, v, w, d), "right": (u, v + d, d, h),
		"front": (u + d, v + d, w, h), "left": (u + d + w, v + d, d, h), "back": (u + 2 * d + w, v + d, w, h)}


def plain_skin():
	"""A stand-in Steve skin (modern 64 x 64 layout, RGBA8): a face on the head's front, hair on top and at the back,
	a shirt with short sleeves, trousers and shoes, so the preview still shows which way he faces."""
	img = np.zeros((SKIN_SIZE, SKIN_SIZE, 4), dtype=np.uint8)
	skin, hair, shirt = (196, 140, 110, 255), (70, 45, 25, 255), (0, 160, 165, 255)
	trousers, shoes = (55, 60, 160, 255), (90, 90, 95, 255)

	def fill(rect, color, rows=None):
		x, y, w, h = rect
		r0, r1 = (0, h) if rows is None else rows
		img[y + r0:y + r1, x:x + w] = color

	head = _box_faces(0, 0, 8, 8, 8)
	for name, rect in head.items():
		fill(rect, skin)
		if name in ("right", "left", "front"):
			fill(rect, hair, (0, 2))
	fill(head["top"], hair)
	fill(head["back"], hair)
	x, y = head["front"][:2]
	img[y + 4, x + 1] = img[y + 4, x + 6] = (240, 240, 240, 255)  # eyes (his right eye is on the viewer's left)
	img[y + 4, x + 2] = img[y + 4, x + 5] = (60, 70, 160, 255)
	img[y + 6, x + 3:x + 5] = (130, 70, 60, 255)
	for name, rect in _box_faces(16, 16, 8, 12, 4).items():
		fill(rect, shirt)
	for u, v in ((40, 16), (32, 48)):  # arms
		for name, rect in _box_faces(u, v, 4, 12, 4).items():
			fill(rect, skin if name == "bottom" else shirt)
			if name not in ("top", "bottom"):
				fill(rect, skin, (4, 12))
	for u, v in ((0, 16), (16, 48)):  # legs
		for name, rect in _box_faces(u, v, 4, 12, 4).items():
			fill(rect, shoes if name == "bottom" else trousers)
			if name not in ("top", "bottom"):
				fill(rect, shoes, (9, 12))
	return img


# id, name, own colour, tint, tinted faces (bit 0 top, 1 side, 2 bottom). No atlas slots: drawn untextured.
DEMO_BLOCKS = (
	(1, "dirt", (0.53, 0.38, 0.26), (1.0, 1.0, 1.0), 0),
	(2, "stone", (0.50, 0.50, 0.50), (1.0, 1.0, 1.0), 0),
	(3, "grass_block", (0.58, 0.47, 0.33), (0.62, 1.0, 0.45), 1),
	(4, "red_wool", (0.72, 0.16, 0.14), (1.0, 1.0, 1.0), 0),
	(5, "gold_block", (0.96, 0.80, 0.26), (1.0, 1.0, 1.0), 0),
	(6, "oak_planks", (0.66, 0.52, 0.32), (1.0, 1.0, 1.0), 0),
	(7, "cobblestone", (0.42, 0.42, 0.43), (1.0, 1.0, 1.0), 0),
)


def build_demo_textures(arena=None):
	"""The demo world: RL's floor as a shell layer at y = floorY - 1 over the pitch and goals (invisible: id 0, as
	the exporter writes the barrier), a few blocks on the floor, and one dug cell with dirt under it and dirt in the
	shell cells around it (the hole's sides, hidden behind RL's floor everywhere else)."""
	arena = arena or Arena()
	tex = Textures.empty()
	for block_id, _, color, tint, faces in DEMO_BLOCKS:
		define_type(tex, block_id, color, tint=tint, tint_faces=faces)
	dirt, stone, grass, wool, gold, planks, cobble = range(1, 8)
	floor = int(arena.floor_y)
	# a floor cell is shell where any of its top corners is on RL's floor (the arena generator's rule)
	x = np.arange(-64, 64)
	cx, cz = np.meshgrid(np.arange(-64, 65), np.arange(-64, 65), indexing="ij")
	corners = arena.contains_rl(arena.to_rl(np.stack([cx, np.full(cx.shape, float(floor)), cz], axis=-1)))
	shell = corners[:-1, :-1] | corners[1:, :-1] | corners[:-1, 1:] | corners[1:, 1:]  # [x, z]
	gy = floor - 1 - int(GRID_LO[1])
	rows = np.arange(128) + 128 * gy  # texel rows of the layer, by z
	tex.cells[rows[:, None], (x + 64)[None, :]] = np.where(shell.T, 120, 0).astype(np.uint8)

	for y in range(floor, floor + 3):
		set_block(tex, (-4, y, -3), stone)
	for bx in range(1, 4):
		for bz in range(-3, 0):
			set_block(tex, (bx, floor, bz), grass)
	set_block(tex, (2, floor + 1, -2), wool)
	set_block(tex, (3, floor + 1, -1), gold)
	for bx in range(-3, 3):
		for y in range(floor, floor + 2):
			set_block(tex, (bx, y, 4), cobble if bx in (-3, 2) else planks)
	# the hole: one dug cell in RL's floor, dirt under it and in the shell around it
	for bx in range(-1, 2):
		for bz in range(-9, -6):
			if (bx, bz) != (0, -8):
				set_block(tex, (bx, floor - 1, bz), dirt)
	set_cell_state(tex, (0, floor - 1, -8), 2)
	set_block(tex, (0, floor - 2, -8), dirt)
	return tex


def build_demo(directory, width=640, height=360):
	"""Writes the demo scene (README.md's formats) to `directory`, with the plain skin as mcrl_steve.png."""
	arena = Arena()
	write_textures(directory, build_demo_textures(arena))
	Image.fromarray(plain_skin(), "RGBA").save(os.path.join(directory, "mcrl_steve.png"))
	floor = arena.floor_y
	scene = {
		"width": width, "height": height, "fovY": 70.0,
		"camera": {"pos": [3.5, floor + 3.6, -12.0], "yaw": 22.0, "pitch": 26.0},
		"steve": {"pos": [-1.8, floor, -6.2, 1.0], "look": [200.0, 185.0, 0.0, 0.0], "anim": [1.2, 0.5, 0.0, 0.0],
			"item": [-1.0, 0.0, 0.0, 0.0]},
		"arena": arena.to_json(),
		"atlas": "",
		"skin": "mcrl_steve.png",
	}
	with open(os.path.join(directory, "scene.json"), "w", encoding="utf-8") as f:
		json.dump(scene, f, indent=1)
		f.write("\n")


# ---- command line ----------------------------------------------------------------------------------------------------

def _parse_size(text):
	try:
		w, h = (int(x) for x in text.lower().split("x"))
	except ValueError:
		raise argparse.ArgumentTypeError(f"size must be WxH, not {text!r}") from None
	if w < 1 or h < 1:
		raise argparse.ArgumentTypeError("size must be positive")
	return w, h


def main(argv=None):
	parser = argparse.ArgumentParser(description="Render a scene the way MCPassthrough.fx draws it in voxel mode.")
	parser.add_argument("scene", nargs="?", help="scene directory: vox.r16, cells.r8, types.rgba32f, scene.json")
	parser.add_argument("--out", help="output PNG (default: <scene>/preview.png, or preview_<debug>.png)")
	parser.add_argument("--debug", choices=DEBUG_VIEWS, help="the shader's DebugView 1..4 instead of the composite "
		"(cells: red shell / green dug at RL's surface)")
	parser.add_argument("--size", type=_parse_size, help="render at WxH instead of the scene's size")
	parser.add_argument("--demo", metavar="OUT_DIR", help="write a small synthetic scene to OUT_DIR and render it")
	args = parser.parse_args(argv)
	if (args.scene is None) == (args.demo is None):
		parser.error("give a scene directory or --demo OUT_DIR")

	def show(message, category, filename, lineno, file=None, line=None):
		print(f"render.py: warning: {message}", file=sys.stderr)

	try:
		with warnings.catch_warnings():
			warnings.simplefilter("always", PreviewWarning)
			warnings.showwarning = show
			directory = args.scene
			if args.demo:
				build_demo(args.demo)
				directory = args.demo
			started = time.perf_counter()
			scene = load_scene(directory, args.size)
			frame = render_frame(scene, args.debug)
			out = args.out or os.path.join(directory, f"preview_{args.debug}.png" if args.debug else "preview.png")
			os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
			Image.fromarray(to_rgb8(frame.image), "RGB").save(out)
	except (OSError, ValueError, TypeError) as e:
		print(f"render.py: {e}", file=sys.stderr)
		return 1
	print(f"wrote {out} ({scene.width} x {scene.height}, {time.perf_counter() - started:.2f} s)")
	return 0


if __name__ == "__main__":
	sys.exit(main())
