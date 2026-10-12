#include "check.hpp"

#include "vox/gpu_layout.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using vox::BlockId;
using vox::BlockRegistry;
using vox::BlockType;
using vox::CellState;
using vox::DirtyRect;
using vox::GridSpec;
using vox::Int3;
using vox::Rgb;
using vox::World;

namespace {

/// The highest id MCTYPES can address (256 x 128 ids).
constexpr int kMaxId = 256 * 128 - 1;

BlockType makeType(const std::string& name, int top, int side, int bottom)
{
	BlockType t;
	t.name = name;
	t.texTop = top;
	t.texSide = side;
	t.texBottom = bottom;
	return t;
}

/// stone 1, grass 2 (tinted top), barrier 3 (invisible), glass 4 (not solid), plain 5 (no textures).
BlockRegistry smallRegistry()
{
	BlockRegistry r;
	r.add(makeType("stone", 1, 1, 1));
	BlockType grass = makeType("grass_block", 0, 3, 2);
	grass.tintFaces = vox::kTintTop;
	grass.tint = Rgb{145, 189, 89};
	grass.color = Rgb{90, 140, 60};
	r.add(grass);
	BlockType barrier = makeType("barrier", -1, -1, -1);
	barrier.visible = false;
	barrier.breakable = false;
	r.add(barrier);
	BlockType glass = makeType("glass", 49, 49, 49);
	glass.solid = false;
	r.add(glass);
	BlockType plain = makeType("plain", -1, -1, -1);
	plain.color = Rgb{255, 1, 254};
	plain.tintFaces = vox::kTintSide | vox::kTintBottom;
	plain.tint = Rgb{0, 128, 255};
	r.add(plain);
	return r;
}

/// Every id MCTYPES can hold, each with data derived from its id (some invisible, some without textures).
BlockType derivedType(int i)
{
	BlockType t;
	t.name = "b" + std::to_string(i);
	t.texTop = i % 5 == 0 ? -1 : i % 4096;
	t.texSide = (i * 7 + 3) % 4096;
	t.texBottom = (i * 13 + 11) % 4097 - 1;
	t.solid = i % 3 != 0;
	t.visible = i % 11 != 0;
	t.tintFaces = static_cast<std::uint8_t>(i % 8);
	t.tint = Rgb{static_cast<std::uint8_t>(i & 255), static_cast<std::uint8_t>((i >> 8) & 255),
		static_cast<std::uint8_t>((i * 3) & 255)};
	t.color = Rgb{static_cast<std::uint8_t>((i * 5) & 255), static_cast<std::uint8_t>((i * 7) & 255),
		static_cast<std::uint8_t>((i >> 7) & 255)};
	return t;
}

/// Built once: BlockRegistry::add is linear in the registry's size.
const BlockRegistry& fullRegistry()
{
	static const BlockRegistry registry = [] {
		BlockRegistry r;
		for (int i = 1; i <= kMaxId; ++i)
			r.add(derivedType(i));
		return r;
	}();
	return registry;
}

std::uint32_t lcg(std::uint32_t& state)
{
	state = state * 1664525u + 1013904223u;
	return state >> 8;
}

// ---- the shader's side, re-implemented from MCPassthrough.fx ----

/// What tex2Dlod with MinFilter/MagFilter = POINT and AddressU/V = CLAMP reads: the texel containing uv * size.
int pointTexel(float uv, int size)
{
	const int t = static_cast<int>(std::floor(uv * static_cast<float>(size)));
	return t < 0 ? 0 : (t >= size ? size - 1 : t);
}

/// vox_at / shell_at / arena_cell address: float4((c.x + 0.5) / 128.0, (c.z + 128.0 * c.y + 0.5) / 5120.0, 0, 0).
/// Checked for both a true division and the multiply by the reciprocal a shader compiler may turn it into.
std::size_t cellTexelIndex(int cx, int cy, int cz, bool& consistent)
{
	const float u = (static_cast<float>(cx) + 0.5f) / 128.0f;
	const float v = (static_cast<float>(cz) + 128.0f * static_cast<float>(cy) + 0.5f) / 5120.0f;
	const float vr = (static_cast<float>(cz) + 128.0f * static_cast<float>(cy) + 0.5f) * (1.0f / 5120.0f);
	const int tx = pointTexel(u, 128), ty = pointTexel(v, 5120);
	consistent = pointTexel(vr, 5120) == ty;
	return static_cast<std::size_t>(ty) * 128 + static_cast<std::size_t>(tx);
}

/// vox_at: int(round(r * 65535.0)) of an R16_UNORM texel.
int shaderVoxAt(const std::vector<std::uint16_t>& tex, int cx, int cy, int cz, bool& consistent)
{
	const float r = static_cast<float>(tex[cellTexelIndex(cx, cy, cz, consistent)]) / 65535.0f;
	return static_cast<int>(std::round(r * 65535.0f));
}

/// shell_at / arena_cell: r > 0.7 ? 2 : (r > 0.25 ? 1 : 0) of an R8_UNORM texel.
int classify(std::uint8_t texel)
{
	const float r = static_cast<float>(texel) / 255.0f;
	return r > 0.7f ? 2 : (r > 0.25f ? 1 : 0);
}

/// type_row: float4((id % 256 + 0.5) / 256.0, (id / 256 + row * 128 + 0.5) / 384.0, 0, 0) of an RGBA32F texture.
std::array<float, 4> shaderTypeRow(const std::vector<float>& tex, int id, int row)
{
	const float u = (static_cast<float>(id % 256) + 0.5f) / 256.0f;
	const float v = (static_cast<float>(id / 256 + row * 128) + 0.5f) / 384.0f;
	const std::size_t texel =
		static_cast<std::size_t>(pointTexel(v, 384)) * 256 + static_cast<std::size_t>(pointTexel(u, 256));
	return {tex[texel * 4], tex[texel * 4 + 1], tex[texel * 4 + 2], tex[texel * 4 + 3]};
}

float unorm(std::uint8_t v) { return static_cast<float>(v) / 255.0f; }

/// Checks what block_color reads for `id` against its BlockType (or the unused-id row).
bool shaderSeesType(const std::vector<float>& tex, int id, const BlockType* t)
{
	const std::array<float, 4> f = shaderTypeRow(tex, id, 0), tn = shaderTypeRow(tex, id, 1),
							   av = shaderTypeRow(tex, id, 2);
	if (!t)
		return f[0] == -1.0f && f[1] == -1.0f && f[2] == -1.0f && f[3] == 0.0f && tn[0] == 0.0f && tn[1] == 0.0f
			&& tn[2] == 0.0f && tn[3] == 0.0f && av[0] == 0.0f && av[1] == 0.0f && av[2] == 0.0f && av[3] == 0.0f;
	const int slots[3] = {t->texTop, t->texSide, t->texBottom};
	for (int face = 0; face < 3; ++face)
	{
		// slot = face == 0 ? f.x : (face == 1 ? f.y : f.z); slot >= 0.0 picks the atlas
		const float slot = face == 0 ? f[0] : (face == 1 ? f[1] : f[2]);
		if (slot != static_cast<float>(slots[face]) || (slot >= 0.0f) != (slots[face] >= 0))
			return false;
		// ((int(tn.w) >> face) & 1) != 0 tints the face
		if ((((static_cast<int>(tn[3]) >> face) & 1) != 0) != (((t->tintFaces >> face) & 1) != 0))
			return false;
	}
	return f[3] == (t->solid ? 1.0f : 0.0f) && static_cast<int>(tn[3]) == t->tintFaces && tn[0] == unorm(t->tint.r)
		&& tn[1] == unorm(t->tint.g) && tn[2] == unorm(t->tint.b) && av[0] == unorm(t->color.r)
		&& av[1] == unorm(t->color.g) && av[2] == unorm(t->color.b) && av[3] == 1.0f;
}

template <typename T>
std::vector<T> subRect(const std::vector<T>& full, int width, const DirtyRect& rect)
{
	std::vector<T> out;
	for (int row = rect.row0; row <= rect.row1; ++row)
		for (int x = rect.x0; x <= rect.x1; ++x)
			out.push_back(full[static_cast<std::size_t>(row) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)]);
	return out;
}

template <typename F>
bool throwsInvalidArgument(F f)
{
	try
	{
		f();
	}
	catch (const std::invalid_argument&)
	{
		return true;
	}
	catch (...)
	{
		return false;
	}
	return false;
}

/// A small grid with a block and a cell state in every cell, from a few registered, invisible and unknown ids.
World patternedWorld(GridSpec spec)
{
	World w(spec);
	const BlockId ids[] = {0, 1, 2, 3, 4, 5, 6, 999, 65535};
	std::vector<BlockId> blocks(w.blocks().size());
	std::vector<CellState> cells(w.cells().size());
	for (std::size_t i = 0; i < blocks.size(); ++i)
	{
		blocks[i] = ids[(i * 7 + i / 3) % 9];
		cells[i] = static_cast<CellState>((i * 5 + i / 4) % 3);
	}
	w.assign(blocks, cells);
	return w;
}

/// The MCVOX value smallRegistry() gives `id`.
std::uint16_t expectedVoxel(BlockId id)
{
	return (id == 1 || id == 2 || id == 4 || id == 5) ? id : static_cast<std::uint16_t>(0);
}

} // namespace

// ---- MCHOLES ----

TEST(gpu_cell_texel_values)
{
	CHECK_EQ(static_cast<int>(vox::cellTexel(CellState::Open)), 0);
	CHECK_EQ(static_cast<int>(vox::cellTexel(CellState::Shell)), 120);
	CHECK_EQ(static_cast<int>(vox::cellTexel(CellState::Dug)), 240);
}

TEST(gpu_cell_texel_classifies_under_shader_thresholds)
{
	CHECK_EQ(classify(vox::cellTexel(CellState::Open)), 0);
	CHECK_EQ(classify(vox::cellTexel(CellState::Shell)), 1);
	CHECK_EQ(classify(vox::cellTexel(CellState::Dug)), 2);
	// with margin: one step either way still classifies the same
	CHECK_EQ(classify(static_cast<std::uint8_t>(vox::cellTexel(CellState::Shell) - 1)), 1);
	CHECK_EQ(classify(static_cast<std::uint8_t>(vox::cellTexel(CellState::Shell) + 1)), 1);
	CHECK_EQ(classify(static_cast<std::uint8_t>(vox::cellTexel(CellState::Dug) - 1)), 2);
	CHECK_EQ(classify(static_cast<std::uint8_t>(vox::cellTexel(CellState::Dug) + 1)), 2);
}

TEST(gpu_cell_texel_unknown_state_is_open)
{
	CHECK_EQ(static_cast<int>(vox::cellTexel(static_cast<CellState>(3))), 0);
	CHECK_EQ(static_cast<int>(vox::cellTexel(static_cast<CellState>(255))), 0);
}

TEST(gpu_export_cells_layout)
{
	const GridSpec spec{{-2, 10, 5}, {3, 4, 5}};
	const World w = patternedWorld(spec);
	std::vector<std::uint8_t> out(7, 99);
	vox::exportCells(w, out);
	REQUIRE(out.size() == static_cast<std::size_t>(3 * 4 * 5));
	// texel (c.x, c.z + size.z * c.y), size.x wide
	for (int y = 0; y < 4; ++y)
		for (int z = 0; z < 5; ++z)
			for (int x = 0; x < 3; ++x)
				CHECK_EQ(out[static_cast<std::size_t>(x + 3 * (z + 5 * y))], vox::cellTexel(w.cell(Int3{x, y, z} + spec.origin)));
}

TEST(gpu_export_cells_whole_default_grid_as_shader_reads)
{
	World w;
	std::vector<CellState> cells(w.cells().size());
	for (std::size_t i = 0; i < cells.size(); ++i)
		cells[i] = static_cast<CellState>((i * 31 + i / 128 + i / 5120) % 3);
	w.assign(std::vector<BlockId>(w.blocks().size(), vox::kAir), cells);
	std::vector<std::uint8_t> tex;
	vox::exportCells(w, tex);
	REQUIRE(tex.size() == static_cast<std::size_t>(128 * 5120));
	int wrong = 0, inconsistent = 0;
	for (int cy = 0; cy < 40; ++cy)
		for (int cz = 0; cz < 128; ++cz)
			for (int cx = 0; cx < 128; ++cx)
			{
				bool consistent = true;
				const int got = classify(tex[cellTexelIndex(cx, cy, cz, consistent)]);
				if (got != static_cast<int>(w.cell(Int3{cx - 64, cy + 48, cz - 64})))
					++wrong;
				if (!consistent)
					++inconsistent;
			}
	CHECK_EQ(wrong, 0);
	CHECK_EQ(inconsistent, 0);
}

TEST(gpu_export_cell_rect_matches_full_export_exhaustively)
{
	const World w = patternedWorld(GridSpec{{0, 0, 0}, {4, 3, 5}});
	std::vector<std::uint8_t> full;
	vox::exportCells(w, full);
	const int width = 4, height = 15;
	int mismatches = 0;
	for (int x0 = 0; x0 < width; ++x0)
		for (int x1 = x0; x1 < width; ++x1)
			for (int row0 = 0; row0 < height; ++row0)
				for (int row1 = row0; row1 < height; ++row1)
				{
					const DirtyRect rect{x0, row0, x1, row1};
					std::vector<std::uint8_t> part;
					vox::exportCellRect(w, rect, part);
					if (part != subRect(full, width, rect))
						++mismatches;
				}
	CHECK_EQ(mismatches, 0);
}

TEST(gpu_export_cell_rect_of_dirty_cells)
{
	World w;
	w.setCell({-64, 48, -64}, CellState::Shell);
	w.setCell({3, 60, -7}, CellState::Dug);
	w.setCell({63, 87, 63}, CellState::Shell);
	w.takeDirtyCells();
	w.setCell({10, 64, 2}, CellState::Dug);
	w.setCell({-5, 63, 20}, CellState::Shell);
	const auto rect = w.takeDirtyCells();
	REQUIRE(rect.has_value());
	std::vector<std::uint8_t> full, part;
	vox::exportCells(w, full);
	vox::exportCellRect(w, *rect, part);
	CHECK_EQ(part.size(), static_cast<std::size_t>(rect->width()) * static_cast<std::size_t>(rect->height()));
	CHECK(part == subRect(full, 128, *rect));
}

TEST(gpu_export_cell_rect_rejects_bad_rects)
{
	const World w(GridSpec{{0, 0, 0}, {4, 3, 5}}); // 4 x 15 texels
	const DirtyRect bad[] = {
		{-1, 0, 0, 0},
		{0, -1, 0, 0},
		{0, 0, 4, 0},
		{0, 0, 0, 15},
		{2, 0, 1, 0},
		{0, 3, 0, 2},
		{3, 14, 4, 15},
		{INT_MIN, 0, INT_MAX, 0},
		{0, INT_MIN, 0, INT_MAX},
		{INT_MAX, INT_MAX, INT_MAX, INT_MAX},
	};
	for (const DirtyRect& rect : bad)
	{
		std::vector<std::uint8_t> out(3, 77);
		CHECK(throwsInvalidArgument([&] { vox::exportCellRect(w, rect, out); }));
		CHECK((out == std::vector<std::uint8_t>(3, 77))); // untouched
	}
	std::vector<std::uint8_t> out;
	vox::exportCellRect(w, DirtyRect{0, 0, 3, 14}, out); // the whole texture is fine
	CHECK_EQ(out.size(), static_cast<std::size_t>(60));
}

// ---- MCVOX ----

TEST(gpu_export_voxels_size_matches_shader_texture)
{
	const World w;
	std::vector<std::uint16_t> out(3, 1);
	vox::exportVoxels(w, smallRegistry(), out);
	CHECK_EQ(out.size(), static_cast<std::size_t>(128 * 5120));
	bool allZero = true;
	for (const std::uint16_t v : out)
		allZero = allZero && v == 0;
	CHECK(allZero);
}

TEST(gpu_export_voxels_maps_invisible_and_unknown_to_zero)
{
	const BlockRegistry reg = smallRegistry();
	World w;
	w.set({0, 60, 0}, 1);       // stone
	w.set({1, 60, 0}, 3);       // barrier: invisible
	w.set({2, 60, 0}, 6);       // not registered
	w.set({3, 60, 0}, 65535);   // not registered
	w.set({4, 60, 0}, 5);       // plain: drawn without textures
	w.set({5, 60, 0}, 4);       // glass: not solid, still drawn
	std::vector<std::uint16_t> out;
	vox::exportVoxels(w, reg, out);
	CHECK_EQ(out[w.indexOf({0, 60, 0})], 1);
	CHECK_EQ(out[w.indexOf({1, 60, 0})], 0);
	CHECK_EQ(out[w.indexOf({2, 60, 0})], 0);
	CHECK_EQ(out[w.indexOf({3, 60, 0})], 0);
	CHECK_EQ(out[w.indexOf({4, 60, 0})], 5);
	CHECK_EQ(out[w.indexOf({5, 60, 0})], 4);
	CHECK_EQ(out[w.indexOf({6, 60, 0})], 0); // air
}

TEST(gpu_export_voxels_layout_non_default_grid)
{
	const GridSpec spec{{7, -3, 100}, {3, 4, 5}};
	const World w = patternedWorld(spec);
	std::vector<std::uint16_t> out;
	vox::exportVoxels(w, smallRegistry(), out);
	REQUIRE(out.size() == static_cast<std::size_t>(3 * 4 * 5));
	for (int y = 0; y < 4; ++y)
		for (int z = 0; z < 5; ++z)
			for (int x = 0; x < 3; ++x)
				CHECK_EQ(out[static_cast<std::size_t>(x + 3 * (z + 5 * y))], expectedVoxel(w.get(Int3{x, y, z} + spec.origin)));
}

TEST(gpu_export_voxels_whole_default_grid_as_shader_reads)
{
	// every cell, ids up to the maximum, invisible ids and ids beyond any registry
	const BlockRegistry& reg = fullRegistry();
	REQUIRE(reg.size() == static_cast<std::size_t>(kMaxId + 1));
	World w;
	std::vector<BlockId> blocks(w.blocks().size());
	std::uint32_t seed = 12345;
	for (std::size_t i = 0; i < blocks.size(); ++i)
	{
		const std::uint32_t pick = lcg(seed);
		blocks[i] = static_cast<BlockId>(pick % 8 == 0 ? 32768 + pick % 32768 : pick % 32768);
	}
	blocks.front() = static_cast<BlockId>(kMaxId);
	blocks.back() = static_cast<BlockId>(kMaxId);
	blocks[w.indexOf({63, 48, -64})] = 256;
	blocks[w.indexOf({-64, 87, 63})] = 255;
	w.assign(blocks, std::vector<CellState>(w.cells().size(), CellState::Open));

	std::vector<std::uint16_t> tex;
	vox::exportVoxels(w, reg, tex);
	REQUIRE(tex.size() == static_cast<std::size_t>(128 * 5120));
	int wrong = 0, inconsistent = 0, drawn = 0;
	for (int cy = 0; cy < 40; ++cy)
		for (int cz = 0; cz < 128; ++cz)
			for (int cx = 0; cx < 128; ++cx)
			{
				const BlockId id = w.get(Int3{cx - 64, cy + 48, cz - 64});
				const int expected = id != 0 && reg.contains(id) && reg.get(id).visible ? id : 0;
				bool consistent = true;
				if (shaderVoxAt(tex, cx, cy, cz, consistent) != expected)
					++wrong;
				if (!consistent)
					++inconsistent;
				if (expected != 0)
					++drawn;
			}
	CHECK_EQ(wrong, 0);
	CHECK_EQ(inconsistent, 0);
	CHECK(drawn > 500000);
	bool c0 = true, c1 = true;
	CHECK_EQ(shaderVoxAt(tex, 0, 0, 0, c0), kMaxId); // kMaxId is visible (32767 % 11 != 0)
	CHECK_EQ(shaderVoxAt(tex, 127, 39, 127, c1), kMaxId);
}

TEST(gpu_r16_round_trips_every_id_through_unorm)
{
	// vox_at reads r * 65535 and rounds: every 16-bit value must come back exactly
	int wrong = 0;
	for (int v = 0; v <= 65535; ++v)
	{
		const float r = static_cast<float>(v) / 65535.0f;
		if (static_cast<int>(std::round(r * 65535.0f)) != v)
			++wrong;
	}
	CHECK_EQ(wrong, 0);
}

TEST(gpu_export_voxel_rect_matches_full_export_exhaustively)
{
	const BlockRegistry reg = smallRegistry();
	const World w = patternedWorld(GridSpec{{0, 0, 0}, {4, 3, 5}});
	std::vector<std::uint16_t> full;
	vox::exportVoxels(w, reg, full);
	const int width = 4, height = 15;
	int mismatches = 0;
	for (int x0 = 0; x0 < width; ++x0)
		for (int x1 = x0; x1 < width; ++x1)
			for (int row0 = 0; row0 < height; ++row0)
				for (int row1 = row0; row1 < height; ++row1)
				{
					const DirtyRect rect{x0, row0, x1, row1};
					std::vector<std::uint16_t> part(2, 9);
					vox::exportVoxelRect(w, reg, rect, part);
					if (part != subRect(full, width, rect))
						++mismatches;
				}
	CHECK_EQ(mismatches, 0);
}

TEST(gpu_export_voxel_rect_default_grid_samples)
{
	const BlockRegistry reg = smallRegistry();
	World w;
	std::vector<BlockId> blocks(w.blocks().size());
	std::uint32_t seed = 99;
	for (BlockId& b : blocks)
		b = static_cast<BlockId>(lcg(seed) % 8);
	w.assign(blocks, std::vector<CellState>(w.cells().size(), CellState::Open));
	std::vector<std::uint16_t> full;
	vox::exportVoxels(w, reg, full);
	std::vector<DirtyRect> rects = {{0, 0, 127, 5119}, {0, 0, 0, 0}, {127, 5119, 127, 5119}, {0, 5119, 127, 5119},
		{127, 0, 127, 5119}, {5, 128, 9, 255}};
	for (int i = 0; i < 40; ++i)
	{
		const int xa = static_cast<int>(lcg(seed) % 128), xb = static_cast<int>(lcg(seed) % 128);
		const int ra = static_cast<int>(lcg(seed) % 5120), rb = static_cast<int>(lcg(seed) % 5120);
		rects.push_back(DirtyRect{(std::min)(xa, xb), (std::min)(ra, rb), (std::max)(xa, xb), (std::max)(ra, rb)});
	}
	for (const DirtyRect& rect : rects)
	{
		std::vector<std::uint16_t> part;
		vox::exportVoxelRect(w, reg, rect, part);
		CHECK_EQ(part.size(), static_cast<std::size_t>(rect.width()) * static_cast<std::size_t>(rect.height()));
		CHECK(part == subRect(full, 128, rect));
	}
}

TEST(gpu_export_voxel_rect_of_dirty_blocks)
{
	const BlockRegistry reg = smallRegistry();
	World w;
	w.set({0, 60, 0}, 1);
	w.takeDirtyBlocks();
	w.set({-10, 70, 12}, 2);
	w.set({20, 55, -30}, 3);
	w.set({21, 55, -30}, 5);
	const auto rect = w.takeDirtyBlocks();
	REQUIRE(rect.has_value());
	std::vector<std::uint16_t> full, part;
	vox::exportVoxels(w, reg, full);
	vox::exportVoxelRect(w, reg, *rect, part);
	CHECK(part == subRect(full, 128, *rect));
}

TEST(gpu_export_voxel_rect_rejects_bad_rects)
{
	const BlockRegistry reg = smallRegistry();
	const World w; // 128 x 5120 texels
	const DirtyRect bad[] = {
		{-1, 0, 5, 5},
		{0, -1, 5, 5},
		{0, 0, 128, 5},
		{0, 0, 5, 5120},
		{6, 0, 5, 5},
		{0, 6, 5, 5},
		{INT_MIN, INT_MIN, INT_MAX, INT_MAX},
		{INT_MAX, 0, INT_MAX, 0},
	};
	for (const DirtyRect& rect : bad)
	{
		std::vector<std::uint16_t> out(2, 4242);
		CHECK(throwsInvalidArgument([&] { vox::exportVoxelRect(w, reg, rect, out); }));
		CHECK((out == std::vector<std::uint16_t>(2, 4242)));
	}
}

// ---- MCTYPES ----

TEST(gpu_export_types_size)
{
	std::vector<float> out(5, 3.0f);
	vox::exportTypes(smallRegistry(), out);
	CHECK_EQ(out.size(), static_cast<std::size_t>(vox::kTypesWidth * vox::kTypesHeight * 4));
	CHECK_EQ(vox::kTypesWidth, 256);
	CHECK_EQ(vox::kTypesHeight, 384);
}

TEST(gpu_export_types_rows_of_a_block)
{
	const BlockRegistry reg = smallRegistry();
	std::vector<float> out;
	vox::exportTypes(reg, out);
	const auto at = [&out](int column, int row, int channel) {
		const std::size_t texel = static_cast<std::size_t>(row) * 256 + static_cast<std::size_t>(column);
		return out[texel * 4 + static_cast<std::size_t>(channel)];
	};
	// grass_block, id 2: column 2 of rows 0, 128, 256
	CHECK_EQ(at(2, 0, 0), 0.0f);
	CHECK_EQ(at(2, 0, 1), 3.0f);
	CHECK_EQ(at(2, 0, 2), 2.0f);
	CHECK_EQ(at(2, 0, 3), 1.0f);
	CHECK_EQ(at(2, 128, 0), 145.0f / 255.0f);
	CHECK_EQ(at(2, 128, 1), 189.0f / 255.0f);
	CHECK_EQ(at(2, 128, 2), 89.0f / 255.0f);
	CHECK_EQ(at(2, 128, 3), 1.0f);
	CHECK_EQ(at(2, 256, 0), 90.0f / 255.0f);
	CHECK_EQ(at(2, 256, 1), 140.0f / 255.0f);
	CHECK_EQ(at(2, 256, 2), 60.0f / 255.0f);
	CHECK_EQ(at(2, 256, 3), 1.0f);
	// plain, id 5: no textures, tinted sides and bottom
	CHECK_EQ(at(5, 0, 0), -1.0f);
	CHECK_EQ(at(5, 0, 1), -1.0f);
	CHECK_EQ(at(5, 0, 2), -1.0f);
	CHECK_EQ(at(5, 128, 0), 0.0f);
	CHECK_EQ(at(5, 128, 1), 128.0f / 255.0f);
	CHECK_EQ(at(5, 128, 2), 1.0f);
	CHECK_EQ(at(5, 128, 3), 6.0f);
	CHECK_EQ(at(5, 256, 0), 1.0f);
	CHECK_EQ(at(5, 256, 1), 1.0f / 255.0f);
	CHECK_EQ(at(5, 256, 2), 254.0f / 255.0f);
	// glass, id 4: not solid
	CHECK_EQ(at(4, 0, 0), 49.0f);
	CHECK_EQ(at(4, 0, 3), 0.0f);
}

TEST(gpu_export_types_air_and_unused_ids)
{
	const BlockRegistry reg = smallRegistry();
	std::vector<float> out(static_cast<std::size_t>(256 * 384 * 4), 7.0f); // stale data must be overwritten
	vox::exportTypes(reg, out);
	int wrong = 0;
	for (int id = 0; id <= kMaxId; ++id)
	{
		if (id != 0 && reg.contains(static_cast<BlockId>(id)))
			continue;
		if (!shaderSeesType(out, id, nullptr))
			++wrong;
	}
	CHECK_EQ(wrong, 0);
	// air explicitly: slots -1, not solid, no tint, black, w 0
	const std::array<float, 4> a0 = shaderTypeRow(out, 0, 0), a1 = shaderTypeRow(out, 0, 1), a2 = shaderTypeRow(out, 0, 2);
	CHECK(a0 == (std::array<float, 4>{-1.0f, -1.0f, -1.0f, 0.0f}));
	CHECK(a1 == (std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}));
	CHECK(a2 == (std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(gpu_export_types_keeps_invisible_blocks)
{
	// the barrier never reaches MCVOX, but its row still carries solid for collision
	const BlockRegistry reg = smallRegistry();
	std::vector<float> out;
	vox::exportTypes(reg, out);
	CHECK(shaderSeesType(out, 3, &reg.get(3)));
	const std::array<float, 4> slots = shaderTypeRow(out, 3, 0), color = shaderTypeRow(out, 3, 2);
	CHECK_EQ(slots[3], 1.0f);
	CHECK_EQ(color[3], 1.0f);
}

TEST(gpu_export_types_small_registry_as_shader_reads)
{
	const BlockRegistry reg = smallRegistry();
	std::vector<float> out;
	vox::exportTypes(reg, out);
	for (BlockId id = 1; id < reg.size(); ++id)
		CHECK(shaderSeesType(out, id, &reg.get(id)));
}

TEST(gpu_export_types_every_id_as_shader_reads)
{
	// ids >= 256 (later rows of each row block) up to the maximum the texture holds
	const BlockRegistry& reg = fullRegistry();
	std::vector<float> out;
	vox::exportTypes(reg, out);
	REQUIRE(out.size() == static_cast<std::size_t>(256 * 384 * 4));
	int wrong = 0;
	for (int id = 0; id <= kMaxId; ++id)
		if (!shaderSeesType(out, id, id == 0 ? nullptr : &reg.get(static_cast<BlockId>(id))))
			++wrong;
	CHECK_EQ(wrong, 0);
	// spot checks against the texel addressing in the contract: column id % 256, row id / 256 + 128 * rowBlock
	const auto texel = [&out](int column, int row) {
		const std::size_t i = (static_cast<std::size_t>(row) * 256 + static_cast<std::size_t>(column)) * 4;
		return std::array<float, 4>{out[i], out[i + 1], out[i + 2], out[i + 3]};
	};
	const BlockType& top = reg.get(static_cast<BlockId>(kMaxId));
	CHECK(texel(255, 127) == (std::array<float, 4>{static_cast<float>(top.texTop), static_cast<float>(top.texSide),
		static_cast<float>(top.texBottom), top.solid ? 1.0f : 0.0f}));
	const std::array<float, 4> topTint = texel(255, 255), topColor = texel(255, 383);
	CHECK_EQ(topTint[3], static_cast<float>(top.tintFaces));
	CHECK_EQ(topColor[0], unorm(top.color.r));
	const BlockType& b256 = reg.get(256);
	const std::array<float, 4> slots256 = texel(0, 1), tint256 = texel(0, 129), color256 = texel(0, 257);
	CHECK_EQ(slots256[1], static_cast<float>(b256.texSide));
	CHECK_EQ(tint256[1], unorm(b256.tint.g));
	CHECK_EQ(color256[2], unorm(b256.color.b));
}

TEST(gpu_export_types_slots_are_exact_integers)
{
	// atlas slots up to 4095 and the tint mask are used as integers: RGBA32F holds them exactly
	BlockRegistry reg;
	reg.add(makeType("odd", 4095, 2049, 4093));
	BlockType t = makeType("mask", 0, 1, 2);
	t.tintFaces = 7;
	reg.add(t);
	std::vector<float> out;
	vox::exportTypes(reg, out);
	const std::array<float, 4> f = shaderTypeRow(out, 1, 0);
	CHECK_EQ(static_cast<int>(f[0]), 4095);
	CHECK_EQ(static_cast<int>(f[1]), 2049);
	CHECK_EQ(static_cast<int>(f[2]), 4093);
	CHECK_EQ(static_cast<int>(shaderTypeRow(out, 2, 1)[3]), 7);
	CHECK_EQ(static_cast<int>(shaderTypeRow(out, 1, 1)[3]), 0);
}
