#include "vox/gpu_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Notes on behaviour the header leaves implicit:
// - Backing exports as shell (120). A CellState outside the four (only reachable through a cast) is exported as
//   open (0).
// - exportVoxelRect / exportCellRect throw std::invalid_argument if `rect` is inverted (x1 < x0 or row1 < row0) or
//   reaches outside the texture (size.x wide, size.z * size.y high); `out` is then left unchanged.
// - exportTypes writes real rows for every registered id except air, including invisible ones (the barrier): MCVOX
//   never points at them, but their solid flag is still the collision data the contract describes.

namespace vox {

namespace {

/// MCTYPES row blocks: atlas slots and solid, tint, own colour.
constexpr int kRowSlots = 0;
constexpr int kRowTint = 128;
constexpr int kRowColor = 256;

struct TextureSize
{
	long long width = 0, height = 0;
};

TextureSize textureSize(const World& world)
{
	const GridSpec& s = world.spec();
	return {static_cast<long long>(s.size.x), static_cast<long long>(s.size.z) * static_cast<long long>(s.size.y)};
}

void checkRect(const World& world, const DirtyRect& rect)
{
	const TextureSize t = textureSize(world);
	if (rect.x0 > rect.x1 || rect.row0 > rect.row1)
		throw std::invalid_argument("texel rect is inverted");
	if (rect.x0 < 0 || rect.row0 < 0 || rect.x1 >= t.width || rect.row1 >= t.height)
		throw std::invalid_argument("texel rect reaches outside the texture");
}

/// The MCVOX value of a block: its id if it's drawn, else 0.
std::uint16_t voxelTexel(const BlockRegistry& registry, BlockId id)
{
	return id != kAir && registry.contains(id) && registry.get(id).visible ? id : static_cast<std::uint16_t>(0);
}

/// voxelTexel() of every registered id, for exporting the whole grid.
std::vector<std::uint16_t> voxelTexels(const BlockRegistry& registry)
{
	std::vector<std::uint16_t> texels(registry.size(), 0);
	for (std::size_t i = 0; i < texels.size(); ++i)
		texels[i] = voxelTexel(registry, static_cast<BlockId>(i));
	return texels;
}

/// Copies the texels of `rect` (already checked) from the world's storage, mapping each through `texel`.
template <typename Out, typename In, typename Map>
void copyRect(
	const World& world, const std::vector<In>& storage, const DirtyRect& rect, std::vector<Out>& out, Map texel)
{
	const std::size_t width = static_cast<std::size_t>(world.spec().size.x);
	// not rect.width() / height(): those are int and a rect may span more than INT_MAX rows of a huge grid
	const std::size_t w = static_cast<std::size_t>(rect.x1) - static_cast<std::size_t>(rect.x0) + 1;
	const std::size_t h = static_cast<std::size_t>(rect.row1) - static_cast<std::size_t>(rect.row0) + 1;
	out.resize(w * h);
	for (std::size_t r = 0; r < h; ++r)
	{
		const std::size_t src = (static_cast<std::size_t>(rect.row0) + r) * width + static_cast<std::size_t>(rect.x0);
		for (std::size_t x = 0; x < w; ++x)
			out[r * w + x] = texel(storage[src + x]);
	}
}

void setTexel(std::vector<float>& out, int column, int row, float r, float g, float b, float a)
{
	const std::size_t i = (static_cast<std::size_t>(row) * kTypesWidth + static_cast<std::size_t>(column)) * 4;
	out[i + 0] = r;
	out[i + 1] = g;
	out[i + 2] = b;
	out[i + 3] = a;
}

float unorm(std::uint8_t v) { return static_cast<float>(v) / 255.0f; }

} // namespace

std::uint8_t cellTexel(CellState state)
{
	// the shader classifies r > 0.7 as dug and r > 0.25 as shell: 120 / 255 = 0.47 and 240 / 255 = 0.94 sit well
	// inside their bands
	switch (state)
	{
	case CellState::Shell:
	case CellState::Backing:
		return 120;
	case CellState::Dug:
		return 240;
	case CellState::Open:
		break;
	}
	return 0;
}

void exportVoxels(const World& world, const BlockRegistry& registry, std::vector<std::uint16_t>& out)
{
	const std::vector<std::uint16_t> texels = voxelTexels(registry);
	const std::vector<BlockId>& blocks = world.blocks();
	out.resize(blocks.size());
	for (std::size_t i = 0; i < blocks.size(); ++i)
		out[i] = blocks[i] < texels.size() ? texels[blocks[i]] : static_cast<std::uint16_t>(0);
}

void exportVoxelRect(const World& world, const BlockRegistry& registry, const DirtyRect& rect,
	std::vector<std::uint16_t>& out)
{
	checkRect(world, rect);
	// usually a few texels after an edit: look ids up directly instead of building voxelTexels()
	copyRect(world, world.blocks(), rect, out, [&registry](BlockId id) { return voxelTexel(registry, id); });
}

void exportCells(const World& world, std::vector<std::uint8_t>& out)
{
	const std::vector<CellState>& cells = world.cells();
	out.resize(cells.size());
	for (std::size_t i = 0; i < cells.size(); ++i)
		out[i] = cellTexel(cells[i]);
}

void exportCellRect(const World& world, const DirtyRect& rect, std::vector<std::uint8_t>& out)
{
	checkRect(world, rect);
	copyRect(world, world.cells(), rect, out, [](CellState state) { return cellTexel(state); });
}

void exportTypes(const BlockRegistry& registry, std::vector<float>& out)
{
	out.assign(static_cast<std::size_t>(kTypesWidth) * kTypesHeight * 4, 0.0f);
	// 128 rows of 256 ids per row block
	const int maxIds = kTypesWidth * (kTypesHeight / 3);
	for (int id = 0; id < maxIds; ++id)
	{
		const int column = id % kTypesWidth;
		const int row = id / kTypesWidth;
		const BlockId blockId = static_cast<BlockId>(id);
		if (blockId == kAir || !registry.contains(blockId))
		{
			setTexel(out, column, row + kRowSlots, -1.0f, -1.0f, -1.0f, 0.0f);
			continue;
		}
		const BlockType& t = registry.get(blockId);
		setTexel(out, column, row + kRowSlots, static_cast<float>(t.texTop), static_cast<float>(t.texSide),
			static_cast<float>(t.texBottom), t.solid ? 1.0f : 0.0f);
		setTexel(out, column, row + kRowTint, unorm(t.tint.r), unorm(t.tint.g), unorm(t.tint.b),
			static_cast<float>(t.tintFaces));
		setTexel(out, column, row + kRowColor, unorm(t.color.r), unorm(t.color.g), unorm(t.color.b), 1.0f);
	}
}

} // namespace vox
