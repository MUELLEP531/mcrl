// The world and block types as the texels MCPassthrough.fx reads in voxel mode (docs/SHADER_CONTRACT.md). No graphics
// API here: the host's renderer (the ReShade add-on) uploads these arrays.
#pragma once

#include "vox/block_registry.hpp"
#include "vox/world.hpp"

#include <cstdint>
#include <vector>

namespace vox {

/// MCTYPES: RGBA32F, 256 x 384.
constexpr int kTypesWidth = 256;
constexpr int kTypesHeight = 384;

/// MCHOLES texel values (R8_UNORM) for each CellState: Open 0, Shell and Backing 120 (~0.47), Dug 240 (~0.94).
std::uint8_t cellTexel(CellState state);

/// MCVOX: R16_UNORM, size.x wide and size.z * size.y high, row-major; block ids, with invisible blocks (and ids the
/// registry doesn't know) written as 0. `out` is resized to fit.
void exportVoxels(const World& world, const BlockRegistry& registry, std::vector<std::uint16_t>& out);
/// The texels of `rect` only, row-major (rect.width() per row), for a partial upload. Throws std::invalid_argument
/// (leaving `out` unchanged) if `rect` is inverted or reaches outside the texture.
void exportVoxelRect(const World& world, const BlockRegistry& registry, const DirtyRect& rect,
	std::vector<std::uint16_t>& out);

/// MCHOLES: R8_UNORM, same layout as exportVoxels.
void exportCells(const World& world, std::vector<std::uint8_t>& out);
/// As exportVoxelRect, for MCHOLES.
void exportCellRect(const World& world, const DirtyRect& rect, std::vector<std::uint8_t>& out);

/// MCTYPES: RGBA32F, kTypesWidth x kTypesHeight x 4 floats, row-major. Ids the registry doesn't have (and air) are
/// all zeros except slots = -1. Invisible registered blocks (the barrier) keep their real rows, so their solid flag is
/// still there for collision; MCVOX never points at them. `out` is resized to fit.
void exportTypes(const BlockRegistry& registry, std::vector<float>& out);

} // namespace vox
