// Saving and loading a world. Blocks are stored by name, so a save still loads after the block list changes.
#pragma once

#include "vox/block_registry.hpp"
#include "vox/world.hpp"

#include <cstddef>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace vox {

struct LoadReport
{
	/// Names in the save that the registry doesn't have; their blocks were loaded as air.
	std::vector<std::string> unknownBlocks;
	/// How many cells were set to air because their name was unknown.
	std::size_t cellsDropped = 0;
};

/// Writes `world` in the binary format "MCRLVOX1" (little-endian: magic, grid origin and size, the names of the
/// block types used, then run-length-encoded blocks and cell states in storage order). Throws std::runtime_error if
/// the stream fails.
void saveWorld(const World& world, const BlockRegistry& registry, std::ostream& out);

/// Reads a world written by saveWorld. The grid in the file must match `spec`. Throws std::runtime_error on a bad
/// magic, a grid mismatch, truncated or corrupt data (runs that overflow the grid, out-of-range palette indices or
/// cell states). Open streams in binary mode.
World loadWorld(std::istream& in, const BlockRegistry& registry, GridSpec spec = {}, LoadReport* report = nullptr);

} // namespace vox
