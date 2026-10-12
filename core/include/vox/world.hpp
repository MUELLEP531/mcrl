// The voxel world: a fixed grid of block ids plus, per cell, its arena state (open / shell / dug). Stored in the same
// layout as the shader's textures, so uploading is a straight copy.
#pragma once

#include "vox/types.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace vox {

/// The grid's extent in world cells. The default is the one MCPassthrough.fx hard-codes.
struct GridSpec
{
	Int3 origin{-64, 48, -64};
	Int3 size{128, 40, 128};

	friend bool operator==(const GridSpec& a, const GridSpec& b) { return a.origin == b.origin && a.size == b.size; }
};

/// A cell's relation to the host game's own surfaces (see docs/SHADER_CONTRACT.md, MCHOLES).
enum class CellState : std::uint8_t
{
	Open = 0,  ///< not part of the host game's geometry
	Shell = 1, ///< contains the host game's surface (a wall, the floor): blocks here are behind that surface
	Dug = 2,   ///< a shell cell that was broken open: the host's surface is gone here
};

/// One block change, for network sync.
struct BlockChange
{
	Int3 pos;
	BlockId before = kAir;
	BlockId after = kAir;
	std::uint64_t seq = 0; ///< 1 for the world's first change, then +1 per change
};

/// A rectangle of texels in the GPU layout (column x, row z + size.z * y, both relative to the grid's corner),
/// inclusive on both ends.
struct DirtyRect
{
	int x0 = 0, row0 = 0, x1 = 0, row1 = 0;

	int width() const { return x1 - x0 + 1; }
	int height() const { return row1 - row0 + 1; }
	friend bool operator==(const DirtyRect& a, const DirtyRect& b)
	{
		return a.x0 == b.x0 && a.row0 == b.row0 && a.x1 == b.x1 && a.row1 == b.row1;
	}
};

class World
{
public:
	/// How many changes the journal keeps for changesSince() before a reader has to resync from scratch.
	static constexpr std::size_t kJournalCapacity = 65536;

	/// An all-air, all-open world. Throws std::invalid_argument if a size is not positive.
	explicit World(GridSpec spec = {});

	const GridSpec& spec() const { return spec_; }
	bool contains(Int3 p) const;
	/// Index into blocks() / cells() of a contained cell: x + size.x * (z + size.z * y), relative to the corner.
	std::size_t indexOf(Int3 p) const;
	/// The cell at a storage index (inverse of indexOf).
	Int3 cellOf(std::size_t index) const;

	/// The block at `p`; air outside the grid.
	BlockId get(Int3 p) const;
	/// Sets the block at `p`. Returns false (and changes nothing) outside the grid or if it already is `id`.
	/// Records the change in the journal and marks it dirty.
	bool set(Int3 p, BlockId id);
	/// Sets every cell in the box [lo, hi] (inclusive, clipped to the grid) to `id`; returns how many changed.
	std::size_t fill(Int3 lo, Int3 hi, BlockId id);

	/// The arena state at `p`; Open outside the grid.
	CellState cell(Int3 p) const;
	/// Sets the arena state at `p` (ignored outside the grid); marks the cells texture dirty if it changed.
	void setCell(Int3 p, CellState state);

	/// Storage in the GPU layout (see indexOf).
	const std::vector<BlockId>& blocks() const { return blocks_; }
	const std::vector<CellState>& cells() const { return cells_; }

	/// The texels of blocks() / cells() changed since the last call (their bounding rectangle), and clears it.
	std::optional<DirtyRect> takeDirtyBlocks();
	std::optional<DirtyRect> takeDirtyCells();
	/// Marks everything dirty (after a load, or when the GPU textures were recreated).
	void markAllDirty();

	/// Sequence number of the latest change (0: none yet).
	std::uint64_t changeSeq() const { return seq_; }
	/// Appends every change with seq > `since` to `out`, oldest first. Returns false if some of them already fell out
	/// of the journal (the reader must resync from the whole world); `out` is then left unchanged.
	bool changesSince(std::uint64_t since, std::vector<BlockChange>& out) const;

	/// Replaces every block and cell state at once (loading a save). Sizes must match the grid, else
	/// std::invalid_argument. Not journalled: the journal is cleared and readers must resync; everything is dirty.
	void assign(std::vector<BlockId> blocks, std::vector<CellState> cells);

private:
	void touchBlocks(std::size_t index);
	void touchCells(std::size_t index);

	GridSpec spec_;
	std::vector<BlockId> blocks_;
	std::vector<CellState> cells_;
	std::optional<DirtyRect> dirtyBlocks_, dirtyCells_;
	std::deque<BlockChange> journal_;
	std::uint64_t seq_ = 0;
	std::uint64_t journalFloor_ = 0; ///< every change with seq <= this has been dropped (or wiped by assign())
};

} // namespace vox
