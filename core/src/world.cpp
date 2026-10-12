#include "vox/world.hpp"

#include <algorithm>
#include <stdexcept>

namespace vox {

namespace {

void grow(std::optional<DirtyRect>& rect, int x, int row)
{
	if (!rect)
	{
		rect = DirtyRect{x, row, x, row};
		return;
	}
	rect->x0 = (std::min)(rect->x0, x);
	rect->x1 = (std::max)(rect->x1, x);
	rect->row0 = (std::min)(rect->row0, row);
	rect->row1 = (std::max)(rect->row1, row);
}

} // namespace

World::World(GridSpec spec)
	: spec_(spec)
{
	if (spec.size.x <= 0 || spec.size.y <= 0 || spec.size.z <= 0)
		throw std::invalid_argument("grid size must be positive");
	const std::size_t count =
		static_cast<std::size_t>(spec.size.x) * static_cast<std::size_t>(spec.size.y) * static_cast<std::size_t>(spec.size.z);
	blocks_.assign(count, kAir);
	cells_.assign(count, CellState::Open);
}

bool World::contains(Int3 p) const
{
	const Int3 r = p - spec_.origin;
	return r.x >= 0 && r.y >= 0 && r.z >= 0 && r.x < spec_.size.x && r.y < spec_.size.y && r.z < spec_.size.z;
}

std::size_t World::indexOf(Int3 p) const
{
	const Int3 r = p - spec_.origin;
	return static_cast<std::size_t>(r.x)
		+ static_cast<std::size_t>(spec_.size.x)
		* (static_cast<std::size_t>(r.z) + static_cast<std::size_t>(spec_.size.z) * static_cast<std::size_t>(r.y));
}

Int3 World::cellOf(std::size_t index) const
{
	const std::size_t sx = static_cast<std::size_t>(spec_.size.x);
	const std::size_t sz = static_cast<std::size_t>(spec_.size.z);
	const int x = static_cast<int>(index % sx);
	const std::size_t row = index / sx;
	const int z = static_cast<int>(row % sz);
	const int y = static_cast<int>(row / sz);
	return Int3{x, y, z} + spec_.origin;
}

BlockId World::get(Int3 p) const { return contains(p) ? blocks_[indexOf(p)] : kAir; }

bool World::set(Int3 p, BlockId id)
{
	if (!contains(p))
		return false;
	const std::size_t i = indexOf(p);
	if (blocks_[i] == id)
		return false;
	BlockChange change;
	change.pos = p;
	change.before = blocks_[i];
	change.after = id;
	change.seq = ++seq_;
	blocks_[i] = id;
	journal_.push_back(change);
	while (journal_.size() > kJournalCapacity)
	{
		journalFloor_ = journal_.front().seq;
		journal_.pop_front();
	}
	touchBlocks(i);
	return true;
}

std::size_t World::fill(Int3 lo, Int3 hi, BlockId id)
{
	const Int3 gridLo = spec_.origin;
	const Int3 gridHi = spec_.origin + spec_.size - Int3{1, 1, 1};
	const Int3 a{(std::max)(lo.x, gridLo.x), (std::max)(lo.y, gridLo.y), (std::max)(lo.z, gridLo.z)};
	const Int3 b{(std::min)(hi.x, gridHi.x), (std::min)(hi.y, gridHi.y), (std::min)(hi.z, gridHi.z)};
	std::size_t changed = 0;
	for (int y = a.y; y <= b.y; ++y)
		for (int z = a.z; z <= b.z; ++z)
			for (int x = a.x; x <= b.x; ++x)
				if (set({x, y, z}, id))
					++changed;
	return changed;
}

CellState World::cell(Int3 p) const { return contains(p) ? cells_[indexOf(p)] : CellState::Open; }

void World::setCell(Int3 p, CellState state)
{
	if (!contains(p))
		return;
	const std::size_t i = indexOf(p);
	if (cells_[i] == state)
		return;
	cells_[i] = state;
	touchCells(i);
}

std::optional<DirtyRect> World::takeDirtyBlocks()
{
	std::optional<DirtyRect> rect = dirtyBlocks_;
	dirtyBlocks_.reset();
	return rect;
}

std::optional<DirtyRect> World::takeDirtyCells()
{
	std::optional<DirtyRect> rect = dirtyCells_;
	dirtyCells_.reset();
	return rect;
}

void World::markAllDirty()
{
	const DirtyRect all{0, 0, spec_.size.x - 1, spec_.size.z * spec_.size.y - 1};
	dirtyBlocks_ = all;
	dirtyCells_ = all;
}

bool World::changesSince(std::uint64_t since, std::vector<BlockChange>& out) const
{
	if (since >= seq_)
		return true;
	if (since < journalFloor_)
		return false;
	// the journal holds seq journalFloor_ + 1 .. seq_, consecutive
	const std::size_t skip = static_cast<std::size_t>(since - journalFloor_);
	out.insert(out.end(), journal_.begin() + static_cast<std::ptrdiff_t>(skip), journal_.end());
	return true;
}

void World::assign(std::vector<BlockId> blocks, std::vector<CellState> cells)
{
	if (blocks.size() != blocks_.size() || cells.size() != cells_.size())
		throw std::invalid_argument("World::assign: sizes don't match the grid");
	blocks_ = std::move(blocks);
	cells_ = std::move(cells);
	// a new sequence number nobody has seen: every reader's changesSince() fails and it resyncs
	journal_.clear();
	journalFloor_ = ++seq_;
	markAllDirty();
}

void World::touchBlocks(std::size_t index)
{
	const std::size_t sx = static_cast<std::size_t>(spec_.size.x);
	grow(dirtyBlocks_, static_cast<int>(index % sx), static_cast<int>(index / sx));
}

void World::touchCells(std::size_t index)
{
	const std::size_t sx = static_cast<std::size_t>(spec_.size.x);
	grow(dirtyCells_, static_cast<int>(index % sx), static_cast<int>(index / sx));
}

} // namespace vox
