#include "check.hpp"

#include "vox/world.hpp"

using vox::BlockChange;
using vox::CellState;
using vox::DirtyRect;
using vox::GridSpec;
using vox::Int3;
using vox::World;

TEST(world_default_grid_matches_shader)
{
	World w;
	CHECK(w.spec().origin == (Int3{-64, 48, -64}));
	CHECK(w.spec().size == (Int3{128, 40, 128}));
	CHECK_EQ(w.blocks().size(), static_cast<std::size_t>(128 * 40 * 128));
	CHECK_EQ(w.cells().size(), w.blocks().size());
}

TEST(world_rejects_empty_grid)
{
	CHECK_THROWS(World(GridSpec{{0, 0, 0}, {0, 1, 1}}));
	CHECK_THROWS(World(GridSpec{{0, 0, 0}, {1, -1, 1}}));
}

TEST(world_contains_bounds)
{
	World w;
	CHECK(w.contains({-64, 48, -64}));
	CHECK(w.contains({63, 87, 63}));
	CHECK(!w.contains({64, 60, 0}));
	CHECK(!w.contains({0, 47, 0}));
	CHECK(!w.contains({0, 88, 0}));
	CHECK(!w.contains({0, 60, -65}));
}

TEST(world_index_is_gpu_layout)
{
	World w;
	// texel (c.x, c.z + 128 * c.y) of a 128-wide texture, c relative to the corner
	const Int3 p{10, 50, -3};
	const Int3 c = p - w.spec().origin;
	CHECK_EQ(w.indexOf(p), static_cast<std::size_t>(c.x + 128 * (c.z + 128 * c.y)));
	CHECK(w.cellOf(w.indexOf(p)) == p);
	CHECK_EQ(w.indexOf({-64, 48, -64}), static_cast<std::size_t>(0));
	CHECK(w.cellOf(w.blocks().size() - 1) == (Int3{63, 87, 63}));
}

TEST(world_set_get_and_outside)
{
	World w;
	CHECK(w.set({1, 60, 2}, 5));
	CHECK_EQ(w.get({1, 60, 2}), 5);
	CHECK(!w.set({1, 60, 2}, 5)); // unchanged
	CHECK(!w.set({100, 60, 2}, 5)); // outside
	CHECK_EQ(w.get({100, 60, 2}), vox::kAir);
	CHECK(w.cell({100, 60, 2}) == CellState::Open);
}

TEST(world_dirty_rect_bounds_changes)
{
	World w;
	CHECK(!w.takeDirtyBlocks());
	w.set({-64, 48, -64}, 1); // texel (0, 0)
	w.set({-60, 49, -62}, 1); // texel (4, 2 + 128)
	const auto rect = w.takeDirtyBlocks();
	REQUIRE(rect.has_value());
	CHECK((*rect == DirtyRect{0, 0, 4, 130}));
	CHECK(!w.takeDirtyBlocks());
	CHECK(!w.takeDirtyCells());
	w.setCell({0, 60, 0}, CellState::Shell);
	CHECK(w.takeDirtyCells().has_value());
	w.setCell({0, 60, 0}, CellState::Shell); // unchanged: not dirty
	CHECK(!w.takeDirtyCells());
}

TEST(world_fill_clips_and_counts)
{
	World w;
	CHECK_EQ(w.fill({60, 86, 60}, {70, 90, 70}, 3), static_cast<std::size_t>(4 * 2 * 4));
	CHECK_EQ(w.get({63, 87, 63}), 3);
	CHECK_EQ(w.fill({60, 86, 60}, {70, 90, 70}, 3), static_cast<std::size_t>(0));
	CHECK_EQ(w.fill({5, 60, 5}, {4, 60, 5}, 3), static_cast<std::size_t>(0)); // empty box
}

TEST(world_journal_reports_changes_in_order)
{
	World w;
	CHECK_EQ(w.changeSeq(), static_cast<std::uint64_t>(0));
	w.set({0, 60, 0}, 1);
	w.set({0, 60, 0}, 2);
	w.set({1, 60, 0}, 3);
	std::vector<BlockChange> out;
	REQUIRE(w.changesSince(1, out));
	REQUIRE(out.size() == 2);
	CHECK_EQ(out[0].before, 1);
	CHECK_EQ(out[0].after, 2);
	CHECK_EQ(out[0].seq, static_cast<std::uint64_t>(2));
	CHECK(out[1].pos == (Int3{1, 60, 0}));
	out.clear();
	CHECK(w.changesSince(3, out));
	CHECK(out.empty());
}

TEST(world_journal_overflow_forces_resync)
{
	World w;
	const std::size_t n = World::kJournalCapacity + 10;
	for (std::size_t i = 0; i < n; ++i)
		w.set({0, 60, 0}, static_cast<vox::BlockId>(1 + i % 2));
	std::vector<BlockChange> out;
	CHECK(!w.changesSince(0, out));
	CHECK(out.empty());
	CHECK(!w.changesSince(9, out));
	REQUIRE(w.changesSince(10, out));
	CHECK_EQ(out.size(), World::kJournalCapacity);
	CHECK_EQ(out.front().seq, static_cast<std::uint64_t>(11));
}

TEST(world_assign_forces_resync_and_dirties_all)
{
	World w;
	w.set({0, 60, 0}, 1);
	w.takeDirtyBlocks();
	const std::uint64_t seen = w.changeSeq();
	std::vector<vox::BlockId> blocks(w.blocks().size(), 2);
	std::vector<CellState> cells(w.cells().size(), CellState::Shell);
	w.assign(blocks, cells);
	CHECK_EQ(w.get({5, 70, 5}), 2);
	CHECK(w.cell({5, 70, 5}) == CellState::Shell);
	std::vector<BlockChange> out;
	CHECK(!w.changesSince(seen, out)); // a reader that was up to date must resync
	const auto rect = w.takeDirtyBlocks();
	REQUIRE(rect.has_value());
	CHECK((*rect == DirtyRect{0, 0, 127, 128 * 40 - 1}));
	CHECK_THROWS(w.assign(std::vector<vox::BlockId>(3), cells));
}
