#include "check.hpp"

#include "rl/arena.hpp"
#include "vox/edit.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using rl::ArenaMapping;
using rl::ArenaOptions;
using rl::RlVec;
using rl::SoccarDimensions;
using vox::BlockId;
using vox::CellState;
using vox::Int3;
using vox::Vec3;
using vox::World;

namespace {

struct Blocks
{
	vox::BlockRegistry registry;
	BlockId barrier = vox::kAir, dirt = vox::kAir, stone = vox::kAir, glass = vox::kAir;
};

Blocks makeBlocks()
{
	Blocks b;
	vox::BlockType barrier;
	barrier.name = "arena_barrier";
	barrier.visible = false;
	b.barrier = b.registry.add(barrier);
	vox::BlockType dirt;
	dirt.name = "dirt";
	b.dirt = b.registry.add(dirt);
	vox::BlockType stone;
	stone.name = "stone";
	b.stone = b.registry.add(stone);
	vox::BlockType glass;
	glass.name = "glass";
	b.glass = b.registry.add(glass);
	return b;
}

/// Shell only: everything outside the arena is air, the worst case for leaks.
ArenaOptions shellOnly(const Blocks& b)
{
	ArenaOptions o;
	o.shellBlock = b.barrier;
	return o;
}

/// Every shell cell holds the barrier (shellTakesGround off), so isShell() can check the block.
ArenaOptions layered(const Blocks& b)
{
	ArenaOptions o;
	o.shellBlock = b.barrier;
	o.shellTakesGround = false;
	o.topsoilBlock = b.dirt;
	o.groundBlock = b.stone;
	o.outsideBlock = b.glass;
	return o;
}

RlVec rlv(float x, float y, float z)
{
	RlVec p;
	p.x = x;
	p.y = y;
	p.z = z;
	return p;
}

Vec3 v3(float x, float y, float z)
{
	Vec3 p;
	p.x = x;
	p.y = y;
	p.z = z;
	return p;
}

/// How many of a cell's 8 corners are inside the arena (the classification buildSoccarArena documents).
int cornersInside(Int3 c, const ArenaOptions& o)
{
	int n = 0;
	for (int dy = 0; dy <= 1; ++dy)
		for (int dz = 0; dz <= 1; ++dz)
			for (int dx = 0; dx <= 1; ++dx)
			{
				const Vec3 p =
					v3(static_cast<float>(c.x + dx), static_cast<float>(c.y + dy), static_cast<float>(c.z + dz));
				if (rl::insideSoccar(rl::toRl(p, o.mapping), o.dims))
					++n;
			}
	return n;
}

bool isShell(const World& w, const Blocks& b, Int3 c)
{
	return w.cell(c) == CellState::Shell && w.get(c) == b.barrier;
}

bool isOpen(const World& w, Int3 c) { return w.cell(c) == CellState::Open && w.get(c) == vox::kAir; }

/// Outside the arena: Open state holding `block` (the build's choice for that layer).
bool isOutside(const World& w, Int3 c, BlockId block) { return w.cell(c) == CellState::Open && w.get(c) == block; }

struct Fill
{
	std::vector<unsigned char> reached;
	std::size_t count = 0;
	std::size_t notInside = 0; ///< reached cells with a corner outside the arena
};

/// 6-connected flood fill through non-solid cells from `start`.
Fill floodFill(const World& w, const vox::BlockRegistry& registry, Int3 start, const ArenaOptions& o)
{
	Fill f;
	f.reached.assign(w.blocks().size(), 0);
	std::vector<Int3> stack;
	if (!w.contains(start) || registry.get(w.get(start)).solid)
		return f;
	stack.push_back(start);
	f.reached[w.indexOf(start)] = 1;
	const Int3 steps[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
	while (!stack.empty())
	{
		const Int3 c = stack.back();
		stack.pop_back();
		++f.count;
		if (cornersInside(c, o) != 8)
			++f.notInside;
		for (const Int3& s : steps)
		{
			const Int3 n = c + s;
			if (!w.contains(n) || f.reached[w.indexOf(n)] || registry.get(w.get(n)).solid)
				continue;
			f.reached[w.indexOf(n)] = 1;
			stack.push_back(n);
		}
	}
	return f;
}

std::size_t countState(const World& w, CellState s)
{
	std::size_t n = 0;
	for (CellState c : w.cells())
		if (c == s)
			++n;
	return n;
}

std::size_t countBlock(const World& w, BlockId id)
{
	std::size_t n = 0;
	for (BlockId c : w.blocks())
		if (c == id)
			++n;
	return n;
}

} // namespace

// ---- mapping --------------------------------------------------------------------------------------------------------

TEST(arena_mapping_axes)
{
	const ArenaMapping m;
	// rl x -> minecraft x, rl y -> minecraft z, rl z -> minecraft y above the floor
	const Vec3 a = rl::toVoxel(rlv(100.0f, 0.0f, 0.0f), m);
	CHECK_NEAR(a.x, 1.0, 1e-6);
	CHECK_NEAR(a.y, 64.0, 1e-6);
	CHECK_NEAR(a.z, 0.0, 1e-6);
	const Vec3 b = rl::toVoxel(rlv(0.0f, 100.0f, 0.0f), m);
	CHECK_NEAR(b.x, 0.0, 1e-6);
	CHECK_NEAR(b.y, 64.0, 1e-6);
	CHECK_NEAR(b.z, 1.0, 1e-6);
	const Vec3 c = rl::toVoxel(rlv(0.0f, 0.0f, 100.0f), m);
	CHECK_NEAR(c.x, 0.0, 1e-6);
	CHECK_NEAR(c.y, 65.0, 1e-6);
	CHECK_NEAR(c.z, 0.0, 1e-6);
	const Vec3 d = rl::toVoxel(rlv(-4096.0f, -5120.0f, 2044.0f), m);
	CHECK_NEAR(d.x, -40.96, 1e-5);
	CHECK_NEAR(d.y, 84.44, 1e-5);
	CHECK_NEAR(d.z, -51.2, 1e-5);
}

TEST(arena_mapping_floor_is_exact)
{
	const ArenaMapping m;
	const Vec3 a = rl::toVoxel(rlv(0.0f, 0.0f, 0.0f), m);
	CHECK_EQ(a.y, 64.0f);
	const RlVec b = rl::toRl(v3(3.0f, 64.0f, -7.0f), m);
	CHECK_EQ(b.x, 300.0f);
	CHECK_EQ(b.y, -700.0f);
	CHECK_EQ(b.z, 0.0f);
	// the goal's back wall lands exactly on a cell boundary
	CHECK_EQ(rl::toRl(v3(0.0f, 65.0f, 60.0f), m).y, 6000.0f);
	CHECK_EQ(rl::toRl(v3(0.0f, 65.0f, -60.0f), m).y, -6000.0f);
}

TEST(arena_mapping_round_trip)
{
	const ArenaMapping m;
	const RlVec pts[] = {rlv(0.0f, 0.0f, 0.0f), rlv(4096.0f, 5120.0f, 2044.0f), rlv(-893.0f, -6000.0f, 642.775f),
		rlv(1234.5f, -987.25f, 17.0f)};
	for (const RlVec& p : pts)
	{
		const RlVec q = rl::toRl(rl::toVoxel(p, m), m);
		CHECK_NEAR(q.x, p.x, 1e-3);
		CHECK_NEAR(q.y, p.y, 1e-3);
		CHECK_NEAR(q.z, p.z, 1e-3);
	}
	const Vec3 v = v3(-12.25f, 70.5f, 33.75f);
	const Vec3 w = rl::toVoxel(rl::toRl(v, m), m);
	CHECK_NEAR(w.x, v.x, 1e-5);
	CHECK_NEAR(w.y, v.y, 1e-5);
	CHECK_NEAR(w.z, v.z, 1e-5);
}

TEST(arena_mapping_custom)
{
	ArenaMapping m;
	m.uuPerBlock = 50.0f;
	m.floorY = 60.5f;
	const Vec3 a = rl::toVoxel(rlv(100.0f, -200.0f, 25.0f), m);
	CHECK_NEAR(a.x, 2.0, 1e-6);
	CHECK_NEAR(a.y, 61.0, 1e-6);
	CHECK_NEAR(a.z, -4.0, 1e-6);
	const RlVec b = rl::toRl(v3(2.0f, 61.0f, -4.0f), m);
	CHECK_NEAR(b.x, 100.0, 1e-4);
	CHECK_NEAR(b.y, -200.0, 1e-4);
	CHECK_NEAR(b.z, 25.0, 1e-4);
	const RlVec c = rl::toRl(v3(0.0f, 60.5f, 0.0f), m);
	CHECK_EQ(c.z, 0.0f);
}

// ---- insideSoccar ---------------------------------------------------------------------------------------------------

TEST(arena_inside_centre_and_floor)
{
	CHECK(rl::insideSoccar(rlv(0.0f, 0.0f, 1000.0f)));
	CHECK(rl::insideSoccar(rlv(0.0f, 0.0f, 0.0f))); // on the floor: closed
	CHECK(rl::insideSoccar(rlv(0.0f, 0.0f, -0.0f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, 0.0f, -0.001f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, 0.0f, -100.0f)));
}

TEST(arena_inside_ceiling)
{
	CHECK(rl::insideSoccar(rlv(0.0f, 0.0f, 2044.0f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, 0.0f, 2044.01f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, 5500.0f, 2044.0f))); // above the goal, behind the back wall
}

TEST(arena_inside_side_walls)
{
	CHECK(rl::insideSoccar(rlv(4096.0f, 0.0f, 100.0f)));
	CHECK(!rl::insideSoccar(rlv(4096.01f, 0.0f, 100.0f)));
	CHECK(rl::insideSoccar(rlv(-4096.0f, 0.0f, 100.0f)));
	CHECK(!rl::insideSoccar(rlv(-4096.01f, 0.0f, 100.0f)));
	CHECK(rl::insideSoccar(rlv(4096.0f, 3968.0f, 0.0f))); // where the side wall meets the corner, on the floor
	CHECK(!rl::insideSoccar(rlv(4096.0f, 3968.01f, 0.0f)));
}

TEST(arena_inside_back_walls)
{
	CHECK(rl::insideSoccar(rlv(2000.0f, 5120.0f, 100.0f)));
	CHECK(!rl::insideSoccar(rlv(2000.0f, 5120.01f, 100.0f)));
	CHECK(rl::insideSoccar(rlv(-2000.0f, -5120.0f, 100.0f)));
	CHECK(!rl::insideSoccar(rlv(-2000.0f, -5120.01f, 100.0f)));
	// beside the goal mouth: back wall
	CHECK(!rl::insideSoccar(rlv(893.01f, 5200.0f, 100.0f)));
	CHECK(rl::insideSoccar(rlv(893.01f, 5120.0f, 100.0f)));
}

TEST(arena_inside_corners)
{
	// |x| + |y| <= 8064, in all four quadrants
	const float sx[] = {1.0f, -1.0f};
	for (float a : sx)
		for (float c : sx)
		{
			CHECK(rl::insideSoccar(rlv(a * 4000.0f, c * 4000.0f, 100.0f)));
			CHECK(rl::insideSoccar(rlv(a * 4000.0f, c * 4064.0f, 100.0f))); // on the chamfer
			CHECK(!rl::insideSoccar(rlv(a * 4000.0f, c * 4064.01f, 100.0f)));
			// inside both the side and back walls' box: only the chamfer cuts it
			CHECK(!rl::insideSoccar(rlv(a * 4090.0f, c * 4000.0f, 100.0f)));
			CHECK(!rl::insideSoccar(rlv(a * 3000.0f, c * 5100.0f, 100.0f)));
			CHECK(rl::insideSoccar(rlv(a * 2944.0f, c * 5120.0f, 100.0f))); // where the chamfer meets the back wall
			CHECK(!rl::insideSoccar(rlv(a * 4096.0f, c * 5120.0f, 100.0f))); // the box's corner is cut away
		}
}

TEST(arena_inside_goals)
{
	CHECK(rl::insideSoccar(rlv(0.0f, 5500.0f, 300.0f)));
	CHECK(rl::insideSoccar(rlv(0.0f, -5500.0f, 300.0f)));
	CHECK(rl::insideSoccar(rlv(893.0f, 6000.0f, 642.775f))); // the goal box's far corner: closed
	CHECK(rl::insideSoccar(rlv(-893.0f, -6000.0f, 0.0f)));
	CHECK(!rl::insideSoccar(rlv(893.01f, 5500.0f, 300.0f)));
	CHECK(!rl::insideSoccar(rlv(-893.01f, -5500.0f, 300.0f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, 5500.0f, -0.01f)));
}

TEST(arena_inside_behind_and_above_goal)
{
	CHECK(rl::insideSoccar(rlv(0.0f, 6000.0f, 300.0f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, 6000.01f, 300.0f))); // behind the goal's back
	CHECK(!rl::insideSoccar(rlv(0.0f, -6000.01f, 300.0f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, 7000.0f, 300.0f)));
	// above the goal mouth: inside the pitch in front of the back wall, outside beyond it
	CHECK(!rl::insideSoccar(rlv(0.0f, 5500.0f, 643.0f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, -5500.0f, 700.0f)));
	CHECK(rl::insideSoccar(rlv(0.0f, 5120.0f, 643.0f)));
	CHECK(rl::insideSoccar(rlv(0.0f, 5100.0f, 2000.0f)));
}

TEST(arena_inside_symmetric)
{
	const RlVec pts[] = {rlv(4096.0f, 100.0f, 50.0f), rlv(4000.0f, 4064.0f, 10.0f), rlv(4000.0f, 4064.01f, 10.0f),
		rlv(893.0f, 6000.0f, 642.775f), rlv(893.0f, 6000.01f, 600.0f), rlv(1000.0f, 5500.0f, 300.0f),
		rlv(12.5f, 5119.9f, 2044.0f), rlv(3000.0f, 3000.0f, 2044.01f), rlv(4096.01f, 0.0f, 0.0f)};
	for (const RlVec& p : pts)
	{
		const bool in = rl::insideSoccar(p);
		CHECK_EQ(rl::insideSoccar(rlv(-p.x, p.y, p.z)), in);
		CHECK_EQ(rl::insideSoccar(rlv(p.x, -p.y, p.z)), in);
		CHECK_EQ(rl::insideSoccar(rlv(-p.x, -p.y, p.z)), in);
	}
}

TEST(arena_inside_custom_dims_and_nan)
{
	SoccarDimensions d;
	d.halfWidth = 100.0f;
	d.halfLength = 200.0f;
	d.height = 50.0f;
	d.cornerSum = 250.0f;
	d.goalHalfWidth = 10.0f;
	d.goalHeight = 5.0f;
	d.goalDepth = 20.0f;
	CHECK(rl::insideSoccar(rlv(100.0f, 0.0f, 50.0f), d));
	CHECK(!rl::insideSoccar(rlv(100.0f, 151.0f, 10.0f), d));
	CHECK(rl::insideSoccar(rlv(10.0f, -220.0f, 5.0f), d));
	CHECK(!rl::insideSoccar(rlv(10.0f, -220.0f, 5.5f), d));
	CHECK(!rl::insideSoccar(rlv(0.0f, 0.0f, 1000.0f), d)); // inside the default arena, not this one
	const float nan = std::numeric_limits<float>::quiet_NaN();
	CHECK(!rl::insideSoccar(rlv(nan, 0.0f, 100.0f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, nan, 100.0f)));
	CHECK(!rl::insideSoccar(rlv(0.0f, 0.0f, nan)));
}

// ---- buildSoccarArena -----------------------------------------------------------------------------------------------

TEST(arena_build_floor)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, shellOnly(b));
	// the floor at y = 64 is a cell boundary: the shell is the layer behind it
	CHECK(isShell(w, b, {0, 63, 0}));
	CHECK(isShell(w, b, {-1, 63, -1}));
	CHECK(isShell(w, b, {20, 63, 50}));
	CHECK(isOutside(w, {39, 63, 50}, vox::kAir)); // beyond the corner
	CHECK(isOpen(w, {0, 64, 0}));
	CHECK(isOpen(w, {-1, 64, -1}));
	CHECK(isOutside(w, {0, 62, 0}, vox::kAir));
}

TEST(arena_build_side_walls)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, shellOnly(b));
	// side wall at x = 40.96 blocks lies inside cell 40 (and -40.96 inside cell -41)
	for (int y = 64; y <= 83; ++y)
	{
		CHECK(isShell(w, b, {40, y, 0}));
		CHECK(isOpen(w, {39, y, 0}));
		CHECK(isOutside(w, {41, y, 0}, vox::kAir));
		CHECK(isShell(w, b, {-41, y, 0}));
		CHECK(isOpen(w, {-40, y, 0}));
		CHECK(isOutside(w, {-42, y, 0}, vox::kAir));
	}
	CHECK(isShell(w, b, {40, 63, 0}));
	CHECK(isShell(w, b, {40, 84, 0}));
	CHECK(isOutside(w, {40, 62, 0}, vox::kAir));
}

TEST(arena_build_ceiling)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, shellOnly(b));
	// ceiling at 64 + 20.44
	CHECK(isOpen(w, {0, 83, 0}));
	CHECK(isShell(w, b, {0, 84, 0}));
	CHECK(isOutside(w, {0, 85, 0}, vox::kAir));
	CHECK(isOutside(w, {0, 87, 0}, vox::kAir));
	CHECK(isShell(w, b, {-30, 84, 40}));
}

TEST(arena_build_back_walls)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, shellOnly(b));
	// back walls at z = +-51.2, beside the goal
	CHECK(isOpen(w, {20, 65, 50}));
	CHECK(isShell(w, b, {20, 65, 51}));
	CHECK(isOutside(w, {20, 65, 52}, vox::kAir));
	CHECK(isOpen(w, {20, 65, -51}));
	CHECK(isShell(w, b, {20, 65, -52}));
	CHECK(isOutside(w, {20, 65, -53}, vox::kAir));
	// above the goal mouth the back wall goes on: the goal's top edge cell and above
	CHECK(isShell(w, b, {0, 70, 51}));
	CHECK(isShell(w, b, {0, 75, 51}));
	CHECK(isShell(w, b, {0, 75, -52}));
}

TEST(arena_build_goal_back_and_sides)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, shellOnly(b));
	// goal backs at z = +-60 exactly: the shell is the cell behind the surface
	for (int y = 64; y <= 69; ++y)
	{
		CHECK(isOpen(w, {0, y, 59}));
		CHECK(isShell(w, b, {0, y, 60}));
		CHECK(isOutside(w, {0, y, 61}, vox::kAir));
		CHECK(isOpen(w, {0, y, -60}));
		CHECK(isShell(w, b, {0, y, -61}));
		CHECK(isOutside(w, {0, y, -62}, vox::kAir));
	}
	// goal sides at x = +-8.93, roof at y = 70.43, floor shell at y = 63
	CHECK(isShell(w, b, {8, 65, 55}));
	CHECK(isOutside(w, {9, 65, 55}, vox::kAir));
	CHECK(isShell(w, b, {-9, 65, 55}));
	CHECK(isOutside(w, {-10, 65, 55}, vox::kAir));
	CHECK(isShell(w, b, {0, 70, 55}));
	CHECK(isOutside(w, {0, 71, 55}, vox::kAir));
	CHECK(isShell(w, b, {0, 63, 55}));
	CHECK(isShell(w, b, {0, 63, -56}));
	CHECK(isOutside(w, {0, 62, 55}, vox::kAir));
}

TEST(arena_build_goal_interior_open)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, shellOnly(b));
	int notOpen = 0;
	for (int y = 64; y <= 69; ++y)
		for (int x = -8; x <= 7; ++x)
			for (int z = -60; z <= 59; ++z) // both goals and the pitch between them
				if (!isOpen(w, {x, y, z}))
					++notOpen;
	CHECK_EQ(notOpen, 0);
}

TEST(arena_build_corner_chamfer)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, shellOnly(b));
	// |x| + |z| <= 80.64 blocks: a cell (x >= 0, z >= 0) has corner sums x + z .. x + z + 2, so along the chamfer
	// x + z = 78 is open, 79 and 80 are shell, 81 is outside. The other quadrants through the cell mirror.
	int bad = 0;
	for (int quadrant = 0; quadrant < 4; ++quadrant)
		for (int x = 30; x <= 39; ++x)
			for (int y = 64; y <= 83; ++y)
				for (int sum = 78; sum <= 81; ++sum)
				{
					Int3 c{x, y, sum - x};
					if (quadrant & 1)
						c.x = -c.x - 1;
					if (quadrant & 2)
						c.z = -c.z - 1;
					const bool ok = sum == 78 ? isOpen(w, c)
						: (sum == 81 ? isOutside(w, c, vox::kAir) : isShell(w, b, c));
					if (!ok)
						++bad;
				}
	CHECK_EQ(bad, 0);
	// where the chamfer meets the side wall (40.96, 39.68) and the back wall (29.44, 51.2)
	CHECK(isShell(w, b, {40, 70, 39}));
	CHECK(isShell(w, b, {40, 70, 40}));
	CHECK(isOutside(w, {40, 70, 41}, vox::kAir));
	CHECK(isShell(w, b, {29, 70, 51}));
	CHECK(isOutside(w, {30, 70, 51}, vox::kAir));
}

TEST(arena_build_mirror_symmetric)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, layered(b));
	int bad = 0;
	for (int y = 48; y < 88; ++y)
		for (int z = -64; z < 64; ++z)
			for (int x = -64; x < 64; ++x)
			{
				const Int3 c{x, y, z};
				const Int3 mx{-x - 1, y, z}, mz{x, y, -z - 1};
				if (w.get(c) != w.get(mx) || w.cell(c) != w.cell(mx) || w.get(c) != w.get(mz)
					|| w.cell(c) != w.cell(mz))
					++bad;
			}
	CHECK_EQ(bad, 0);
}

TEST(arena_build_classification_matches_corners)
{
	// every cell, against the header's rule evaluated independently here
	const Blocks b = makeBlocks();
	const ArenaOptions o = layered(b);
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	int bad = 0;
	for (std::size_t i = 0; i < w.blocks().size(); ++i)
	{
		const Int3 c = w.cellOf(i);
		const int n = cornersInside(c, o);
		const bool ok = n == 8 ? isOpen(w, c)
			: (n > 0 ? isShell(w, b, c) : (w.cell(c) == CellState::Open && w.get(c) != vox::kAir));
		if (!ok)
			++bad;
	}
	CHECK_EQ(bad, 0);
}

TEST(arena_build_counts)
{
	// cross-checked by an exact-arithmetic Python classification and by hand:
	// open = 20 layers * 7896 (80 x 102 minus the corners) + 2 goals * 16 * 6 * 9
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, layered(b));
	CHECK_EQ(countState(w, CellState::Shell), static_cast<std::size_t>(24752));
	CHECK_EQ(countBlock(w, b.barrier), static_cast<std::size_t>(24752));
	CHECK_EQ(countBlock(w, vox::kAir), static_cast<std::size_t>(159648));
	CHECK_EQ(countState(w, CellState::Open), static_cast<std::size_t>(159648 + 470960));
	CHECK_EQ(countState(w, CellState::Dug), static_cast<std::size_t>(0));
	// outside: 217404 at or above the floor, 40564 topsoil (y 61..63), 212992 ground (y 48..60)
	CHECK_EQ(countBlock(w, b.glass), static_cast<std::size_t>(217404));
	CHECK_EQ(countBlock(w, b.dirt), static_cast<std::size_t>(40564));
	CHECK_EQ(countBlock(w, b.stone), static_cast<std::size_t>(212992));
}

TEST(arena_build_watertight)
{
	const Blocks b = makeBlocks();
	const ArenaOptions o = shellOnly(b);
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	const Fill f = floodFill(w, b.registry, {0, 65, 0}, o);
	CHECK_EQ(f.notInside, static_cast<std::size_t>(0));
	CHECK_EQ(f.count, static_cast<std::size_t>(159648)); // every open cell, nothing else
	CHECK(f.reached[w.indexOf({0, 65, 59})]);
	CHECK(f.reached[w.indexOf({-8, 69, 55})]);
	CHECK(f.reached[w.indexOf({0, 65, -60})]);
	CHECK(f.reached[w.indexOf({7, 64, -55})]);
	CHECK(!f.reached[w.indexOf({0, 65, 61})]);
	CHECK(!f.reached[w.indexOf({0, 86, 0})]);
	CHECK(!f.reached[w.indexOf({0, 50, 0})]);
}

TEST(arena_build_watertight_unaligned_mapping)
{
	// surfaces that cut through cells everywhere
	const Blocks b = makeBlocks();
	ArenaOptions o = shellOnly(b);
	o.mapping.uuPerBlock = 137.5f;
	o.mapping.floorY = 60.3f;
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	const Int3 start = vox::cellAt(rl::toVoxel(rlv(0.0f, 0.0f, 500.0f), o.mapping));
	REQUIRE(isOpen(w, start));
	const Fill f = floodFill(w, b.registry, start, o);
	CHECK_EQ(f.notInside, static_cast<std::size_t>(0));
	CHECK(f.count > static_cast<std::size_t>(10000));
	const Int3 goalA = vox::cellAt(rl::toVoxel(rlv(0.0f, 5700.0f, 300.0f), o.mapping));
	const Int3 goalB = vox::cellAt(rl::toVoxel(rlv(0.0f, -5700.0f, 300.0f), o.mapping));
	CHECK(f.reached[w.indexOf(goalA)]);
	CHECK(f.reached[w.indexOf(goalB)]);
	// the floor's shell is the layer holding y = 60.3
	CHECK(isShell(w, b, {0, 60, 0}));
	CHECK(isOpen(w, {0, 61, 0}));
}

TEST(arena_build_custom_mapping)
{
	const Blocks b = makeBlocks();
	ArenaOptions o = layered(b);
	o.mapping.uuPerBlock = 200.0f; // half the default scale
	o.mapping.floorY = 60.0f;
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	CHECK(isShell(w, b, {0, 59, 0}));
	CHECK(isOpen(w, {0, 60, 0}));
	// side wall at 20.48, ceiling at 70.22, goal back at 30
	CHECK(isOpen(w, {19, 62, 0}));
	CHECK(isShell(w, b, {20, 62, 0}));
	CHECK(isShell(w, b, {-21, 62, 0}));
	CHECK(isOutside(w, {21, 62, 0}, b.glass));
	CHECK(isShell(w, b, {0, 70, 0}));
	CHECK(isOutside(w, {0, 71, 0}, b.glass));
	CHECK(isShell(w, b, {0, 61, 30}));
	CHECK(isOpen(w, {0, 61, 29}));
	CHECK(isOutside(w, {0, 61, 31}, b.glass));
	// ground follows the floor: topsoil 57..59, stone below
	CHECK(isOutside(w, {30, 59, 0}, b.dirt));
	CHECK(isOutside(w, {30, 57, 0}, b.dirt));
	CHECK(isOutside(w, {30, 56, 0}, b.stone));
	CHECK(isOutside(w, {30, 60, 0}, b.glass));
}

TEST(arena_build_ground_layers)
{
	const Blocks b = makeBlocks();
	World w;
	rl::buildSoccarArena(w, b.registry, layered(b));
	// under the centre: the floor's shell, then the rest of the topsoil, then ground
	CHECK(isShell(w, b, {0, 63, 0}));
	CHECK(isOutside(w, {0, 62, 0}, b.dirt));
	CHECK(isOutside(w, {0, 61, 0}, b.dirt));
	for (int y = 48; y <= 60; ++y)
		CHECK(isOutside(w, {0, y, 0}, b.stone));
	// behind the side wall: 3 layers of topsoil level with the floor, the outside block above
	CHECK(isOutside(w, {50, 64, 0}, b.glass));
	CHECK(isOutside(w, {50, 87, 0}, b.glass));
	CHECK(isOutside(w, {50, 63, 0}, b.dirt));
	CHECK(isOutside(w, {50, 62, 0}, b.dirt));
	CHECK(isOutside(w, {50, 61, 0}, b.dirt));
	CHECK(isOutside(w, {50, 60, 0}, b.stone));
	CHECK(isOutside(w, {50, 48, 0}, b.stone));
	// above the ceiling and behind the goal
	CHECK(isOutside(w, {0, 85, 0}, b.glass));
	CHECK(isOutside(w, {0, 65, 62}, b.glass));
	CHECK(isOutside(w, {0, 63, 62}, b.dirt));
}

TEST(arena_build_topsoil_depth)
{
	const Blocks b = makeBlocks();
	ArenaOptions o = layered(b);
	World w;
	o.topsoilDepth = 0;
	rl::buildSoccarArena(w, b.registry, o);
	CHECK(isOutside(w, {50, 63, 0}, b.stone));
	CHECK(isOutside(w, {0, 62, 0}, b.stone));
	CHECK(isOutside(w, {50, 64, 0}, b.glass));
	o.topsoilDepth = 1;
	rl::buildSoccarArena(w, b.registry, o);
	CHECK(isOutside(w, {50, 63, 0}, b.dirt));
	CHECK(isOutside(w, {50, 62, 0}, b.stone));
	CHECK(isOutside(w, {0, 62, 0}, b.stone));
	o.topsoilDepth = 1000; // deeper than the grid: all topsoil
	rl::buildSoccarArena(w, b.registry, o);
	CHECK(isOutside(w, {50, 48, 0}, b.dirt));
	CHECK(isOutside(w, {0, 48, 0}, b.dirt));
	CHECK_EQ(countBlock(w, b.stone), static_cast<std::size_t>(0));
}

TEST(arena_build_fractional_floor_layers)
{
	// floorY = 64.5: the floor's surface is inside layer 64 (its shell); the floor's level is floor(64.5) = 64,
	// so layer 64 outside the arena gets the outside block and topsoil is 61..63
	const Blocks b = makeBlocks();
	ArenaOptions o = layered(b);
	o.mapping.floorY = 64.5f;
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	CHECK(isShell(w, b, {0, 64, 0}));
	CHECK(isOpen(w, {0, 65, 0}));
	CHECK(isOutside(w, {0, 63, 0}, b.dirt));
	CHECK(isOutside(w, {0, 61, 0}, b.dirt));
	CHECK(isOutside(w, {0, 60, 0}, b.stone));
	CHECK(isOutside(w, {50, 64, 0}, b.glass));
	CHECK(isOutside(w, {50, 63, 0}, b.dirt));
	CHECK(isOutside(w, {50, 60, 0}, b.stone));
}

TEST(arena_build_air_ground_leaves_empty)
{
	const Blocks b = makeBlocks();
	World w;
	w.fill({-64, 48, -64}, {63, 87, 63}, b.stone);
	rl::buildSoccarArena(w, b.registry, shellOnly(b));
	CHECK_EQ(countBlock(w, b.stone), static_cast<std::size_t>(0));
	CHECK_EQ(countBlock(w, b.barrier) + countBlock(w, vox::kAir), w.blocks().size());
}

TEST(arena_build_resets_dug_and_edits)
{
	const Blocks b = makeBlocks();
	const ArenaOptions o = layered(b);
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	// dig two shell cells, place a block in the air, change the ground
	w.set({0, 63, 0}, vox::kAir);
	w.setCell({0, 63, 0}, CellState::Dug);
	w.set({40, 70, 0}, vox::kAir);
	w.setCell({40, 70, 0}, CellState::Dug);
	w.set({0, 64, 0}, b.stone);
	w.set({0, 62, 0}, vox::kAir);
	w.setCell({5, 70, 5}, CellState::Shell);
	rl::buildSoccarArena(w, b.registry, o);
	CHECK(isShell(w, b, {0, 63, 0}));
	CHECK(isShell(w, b, {40, 70, 0}));
	CHECK(isOpen(w, {0, 64, 0}));
	CHECK(isOutside(w, {0, 62, 0}, b.dirt));
	CHECK(isOpen(w, {5, 70, 5}));
	CHECK_EQ(countState(w, CellState::Dug), static_cast<std::size_t>(0));
	CHECK_EQ(countState(w, CellState::Shell), static_cast<std::size_t>(24752));
}

TEST(arena_build_readers_resync)
{
	// installed with World::assign: the journal is wiped (readers resync) and everything is dirty
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 70, 0}, b.stone);
	const std::uint64_t before = w.changeSeq();
	(void)w.takeDirtyBlocks();
	(void)w.takeDirtyCells();
	rl::buildSoccarArena(w, b.registry, layered(b));
	CHECK(w.changeSeq() > before);
	std::vector<vox::BlockChange> out;
	CHECK(!w.changesSince(before, out));
	CHECK(out.empty());
	const vox::DirtyRect all{0, 0, 127, 128 * 40 - 1};
	const auto blocks = w.takeDirtyBlocks();
	const auto cells = w.takeDirtyCells();
	REQUIRE(blocks.has_value());
	REQUIRE(cells.has_value());
	CHECK(*blocks == all);
	CHECK(*cells == all);
}

TEST(arena_build_custom_grid)
{
	// works on whatever grid the world has
	const Blocks b = makeBlocks();
	vox::GridSpec spec;
	spec.origin = {-45, 60, -4};
	spec.size = {90, 30, 8};
	World w(spec);
	rl::buildSoccarArena(w, b.registry, layered(b));
	CHECK(isShell(w, b, {0, 63, 0}));
	CHECK(isOpen(w, {0, 64, 0}));
	CHECK(isOutside(w, {0, 62, 0}, b.dirt));
	CHECK(isOutside(w, {0, 60, 0}, b.stone));
	CHECK(isShell(w, b, {40, 70, 0}));
	CHECK(isShell(w, b, {-41, 70, -4}));
	CHECK(isOutside(w, {44, 70, 3}, b.glass));
	CHECK(isShell(w, b, {0, 84, 3}));
	CHECK(isOutside(w, {0, 89, 3}, b.glass));
}

TEST(arena_build_validates_arguments)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 70, 0}, b.stone);
	const std::uint64_t seq = w.changeSeq();
	const BlockId unknown = static_cast<BlockId>(b.registry.size());

	ArenaOptions o; // default: shellBlock is air
	CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));
	o = layered(b);
	o.shellBlock = unknown;
	CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));
	o = layered(b);
	o.topsoilBlock = unknown;
	CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));
	o = layered(b);
	o.groundBlock = unknown;
	CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));
	o = layered(b);
	o.outsideBlock = unknown;
	CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));
	o = layered(b);
	o.topsoilDepth = -1;
	CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));
	const float bad[] = {
		0.0f, -100.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
	for (float v : bad)
	{
		o = layered(b);
		o.mapping.uuPerBlock = v;
		CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));
	}
	o = layered(b);
	o.mapping.floorY = std::numeric_limits<float>::quiet_NaN();
	CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));
	o.mapping.floorY = -std::numeric_limits<float>::infinity();
	CHECK_THROWS(rl::buildSoccarArena(w, b.registry, o));

	// nothing was changed by a rejected build
	CHECK_EQ(w.changeSeq(), seq);
	CHECK_EQ(w.get({0, 70, 0}), b.stone);
	CHECK_EQ(countState(w, CellState::Shell), static_cast<std::size_t>(0));

	// air is a valid ground / outside block
	o = shellOnly(b);
	rl::buildSoccarArena(w, b.registry, o);
	CHECK(isShell(w, b, {0, 63, 0}));
}

TEST(arena_build_throws_invalid_argument)
{
	const Blocks b = makeBlocks();
	World w;
	bool caught = false;
	try
	{
		rl::buildSoccarArena(w, b.registry, ArenaOptions{});
	}
	catch (const std::invalid_argument&)
	{
		caught = true;
	}
	CHECK(caught);
}

TEST(arena_shell_takes_ground_under_the_floor)
{
	// default: shell cells hold their layer's solid ground, so a hole dug into RL's floor has dirt walls
	const Blocks b = makeBlocks();
	ArenaOptions o = layered(b);
	o.shellTakesGround = true;
	o.outsideBlock = vox::kAir;
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	CHECK(w.cell({0, 63, 0}) == CellState::Shell);
	CHECK_EQ(w.get({0, 63, 0}), b.dirt);     // the floor's shell: topsoil
	CHECK(isOpen(w, {0, 64, 0}));
	CHECK(w.cell({40, 70, 0}) == CellState::Shell);
	CHECK_EQ(w.get({40, 70, 0}), b.barrier); // side wall above the floor: outside is air, so the barrier
	CHECK(w.cell({0, 84, 0}) == CellState::Shell);
	CHECK_EQ(w.get({0, 84, 0}), b.barrier);  // ceiling
	// every shell cell is solid either way
	std::size_t shells = 0, solidShells = 0;
	for (std::size_t i = 0; i < w.cells().size(); ++i)
		if (w.cells()[i] == CellState::Shell)
		{
			++shells;
			solidShells += b.registry.get(w.blocks()[i]).solid ? 1u : 0u;
		}
	CHECK_EQ(shells, static_cast<std::size_t>(24752));
	CHECK_EQ(solidShells, shells);
}

TEST(arena_shell_takes_solid_outside_block)
{
	const Blocks b = makeBlocks();
	ArenaOptions o = layered(b);
	o.shellTakesGround = true; // outsideBlock is glass (solid in this registry)
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	CHECK_EQ(w.get({40, 70, 0}), b.glass);
	CHECK_EQ(w.get({0, 84, 0}), b.glass);
	CHECK_EQ(w.get({0, 63, 0}), b.dirt);
	CHECK_EQ(countBlock(w, b.barrier), static_cast<std::size_t>(0));
}

TEST(arena_shell_skips_non_solid_ground)
{
	Blocks b = makeBlocks();
	vox::BlockType fog;
	fog.name = "fog";
	fog.solid = false;
	const BlockId fogId = b.registry.add(fog);
	ArenaOptions o = layered(b);
	o.shellTakesGround = true;
	o.topsoilBlock = fogId; // not solid: the floor's shell must stay walkable
	o.outsideBlock = vox::kAir;
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	CHECK_EQ(w.get({0, 63, 0}), b.barrier);
	CHECK_EQ(w.get({50, 63, 0}), fogId); // outside cells still get the layer's block
	CHECK_EQ(w.get({0, 60, 0}), b.stone);
}

TEST(arena_shell_takes_ground_breaks_to_dug)
{
	const Blocks b = makeBlocks();
	ArenaOptions o = layered(b);
	o.shellTakesGround = true;
	World w;
	rl::buildSoccarArena(w, b.registry, o);
	REQUIRE(vox::breakBlock(w, b.registry, {0, 63, 0}) == vox::EditResult::Ok);
	CHECK(w.cell({0, 63, 0}) == CellState::Dug);
	CHECK_EQ(w.get({0, 63, 0}), vox::kAir);
	CHECK_EQ(w.get({1, 63, 0}), b.dirt); // the hole's wall
}
