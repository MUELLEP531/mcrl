#include "check.hpp"

#include "vox/raycast.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using vox::BlockId;
using vox::GridSpec;
using vox::Int3;
using vox::RayHit;
using vox::Vec3;
using vox::World;
using vox::raycast;

namespace {

constexpr BlockId kStone = 1;
constexpr BlockId kGlass = 2;
const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();
const double kSqrt2 = std::sqrt(2.0);

std::string str(const Int3& p)
{
	return "(" + std::to_string(p.x) + "," + std::to_string(p.y) + "," + std::to_string(p.z) + ")";
}

std::string str(const std::vector<Int3>& cells)
{
	std::string s;
	for (const Int3& c : cells)
		s += str(c);
	return s;
}

// Every cell the ray is offered to the filter in, in order, on a world filled with stone (so no cell is air).
std::vector<Int3> visits(const World& world, Vec3 origin, Vec3 dir, float maxDistance)
{
	std::vector<Int3> cells;
	const auto hit = raycast(world, origin, dir, maxDistance, [&](BlockId, Int3 cell) {
		cells.push_back(cell);
		return false;
	});
	CHECK(!hit);
	return cells;
}

World filledWorld(GridSpec spec)
{
	World w(spec);
	w.fill(spec.origin, spec.origin + spec.size - Int3{1, 1, 1}, kStone);
	return w;
}

// Deterministic pseudo-random numbers (splitmix64), identical on every platform.
struct Rng
{
	std::uint64_t state = 0;

	explicit Rng(std::uint64_t seed) : state(seed) {}

	std::uint64_t next()
	{
		std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		return z ^ (z >> 31);
	}
	double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); } // [0, 1)
	double range(double a, double b) { return a + (b - a) * uniform(); }
	int below(int n) { return static_cast<int>(next() % static_cast<std::uint64_t>(n)); }
};

struct Ray
{
	Vec3 origin, dir;
	float maxDistance = 0.0f;
};

constexpr double kEps = 1e-6;

// The exact (double) intersection of the ray's line with the closed unit cube of `cell`, along the unit direction `u`.
struct Slab
{
	bool touches = false; ///< the line meets the cube
	bool grazes = false;  ///< the line passes within kEps of an edge or corner, or runs within kEps of a face plane
	double tEnter = 0.0, tExit = 0.0;
	int enterAxis = -1;       ///< the axis of the entry plane (-1 if parallel to all the others)
	double secondEnter = 0.0; ///< the second latest entry plane: close to tEnter for an entry through an edge
};

Slab slab(const double o[3], const double u[3], Int3 cell)
{
	const double inf = std::numeric_limits<double>::infinity();
	const int c[3] = {cell.x, cell.y, cell.z};
	Slab s;
	s.tEnter = -inf;
	s.secondEnter = -inf;
	s.tExit = inf;
	bool parallelOutside = false;
	for (int a = 0; a < 3; ++a)
	{
		const double lo = c[a], hi = c[a] + 1.0;
		if (u[a] == 0.0)
		{
			if (o[a] < lo - kEps || o[a] > hi + kEps)
				return Slab{}; // clearly outside the slab it runs parallel to
			if (std::fabs(o[a] - lo) <= kEps || std::fabs(o[a] - hi) <= kEps)
				s.grazes = true;
			if (o[a] < lo || o[a] > hi)
				parallelOutside = true;
			continue;
		}
		double t0 = (lo - o[a]) / u[a], t1 = (hi - o[a]) / u[a];
		if (t0 > t1)
			std::swap(t0, t1);
		if (t0 > s.tEnter)
		{
			s.secondEnter = s.tEnter;
			s.tEnter = t0;
			s.enterAxis = a;
		}
		else if (t0 > s.secondEnter)
			s.secondEnter = t0;
		s.tExit = (std::min)(s.tExit, t1);
	}
	s.touches = !parallelOutside && s.tEnter <= s.tExit;
	if (std::fabs(s.tExit - s.tEnter) <= kEps)
		s.grazes = true;
	return s;
}

// How a cell relates to the part 0 <= t <= maxD of the ray: not at all, too close to call, or crossed with a chord
// longer than kEps starting at `entry`.
enum class Relation
{
	None,
	Marginal,
	Crossed,
};

Relation relate(const Slab& s, double maxD, double& entry)
{
	if (!s.touches && !s.grazes)
		return Relation::None;
	const double first = (std::min)(s.tEnter, s.tExit), last = (std::max)(s.tEnter, s.tExit);
	if (last < -kEps || first > maxD + kEps)
		return Relation::None;
	entry = (std::max)(first, 0.0);
	if (s.grazes || std::fabs(s.tEnter) <= kEps)
		return Relation::Marginal; // past an edge, along a face, or the origin on (or next to) the entry face
	const double margin = (std::min)(s.tExit, maxD) - entry;
	if (margin < -kEps)
		return Relation::None;
	return margin <= kEps ? Relation::Marginal : Relation::Crossed;
}

struct Expected
{
	bool ambiguous = false; ///< too close to call in double: skip the comparison
	bool hit = false;
	Int3 cell, normal;
	double distance = 0.0;
};

// What raycast must return, from slab tests against every candidate cell: the one the ray reaches first. Rays that pass
// within kEps of an edge, a corner, a face or the maxDistance limit near the answer are flagged ambiguous (generic
// rays almost never are).
Expected reference(const Ray& ray, const std::vector<Int3>& candidates)
{
	const double o[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
	const double d[3] = {ray.dir.x, ray.dir.y, ray.dir.z};
	const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	const double u[3] = {d[0] / len, d[1] / len, d[2] / len};
	const double maxD = ray.maxDistance;
	const double inf = std::numeric_limits<double>::infinity();

	Expected e;
	double best = inf, second = inf, marginal = inf, bestSecondEnter = -inf;
	int bestAxis = -1;
	for (const Int3& cell : candidates)
	{
		const Slab s = slab(o, u, cell);
		double entry = 0.0;
		const Relation r = relate(s, maxD, entry);
		if (r == Relation::Marginal)
			marginal = (std::min)(marginal, entry);
		if (r != Relation::Crossed)
			continue;
		if (entry < best)
		{
			second = best;
			best = entry;
			e.cell = cell;
			bestAxis = s.tEnter > 0.0 ? s.enterAxis : -1;
			bestSecondEnter = s.secondEnter;
		}
		else if (entry < second)
			second = entry;
	}
	if (best < inf)
	{
		e.hit = true;
		e.distance = best;
		if (bestAxis >= 0)
		{
			const int n = u[bestAxis] > 0 ? -1 : 1;
			e.normal = Int3{bestAxis == 0 ? n : 0, bestAxis == 1 ? n : 0, bestAxis == 2 ? n : 0};
			if (best - bestSecondEnter < kEps)
				e.ambiguous = true; // entered next to an edge: the face is down to the tie rule
		}
		if (second - best < kEps || marginal < best + kEps)
			e.ambiguous = true;
	}
	else if (marginal < inf)
		e.ambiguous = true;
	return e;
}

bool hasIntegralCoordinate(const Vec3& p)
{
	return std::floor(p.x) == p.x || std::floor(p.y) == p.y || std::floor(p.z) == p.z;
}

// A ray with a random origin around the grid (often outside it), a random unnormalised direction (sometimes with exact
// zero components: axis-parallel rays) and a random reach.
Ray randomRay(Rng& rng, const GridSpec& spec, double margin, double maxReach)
{
	Ray r;
	do
	{
		r.origin = Vec3{static_cast<float>(rng.range(spec.origin.x - margin, spec.origin.x + spec.size.x + margin)),
			static_cast<float>(rng.range(spec.origin.y - margin, spec.origin.y + spec.size.y + margin)),
			static_cast<float>(rng.range(spec.origin.z - margin, spec.origin.z + spec.size.z + margin))};
	} while (hasIntegralCoordinate(r.origin));
	do
	{
		float c[3];
		for (float& v : c)
			v = rng.below(6) == 0 ? 0.0f : static_cast<float>(rng.range(-1.0, 1.0));
		const double scale = std::pow(10.0, rng.range(-3.0, 3.0));
		r.dir = Vec3{static_cast<float>(c[0] * scale), static_cast<float>(c[1] * scale), static_cast<float>(c[2] * scale)};
	} while (r.dir.x == 0.0f && r.dir.y == 0.0f && r.dir.z == 0.0f);
	r.maxDistance = static_cast<float>(rng.range(0.0, maxReach));
	return r;
}

struct Tally
{
	int compared = 0, ambiguous = 0, hits = 0, misses = 0;
};

void compare(const World& world, const Ray& ray, const std::vector<Int3>& candidates, const vox::RayFilter& filter,
	Tally& tally)
{
	const Expected e = reference(ray, candidates);
	if (e.ambiguous)
	{
		++tally.ambiguous;
		return;
	}
	++tally.compared;
	const auto hit = raycast(world, ray.origin, ray.dir, ray.maxDistance, filter);
	CHECK_EQ(hit.has_value(), e.hit);
	if (!hit || !e.hit)
	{
		++tally.misses;
		return;
	}
	++tally.hits;
	CHECK_EQ(str(hit->cell), str(e.cell));
	CHECK_EQ(str(hit->normal), str(e.normal));
	CHECK_NEAR(hit->distance, e.distance, 1e-4 + 1e-6 * e.distance);
	CHECK_EQ(hit->block, world.get(hit->cell));
}


// The walk as the header describes it, done the slow way: from the origin's own cell, through every cell outside the
// grid too, with plain crossing arithmetic. raycast skips the outside part; its answers must be identical, ties and
// maxDistance boundaries included.
std::optional<RayHit> fullWalk(const World& world, Vec3 origin, Vec3 dir, float maxDistance)
{
	const double o[3] = {origin.x, origin.y, origin.z};
	const double d[3] = {dir.x, dir.y, dir.z};
	const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	const double inf = std::numeric_limits<double>::infinity();
	int c[3], step[3];
	double next[3];
	const auto exitPlane = [&](int a) { return step[a] > 0 ? c[a] + 1.0 : static_cast<double>(c[a]); };
	for (int a = 0; a < 3; ++a)
	{
		c[a] = static_cast<int>(std::floor(o[a]));
		step[a] = d[a] > 0 ? 1 : (d[a] < 0 ? -1 : 0);
		next[a] = step[a] != 0 ? (exitPlane(a) - o[a]) / d[a] : inf;
	}
	double s = -inf, dist = 0.0;
	int axis = -1;
	for (int i = 0; i < 10000; ++i)
	{
		const Int3 cell{c[0], c[1], c[2]};
		if (world.get(cell) != vox::kAir)
		{
			RayHit hit;
			hit.cell = cell;
			hit.block = world.get(cell);
			if (axis >= 0)
			{
				hit.normal = Int3{axis == 0 ? -step[0] : 0, axis == 1 ? -step[1] : 0, axis == 2 ? -step[2] : 0};
				hit.distance = static_cast<float>(dist);
			}
			return hit;
		}
		int a = 0;
		if (next[1] < next[a])
			a = 1;
		if (next[2] < next[a])
			a = 2;
		if (axis < 0 || next[a] != s)
		{
			dist = (exitPlane(a) - o[a]) * len / d[a];
			if (!(dist > 0.0))
				dist = 0.0;
		}
		if (dist > maxDistance)
			return std::nullopt;
		s = next[a];
		axis = a;
		c[a] += step[a];
		next[a] = (exitPlane(a) - o[a]) / d[a];
	}
	return std::nullopt;
}

} // namespace

TEST(raycast_rejects_zero_direction)
{
	World w;
	w.set({0, 60, 0}, kStone);
	CHECK(!raycast(w, {0.5f, 60.5f, 0.5f}, {0, 0, 0}, 10.0f));
	CHECK(!raycast(w, {0.5f, 60.5f, 0.5f}, {-0.0f, 0.0f, -0.0f}, 10.0f));
}

TEST(raycast_rejects_non_finite_input)
{
	World w;
	w.set({0, 60, 0}, kStone);
	const Vec3 o{0.5f, 60.5f, 0.5f};
	CHECK(raycast(w, o, {1, 0, 0}, 10.0f)); // sanity: the same ray with finite input hits
	CHECK(!raycast(w, {kNaN, 60.5f, 0.5f}, {1, 0, 0}, 10.0f));
	CHECK(!raycast(w, {0.5f, kInf, 0.5f}, {1, 0, 0}, 10.0f));
	CHECK(!raycast(w, {0.5f, 60.5f, -kInf}, {1, 0, 0}, 10.0f));
	CHECK(!raycast(w, o, {kNaN, 0, 0}, 10.0f));
	CHECK(!raycast(w, o, {0, kInf, 0}, 10.0f));
	CHECK(!raycast(w, o, {1, 0, -kInf}, 10.0f));
	CHECK(!raycast(w, o, {1, 0, 0}, kNaN));
	CHECK(!raycast(w, o, {1, 0, 0}, kInf)); // non-finite, even though "unlimited" might seem meant
}

TEST(raycast_rejects_negative_max_distance)
{
	World w;
	w.set({0, 60, 0}, kStone);
	const Vec3 o{0.5f, 60.5f, 0.5f};
	CHECK(!raycast(w, o, {1, 0, 0}, -1.0f));
	CHECK(!raycast(w, o, {1, 0, 0}, -std::numeric_limits<float>::denorm_min()));
	// zero (of either sign) still allows t = 0: the block the origin is in
	CHECK(raycast(w, o, {1, 0, 0}, 0.0f));
	CHECK(raycast(w, o, {1, 0, 0}, -0.0f));
}

TEST(raycast_starts_inside_block)
{
	World w;
	w.set({0, 60, 0}, kGlass);
	const auto hit = raycast(w, {0.25f, 60.75f, 0.5f}, {0.3f, -1, 0.2f}, 5.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{0, 60, 0}));
	CHECK(hit->normal == (Int3{0, 0, 0}));
	CHECK_EQ(hit->distance, 0.0f);
	CHECK(!std::signbit(hit->distance));
	CHECK_EQ(hit->block, kGlass);
}

TEST(raycast_hits_along_each_axis)
{
	World w;
	w.set({0, 60, 0}, kStone);
	struct Case
	{
		Vec3 origin, dir;
		Int3 normal;
	};
	const Case cases[] = {
		{{3.75f, 60.5f, 0.5f}, {-1, 0, 0}, {1, 0, 0}},
		{{-2.75f, 60.5f, 0.5f}, {1, 0, 0}, {-1, 0, 0}},
		{{0.5f, 63.75f, 0.5f}, {0, -1, 0}, {0, 1, 0}},
		{{0.5f, 57.25f, 0.5f}, {0, 1, 0}, {0, -1, 0}},
		{{0.5f, 60.5f, 3.75f}, {0, 0, -1}, {0, 0, 1}},
		{{0.5f, 60.5f, -2.75f}, {0, 0, 1}, {0, 0, -1}},
	};
	for (const Case& c : cases)
	{
		const auto hit = raycast(w, c.origin, c.dir, 10.0f);
		REQUIRE(hit.has_value());
		CHECK(hit->cell == (Int3{0, 60, 0}));
		CHECK_EQ(str(hit->normal), str(c.normal));
		CHECK_EQ(hit->distance, 2.75f);
		CHECK_EQ(hit->block, kStone);
	}
}

TEST(raycast_direction_need_not_be_unit)
{
	World w;
	w.set({0, 60, 0}, kStone);
	for (const float scale : {1e-3f, 0.37f, 7.0f, 1e4f, 1e30f, 1e-30f})
	{
		const auto hit = raycast(w, {3.75f, 60.5f, 0.5f}, {-scale, 0, 0}, 10.0f);
		REQUIRE(hit.has_value());
		CHECK_EQ(hit->distance, 2.75f); // axis-aligned: exact whatever the length
		CHECK(hit->normal == (Int3{1, 0, 0}));
	}
	// a diagonal: (3, 4, 0) has length 5; the ray (0.5 + 3s, 60.5 + 4s) enters x = 2 at s = 0.5, distance 2.5
	w.set({0, 60, 0}, vox::kAir);
	w.set({2, 62, 0}, kStone);
	for (const float scale : {1e-3f, 0.2f, 1.0f, 3e3f})
	{
		const auto hit = raycast(w, {0.5f, 60.5f, 0.5f}, {3 * scale, 4 * scale, 0}, 10.0f);
		REQUIRE(hit.has_value());
		CHECK(hit->cell == (Int3{2, 62, 0}));
		CHECK(hit->normal == (Int3{-1, 0, 0}));
		CHECK_NEAR(hit->distance, 2.5, 1e-6);
	}
}

TEST(raycast_max_distance_is_inclusive)
{
	World w;
	w.set({0, 60, 0}, kStone);
	const Vec3 o{3.75f, 60.5f, 0.5f};
	CHECK(raycast(w, o, {-1, 0, 0}, 2.75f));
	CHECK(!raycast(w, o, {-1, 0, 0}, std::nextafter(2.75f, 0.0f)));
	CHECK(raycast(w, o, {-3, 0, 0}, 2.75f)); // still exact with a non-unit direction
	CHECK(!raycast(w, o, {-3, 0, 0}, std::nextafter(2.75f, 0.0f)));
	CHECK(raycast(w, o, {-0.1f, 0, 0}, 2.75f));
	// through a corner: (3, 4, 0) from (0.5, 60, 0.5) reaches the corner (2, 62) at s = 0.5, distance exactly 2.5
	World v;
	v.set({2, 62, 0}, kStone);
	CHECK(raycast(v, {0.5f, 60.0f, 0.5f}, {3, 4, 0}, 2.5f));
	CHECK(!raycast(v, {0.5f, 60.0f, 0.5f}, {3, 4, 0}, std::nextafter(2.5f, 0.0f)));
}

TEST(raycast_misses_return_nothing)
{
	World w;
	CHECK(!raycast(w, {0.5f, 60.5f, 0.5f}, {1, 0.2f, -0.3f}, 1000.0f)); // empty world
	w.set({0, 60, 0}, kStone);
	CHECK(!raycast(w, {3.75f, 60.5f, 0.5f}, {1, 0, 0}, 1000.0f));   // pointing away
	CHECK(!raycast(w, {3.75f, 61.5f, 0.5f}, {-1, 0, 0}, 1000.0f));  // passing above
	CHECK(!raycast(w, {3.75f, 60.5f, 0.5f}, {-1, 0, 0}, 2.0f));     // too short
	CHECK(!raycast(w, {0.5f, 60.5f, 3.75f}, {0, 0, 1}, 3.4e38f));   // huge reach, still terminates
}

TEST(raycast_origin_outside_grid_enters_it)
{
	World w;
	w.set({63, 60, 0}, kStone);
	auto hit = raycast(w, {100.5f, 60.5f, 0.25f}, {-1, 0, 0}, 100.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{63, 60, 0}));
	CHECK(hit->normal == (Int3{1, 0, 0}));
	CHECK_EQ(hit->distance, 36.5f);
	CHECK(!raycast(w, {100.5f, 60.5f, 0.25f}, {-1, 0, 0}, 36.4f));

	w.set({5, 87, 5}, kStone);
	hit = raycast(w, {5.5f, 200.25f, 5.5f}, {0, -1, 0}, 200.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{5, 87, 5}));
	CHECK(hit->normal == (Int3{0, 1, 0}));
	CHECK_EQ(hit->distance, 112.25f);

	w.set({-64, 48, -64}, kGlass); // the grid's corner cell, from below and behind
	hit = raycast(w, {-64.25f, 40.5f, -64.5f}, {0.5f, 8, 1}, 100.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{-64, 48, -64}));
	CHECK(hit->normal == (Int3{0, -1, 0})); // x and z come into range at s = 0.5, y last at s = 7.5 / 8
	CHECK_NEAR(hit->distance, 7.5 / 8 * std::sqrt(65.25), 1e-5);
}

TEST(raycast_origin_outside_grid_misses)
{
	World w;
	w.fill({-64, 48, -64}, {63, 87, 63}, kStone);
	CHECK(!raycast(w, {100.5f, 60.5f, 0.5f}, {1, 0, 0}, 1000.0f));    // moving away
	CHECK(!raycast(w, {100.5f, 90.5f, 0.5f}, {-1, 0, 0}, 1000.0f));   // parallel, above the grid
	CHECK(!raycast(w, {100.5f, 60.5f, 0.5f}, {-1, 1, 0}, 1000.0f));   // over the top (y = 88) before x = 64
	CHECK(raycast(w, {100.5f, 60.5f, 0.5f}, {-1, 0.5f, 0}, 1000.0f));  // (a shallower one gets in)
	CHECK(!raycast(w, {70.5f, 60.5f, 70.5f}, {-1, 0, 1}, 1000.0f));   // past a corner
	CHECK(!raycast(w, {0.5f, 88.0f, 0.5f}, {1, 0, 1}, 1000.0f));      // in the plane of the top face: cells y = 88
	// x leaves the range at s = 1.5 (crossing x = 64), y enters it later at s = 2.5 (crossing y = 88)
	CHECK(!raycast(w, {62.5f, 90.5f, 0.5f}, {1, -1, 0}, 1000.0f));
}

TEST(raycast_enters_grid_through_an_edge_with_tie_rule)
{
	World w;
	// x and z both come into range at s = 2.5: x steps first (to an outside cell), then z enters the grid
	w.set({63, 60, 63}, kStone);
	auto hit = raycast(w, {66.5f, 60.5f, 66.5f}, {-1, 0, -1}, 100.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{63, 60, 63}));
	CHECK(hit->normal == (Int3{0, 0, 1}));
	CHECK_NEAR(hit->distance, 2.5 * kSqrt2, 1e-5);

	// z comes into range at s = 0.5 just as x crosses an inner plane: x steps first, so the walk enters the grid in
	// cell (-1, 60, 63); cell (0, 60, 63) is only touched along an edge and never visited
	World v;
	v.set({0, 60, 63}, kStone);
	CHECK(!raycast(v, {0.5f, 60.5f, 64.5f}, {-1, 0, -1}, 100.0f));
	v.set({-1, 60, 63}, kGlass);
	hit = raycast(v, {0.5f, 60.5f, 64.5f}, {-1, 0, -1}, 100.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{-1, 60, 63}));
	CHECK(hit->normal == (Int3{0, 0, 1}));
	CHECK_NEAR(hit->distance, 0.5 * kSqrt2, 1e-6);

	// x comes into range at s = 0.5 just as y crosses an inner plane: x first, so (63, 61, 0) is visited
	World u;
	u.set({63, 61, 0}, kStone);
	hit = raycast(u, {64.5f, 61.5f, 0.5f}, {-1, -1, 0}, 100.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{63, 61, 0}));
	CHECK(hit->normal == (Int3{1, 0, 0}));
	u.set({63, 61, 0}, vox::kAir);
	u.set({63, 60, 0}, kStone);
	hit = raycast(u, {64.5f, 61.5f, 0.5f}, {-1, -1, 0}, 100.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{63, 60, 0}));
	CHECK(hit->normal == (Int3{0, 1, 0}));
	CHECK_NEAR(hit->distance, 0.5 * kSqrt2, 1e-6);

	// x comes into range at s = 0.5 just as z crosses z = 1 going +z: x first, so (-64, 60, 0) is visited before z
	// steps (the walk must not start with z already past the plane it reaches exactly then)
	World t;
	t.set({-64, 60, 0}, kStone);
	hit = raycast(t, {-64.5f, 60.5f, 0.5f}, {1, 0, 1}, 100.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{-64, 60, 0}));
	CHECK(hit->normal == (Int3{-1, 0, 0}));
	CHECK_NEAR(hit->distance, 0.5 * kSqrt2, 1e-6);
	t.set({-64, 60, 0}, vox::kAir);
	t.set({-64, 60, 1}, kGlass);
	hit = raycast(t, {-64.5f, 60.5f, 0.5f}, {1, 0, 1}, 100.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{-64, 60, 1}));
	CHECK(hit->normal == (Int3{0, 0, -1}));
	CHECK_NEAR(hit->distance, 0.5 * kSqrt2, 1e-6);
	// the same from far away, where the cell estimate comes from a long multiply
	t.set({-64, 60, 1}, vox::kAir);
	t.set({-64, 60, 0}, kStone);
	hit = raycast(t, {-1064.5f, 60.5f, -999.5f}, {1, 0, 1}, 2000.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{-64, 60, 0}));
	CHECK(hit->normal == (Int3{-1, 0, 0}));
	CHECK_NEAR(hit->distance, 1000.5 * kSqrt2, 1e-3);
}

TEST(raycast_origin_on_boundary_belongs_to_floor_cell)
{
	World w;
	w.set({1, 60, 0}, kStone);
	// x = 1.0 is in cell 1, whichever way the ray goes
	for (const float dx : {1.0f, -1.0f})
	{
		const auto hit = raycast(w, {1.0f, 60.5f, 0.5f}, {dx, 0, 0}, 5.0f);
		REQUIRE(hit.has_value());
		CHECK(hit->cell == (Int3{1, 60, 0}));
		CHECK(hit->normal == (Int3{0, 0, 0}));
		CHECK_EQ(hit->distance, 0.0f);
	}
	// going -x from x = 1.0 crosses into cell 0 at once: entered at distance 0 through its +x face
	World v;
	v.set({0, 60, 0}, kStone);
	auto hit = raycast(v, {1.0f, 60.5f, 0.5f}, {-1, 0, 0}, 0.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{0, 60, 0}));
	CHECK(hit->normal == (Int3{1, 0, 0}));
	CHECK_EQ(hit->distance, 0.0f);
	CHECK(!std::signbit(hit->distance));
	CHECK(!raycast(v, {1.0f, 60.5f, 0.5f}, {1, 0, 0}, 5.0f));

	// the same on the grid's own faces: x = 64 is outside, x = -64 inside
	World g;
	g.set({63, 60, 0}, kStone);
	g.set({-64, 60, 0}, kStone);
	hit = raycast(g, {64.0f, 60.5f, 0.5f}, {-1, 0, 0}, 0.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{63, 60, 0}));
	CHECK(hit->normal == (Int3{1, 0, 0}));
	CHECK_EQ(hit->distance, 0.0f);
	CHECK(!raycast(g, {64.0f, 60.5f, 0.5f}, {1, 0, 0}, 5.0f));
	hit = raycast(g, {-64.0f, 60.5f, 0.5f}, {1, 0, 0}, 5.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{-64, 60, 0}));
	CHECK(hit->normal == (Int3{0, 0, 0}));
}

TEST(raycast_parallel_ray_in_a_face_plane)
{
	World w;
	w.set({3, 60, 0}, kStone); // below the plane y = 61: not hit
	CHECK(!raycast(w, {0.5f, 61.0f, 0.5f}, {1, 0, 0}, 10.0f));
	w.set({5, 61, 0}, kStone);
	const auto hit = raycast(w, {0.5f, 61.0f, 0.5f}, {1, 0, 0}, 10.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{5, 61, 0}));
	CHECK(hit->normal == (Int3{-1, 0, 0}));
	CHECK_EQ(hit->distance, 4.5f);
	// along an edge (x = 3, y = 61): the cell with both coordinates floored
	World v;
	v.set({2, 60, 4}, kStone);
	v.set({3, 60, 4}, kStone);
	v.set({2, 61, 4}, kStone);
	CHECK(!raycast(v, {3.0f, 61.0f, 0.5f}, {0, 0, 1}, 10.0f));
	v.set({3, 61, 4}, kGlass);
	const auto edge = raycast(v, {3.0f, 61.0f, 0.5f}, {0, 0, 1}, 10.0f);
	REQUIRE(edge.has_value());
	CHECK(edge->cell == (Int3{3, 61, 4}));
	CHECK(edge->normal == (Int3{0, 0, -1}));
	CHECK_EQ(edge->distance, 3.5f);
}

TEST(raycast_tie_rule_steps_x_before_y_before_z)
{
	const World w = filledWorld(GridSpec{{-4, -4, -4}, {8, 8, 8}});
	const Vec3 o{0.5f, 0.5f, 0.5f};
	// edge crossings (x, y) at s = 0.5 and 1.5
	CHECK_EQ(str(visits(w, o, {1, 1, 0}, 2.2f)),
		str(std::vector<Int3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {2, 1, 0}, {2, 2, 0}}));
	// corners
	CHECK_EQ(str(visits(w, o, {1, 1, 1}, 2.7f)),
		str(std::vector<Int3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {2, 1, 1}, {2, 2, 1}, {2, 2, 2}}));
	CHECK_EQ(str(visits(w, o, {-1, -1, -1}, 0.9f)),
		str(std::vector<Int3>{{0, 0, 0}, {-1, 0, 0}, {-1, -1, 0}, {-1, -1, -1}}));
	// the other pairs, with negative steps and unequal magnitudes
	CHECK_EQ(str(visits(w, o, {0, -2, 2}, 0.8f)), str(std::vector<Int3>{{0, 0, 0}, {0, -1, 0}, {0, -1, 1}}));
	CHECK_EQ(str(visits(w, o, {2, 0, -2}, 0.8f)), str(std::vector<Int3>{{0, 0, 0}, {1, 0, 0}, {1, 0, -1}}));
	CHECK_EQ(str(visits(w, o, {-1, 0, -1}, 0.8f)), str(std::vector<Int3>{{0, 0, 0}, {-1, 0, 0}, {-1, 0, -1}}));
	// (1, 2, 0) from y = 0: x = 1 and y = 1 are both crossed at s = 0.5, then y = 2 at s = 1
	CHECK_EQ(str(visits(w, {0.5f, 0.0f, 0.5f}, {1, 2, 0}, 2.3f)),
		str(std::vector<Int3>{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 2, 0}}));
}

TEST(raycast_tie_rule_decides_the_face_and_skips_edge_neighbours)
{
	World w;
	w.set({1, 61, 0}, kStone);
	const Vec3 o{0.5f, 60.5f, 0.5f};
	auto hit = raycast(w, o, {1, 1, 0}, 5.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->normal == (Int3{0, -1, 0})); // x stepped first, y last: entered through the bottom face
	CHECK_NEAR(hit->distance, 0.5 * kSqrt2, 1e-6);

	World v;
	v.set({1, 60, 0}, kStone); // visited on the way through the edge
	hit = raycast(v, o, {1, 1, 0}, 5.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{1, 60, 0}));
	CHECK(hit->normal == (Int3{-1, 0, 0}));
	CHECK_NEAR(hit->distance, 0.5 * kSqrt2, 1e-6);

	World u;
	u.set({0, 61, 0}, kStone); // only touched along the edge: y would have to step before x
	CHECK(!raycast(u, o, {1, 1, 0}, 5.0f));

	World c;
	c.set({1, 61, 1}, kStone); // through a corner: z steps last
	hit = raycast(c, o, {1, 1, 1}, 5.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->normal == (Int3{0, 0, -1}));
	CHECK_NEAR(hit->distance, 0.5 * std::sqrt(3.0), 1e-6);
}

TEST(raycast_long_rays_stay_exact)
{
	World w;
	w.set({-64, 60, 0}, kStone);
	auto hit = raycast(w, {-99999.5f, 60.5f, 0.5f}, {1, 0, 0}, 1e6f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{-64, 60, 0}));
	CHECK_EQ(hit->distance, 99935.5f);

	// a diagonal from far away, checked against the exact slab distance
	World v;
	v.set({10, 70, -20}, kStone);
	const Vec3 target{10.5f, 70.5f, -19.5f};
	const Vec3 dir{0.36f, 0.48f, -0.8f};
	const Ray ray{target - dir * 60000.0f, dir, 1e5f};
	const Expected e = reference(ray, {{10, 70, -20}});
	REQUIRE(e.hit && !e.ambiguous);
	hit = raycast(v, ray.origin, ray.dir, ray.maxDistance);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{10, 70, -20}));
	CHECK_EQ(str(hit->normal), str(e.normal));
	CHECK_NEAR(hit->distance, e.distance, 0.01);

	// across the whole grid from corner to corner: the far cell's entry is exact, not a sum of 300 steps
	World g;
	g.set({63, 87, 63}, kStone);
	const Vec3 from{-63.75f, 48.25f, -63.5f};
	const Vec3 d2{127.0f, 39.0f, 126.75f};
	const Ray across{from, d2, 1000.0f};
	const Expected e2 = reference(across, {{63, 87, 63}});
	REQUIRE(e2.hit && !e2.ambiguous);
	hit = raycast(g, from, d2, 1000.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{63, 87, 63}));
	CHECK_EQ(str(hit->normal), str(e2.normal));
	CHECK_NEAR(hit->distance, e2.distance, 2e-5);
}

TEST(raycast_extreme_direction_components)
{
	World w;
	w.set({4, 60, 0}, kStone);
	const Vec3 o{0.5f, 60.5f, 0.5f};
	auto hit = raycast(w, o, {1, 1e-30f, -1e-35f}, 10.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{4, 60, 0}));
	CHECK(hit->normal == (Int3{-1, 0, 0}));
	CHECK_NEAR(hit->distance, 3.5, 1e-6);
	hit = raycast(w, o, {std::numeric_limits<float>::denorm_min(), 0, 0}, 10.0f);
	REQUIRE(hit.has_value());
	CHECK_EQ(hit->distance, 3.5f);
	hit = raycast(w, o, {std::numeric_limits<float>::max(), 0, 0}, 10.0f);
	REQUIRE(hit.has_value());
	CHECK_EQ(hit->distance, 3.5f);
	hit = raycast(w, o, {3e38f, 3e38f, 0}, 10.0f); // the length overflows float, not double
	CHECK(!hit);                                      // the diagonal passes (4, 60, 0) by
	w.set({3, 63, 0}, kStone);
	hit = raycast(w, o, {3e38f, 3e38f, 0}, 10.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{3, 63, 0}));
}

TEST(raycast_far_origin_coordinates)
{
	World w;
	w.set({0, 60, 0}, kStone);
	// an origin so far away that a cell index would overflow int: still a clean answer
	CHECK(!raycast(w, {1e20f, 60.5f, 0.5f}, {1, 0, 0}, 1e30f));
	CHECK(!raycast(w, {3e38f, 60.5f, 0.5f}, {-1, 0, 0}, 1e30f)); // too far for maxDistance
	const auto hit = raycast(w, {3e38f, 60.5f, 0.5f}, {-1, 0, 0}, 3.4e38f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{63, 60, 0}) || hit->cell == (Int3{0, 60, 0}));
}

TEST(raycast_filter_chooses_blocks)
{
	World w;
	w.set({2, 60, 0}, kGlass);
	w.set({4, 60, 0}, kStone);
	std::vector<Int3> asked;
	std::vector<BlockId> blocks;
	const auto hit = raycast(w, {0.5f, 60.5f, 0.5f}, {1, 0, 0}, 10.0f, [&](BlockId block, Int3 cell) {
		asked.push_back(cell);
		blocks.push_back(block);
		return block != kGlass;
	});
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{4, 60, 0}));
	CHECK_EQ(hit->block, kStone);
	CHECK(hit->normal == (Int3{-1, 0, 0}));
	CHECK_EQ(hit->distance, 3.5f);
	// asked about non-air blocks only, in order
	CHECK_EQ(str(asked), str(std::vector<Int3>{{2, 60, 0}, {4, 60, 0}}));
	REQUIRE(blocks.size() == 2);
	CHECK_EQ(blocks[0], kGlass);
	CHECK_EQ(blocks[1], kStone);

	// a filter that accepts everything can't make air (or the outside) stop the ray
	CHECK(!raycast(World{}, {0.5f, 60.5f, 0.5f}, {1, 0, 0}, 10.0f, [](BlockId, Int3) { return true; }));
	CHECK(!raycast(World{}, {500.5f, 60.5f, 0.5f}, {-1, 0, 0}, 1000.0f, [](BlockId, Int3) { return true; }));
	// the start cell goes through the filter too
	const auto self = raycast(w, {2.5f, 60.5f, 0.5f}, {1, 0, 0}, 10.0f, [](BlockId b, Int3) { return b == kGlass; });
	REQUIRE(self.has_value());
	CHECK(self->cell == (Int3{2, 60, 0}));
	CHECK(self->normal == (Int3{0, 0, 0}));
	CHECK(!raycast(w, {0.5f, 60.5f, 0.5f}, {1, 0, 0}, 10.0f, [](BlockId, Int3) { return false; }));
}

TEST(raycast_custom_grid)
{
	World w(GridSpec{{100, -20, 7}, {5, 4, 3}});
	w.set({104, -17, 9}, kStone); // the far corner cell
	auto hit = raycast(w, {104.5f, -10.5f, 9.5f}, {0, -1, 0}, 10.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{104, -17, 9}));
	CHECK(hit->normal == (Int3{0, 1, 0}));
	CHECK_EQ(hit->distance, 5.5f);
	hit = raycast(w, {0.5f, -16.5f, 9.5f}, {1, 0, 0}, 200.0f);
	REQUIRE(hit.has_value());
	CHECK_EQ(hit->distance, 103.5f);
	// y = -16 is the grid's top plane, outside it; going down enters the corner cell at once
	hit = raycast(w, {104.5f, -16.0f, 9.5f}, {0, -1, 0}, 10.0f);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == (Int3{104, -17, 9}));
	CHECK(hit->normal == (Int3{0, 1, 0}));
	CHECK_EQ(hit->distance, 0.0f);
	CHECK(!raycast(w, {105.0f, -16.5f, 9.5f}, {0, 0, -1}, 10.0f)); // x = 105 is past the grid
}

TEST(raycast_visits_cells_in_brute_force_order)
{
	const GridSpec spec{{-3, 5, -4}, {7, 6, 8}};
	const World w = filledWorld(spec);
	std::vector<Int3> all;
	for (int y = spec.origin.y; y < spec.origin.y + spec.size.y; ++y)
		for (int z = spec.origin.z; z < spec.origin.z + spec.size.z; ++z)
			for (int x = spec.origin.x; x < spec.origin.x + spec.size.x; ++x)
				all.push_back({x, y, z});

	Rng rng(12345);
	int compared = 0, skipped = 0, longest = 0;
	for (int i = 0; i < 1500; ++i)
	{
		const Ray ray = randomRay(rng, spec, 6.0, 25.0);
		const double o[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
		const double d[3] = {ray.dir.x, ray.dir.y, ray.dir.z};
		const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
		const double u[3] = {d[0] / len, d[1] / len, d[2] / len};
		struct Visit
		{
			double entry;
			Int3 cell;
		};
		std::vector<Visit> expected;
		bool ambiguous = false;
		for (const Int3& cell : all)
		{
			double entry = 0.0;
			const Relation r = relate(slab(o, u, cell), ray.maxDistance, entry);
			if (r == Relation::Marginal)
				ambiguous = true;
			else if (r == Relation::Crossed)
				expected.push_back({entry, cell});
		}
		std::sort(expected.begin(), expected.end(), [](const Visit& a, const Visit& b) { return a.entry < b.entry; });
		for (std::size_t k = 1; k < expected.size(); ++k)
			if (expected[k].entry - expected[k - 1].entry < 1e-6)
				ambiguous = true;
		if (ambiguous)
		{
			++skipped;
			continue;
		}
		++compared;
		std::vector<Int3> cells;
		for (const Visit& v : expected)
			cells.push_back(v.cell);
		longest = (std::max)(longest, static_cast<int>(cells.size()));
		CHECK_EQ(str(visits(w, ray.origin, ray.dir, ray.maxDistance)), str(cells));
	}
	CHECK_EQ(compared + skipped, 1500);
	CHECK(compared > 1400);
	CHECK(longest > 10);
}

TEST(raycast_matches_brute_force_on_random_rays)
{
	const GridSpec spec{{-5, 2, -7}, {12, 9, 10}};
	Rng rng(2024);
	World w(spec);
	std::vector<Int3> stones, glass;
	for (int y = spec.origin.y; y < spec.origin.y + spec.size.y; ++y)
		for (int z = spec.origin.z; z < spec.origin.z + spec.size.z; ++z)
			for (int x = spec.origin.x; x < spec.origin.x + spec.size.x; ++x)
			{
				const int roll = rng.below(100);
				if (roll < 7)
				{
					w.set({x, y, z}, kStone);
					stones.push_back({x, y, z});
				}
				else if (roll < 14)
				{
					w.set({x, y, z}, kGlass);
					glass.push_back({x, y, z});
				}
			}
	std::vector<Int3> both = stones;
	both.insert(both.end(), glass.begin(), glass.end());
	const vox::RayFilter stoneOnly = [](BlockId b, Int3) { return b == kStone; };

	Tally any, filtered;
	for (int i = 0; i < 4000; ++i)
	{
		const Ray ray = randomRay(rng, spec, 3.0, 30.0);
		compare(w, ray, both, {}, any);
		compare(w, ray, stones, stoneOnly, filtered);
	}
	CHECK(any.compared > 3800);
	CHECK(filtered.compared > 3800);
	CHECK(any.hits > 700);
	CHECK(any.misses > 500);
	CHECK(filtered.hits > 400);
}

TEST(raycast_matches_brute_force_from_far_away)
{
	// the default grid, a scattering of blocks, origins up to ~1e4 blocks away aimed roughly at the grid
	World w;
	Rng rng(77);
	std::vector<Int3> blocks;
	for (int i = 0; i < 8000; ++i)
	{
		const Int3 c{rng.below(128) - 64, 48 + rng.below(40), rng.below(128) - 64};
		if (w.set(c, kStone))
			blocks.push_back(c);
	}
	Tally tally;
	for (int i = 0; i < 1500; ++i)
	{
		const double far = std::pow(10.0, rng.range(1.0, 4.0));
		const Vec3 aim{static_cast<float>(rng.range(-64, 64)), static_cast<float>(rng.range(48, 88)),
			static_cast<float>(rng.range(-64, 64))};
		Vec3 dir{static_cast<float>(rng.range(-1, 1)), static_cast<float>(rng.range(-1, 1)),
			static_cast<float>(rng.range(-1, 1))};
		if (rng.below(5) == 0)
			dir.y = 0.0f;
		if (vox::length(dir) < 0.1f)
			continue;
		Ray ray;
		ray.dir = dir;
		ray.origin = aim - dir * static_cast<float>(far / vox::length(dir));
		if (hasIntegralCoordinate(ray.origin))
			continue;
		ray.maxDistance = static_cast<float>(far + rng.range(-50, 150));
		compare(w, ray, blocks, {}, tally);
	}
	CHECK(tally.compared > 1300);
	CHECK(tally.hits > 700);
	CHECK(tally.misses > 100);
}

TEST(raycast_equals_full_walk_on_tie_rich_rays)
{
	// quarter-block origins, small integer directions and quarter-block reaches: rays through edges and corners, into
	// the grid through its edges, along face planes and ending exactly on a face are the rule here, not the exception
	const GridSpec spec{{-5, 2, -7}, {12, 9, 10}};
	Rng rng(99);
	World w(spec);
	for (int y = spec.origin.y; y < spec.origin.y + spec.size.y; ++y)
		for (int z = spec.origin.z; z < spec.origin.z + spec.size.z; ++z)
			for (int x = spec.origin.x; x < spec.origin.x + spec.size.x; ++x)
				if (rng.below(100) < 12)
					w.set({x, y, z}, rng.below(2) == 0 ? kStone : kGlass);
	const auto quarter = [&](int lo, int hi) { return static_cast<float>(lo * 4 + rng.below((hi - lo) * 4 + 1)) / 4.0f; };
	int hits = 0, outsideStarts = 0;
	for (int i = 0; i < 20000; ++i)
	{
		const Vec3 origin{quarter(-9, 11), quarter(-2, 15), quarter(-11, 7)};
		Vec3 dir{static_cast<float>(rng.below(7) - 3), static_cast<float>(rng.below(7) - 3),
			static_cast<float>(rng.below(7) - 3)};
		if (dir.x == 0.0f && dir.y == 0.0f && dir.z == 0.0f)
			continue;
		const float scale = rng.below(3) == 0 ? 0.1f : 1.0f;
		dir = dir * scale;
		const float maxDistance = quarter(0, 25);
		if (!w.contains(vox::cellAt(origin)))
			++outsideStarts;
		const auto expected = fullWalk(w, origin, dir, maxDistance);
		const auto hit = raycast(w, origin, dir, maxDistance);
		CHECK_EQ(hit.has_value(), expected.has_value());
		if (!hit || !expected)
			continue;
		++hits;
		CHECK_EQ(str(hit->cell), str(expected->cell));
		CHECK_EQ(str(hit->normal), str(expected->normal));
		CHECK_EQ(hit->distance, expected->distance);
		CHECK_EQ(hit->block, expected->block);
	}
	CHECK(hits > 2500);
	CHECK(outsideStarts > 10000);
}
