#include "vox/raycast.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vox {

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// One axis of the walk (Amanatides & Woo). Crossings are measured in "s", the parameter of the caller's unnormalised
// direction, and each one is computed directly as (boundary - origin) / dir rather than by adding a step to the last:
// nothing accumulates along a long ray, and two crossings that coincide exactly (the ray passes through an edge or a
// corner) give bit-identical values, so the tie rule below sees a true tie. Normalising first would round the
// direction and could split such a tie. The inputs are floats, so boundary - origin is exact in double for any sane
// origin (one float's 24 bits next to the boundary's integer bits) and a quotient is correctly rounded: equal crossings
// stay equal, and a strict order can only collapse into a tie when two crossings are within one double ulp.
struct AxisWalk
{
	double o = 0.0;   // origin
	double d = 0.0;   // direction (unnormalised, as given)
	int lo = 0;       // the grid's cells on this axis are lo .. hi - 1
	int hi = 0;
	int step = 0;     // +1, -1, or 0 for a ray parallel to this axis
	int c = 0;        // the walk's current cell
	double next = 0;  // s of the next crossing out of cell c (infinity if step == 0)

	bool inRange(int cell) const { return cell >= lo && cell < hi; }
	double crossing(double boundary) const { return (boundary - o) / d; }
	// The plane the walk crosses to enter / to leave `cell` (the cell on the far side of a plane is the one the walk
	// enters, which is what makes a point exactly on a boundary belong to floor(coordinate)).
	double entryBoundary(int cell) const { return step > 0 ? static_cast<double>(cell) : cell + 1.0; }
	double exitBoundary(int cell) const { return step > 0 ? cell + 1.0 : static_cast<double>(cell); }
	void updateNext() { next = step != 0 ? crossing(exitBoundary(c)) : kInfinity; }
};

// The origin's cell is outside the grid. Cells there are air, which never stops the ray, so instead of walking them one
// by one this puts the walk into the exact state a full walk from the origin's cell would reach just before its first
// step into the grid: every crossing with s < sEnter done, none with s == sEnter (those are left to the walk loop, so a
// grid entry through an edge or corner still steps x before y before z). sEnter is the last axis' crossing into the
// grid's range; the crossings are computed with the same expression the walk loop uses, so the state is consistent
// with it bit for bit. Returns false if the ray never enters the grid.
bool skipToGrid(AxisWalk (&axes)[3])
{
	bool outside[3] = {false, false, false};
	double sIn[3] = {-kInfinity, -kInfinity, -kInfinity};
	double sOut[3] = {kInfinity, kInfinity, kInfinity};
	double sEnter = -kInfinity;
	for (int a = 0; a < 3; ++a)
	{
		AxisWalk& w = axes[a];
		outside[a] = !(w.o >= w.lo && w.o < w.hi);
		if (w.step == 0)
		{
			if (outside[a])
				return false; // parallel to the grid's faces and outside its slab on this axis
			continue;
		}
		const bool approaching = w.step > 0 ? w.o < w.lo : w.o >= w.hi;
		if (outside[a] && !approaching)
			return false; // past the grid on this axis and moving away
		if (outside[a])
			sIn[a] = w.crossing(w.step > 0 ? w.lo : w.hi);
		sOut[a] = w.crossing(w.step > 0 ? w.hi : w.lo);
		sEnter = (std::max)(sEnter, sIn[a]);
	}
	// some axis leaves the grid's range before the last one enters it
	for (int a = 0; a < 3; ++a)
		if (sOut[a] < sEnter)
			return false;

	for (int a = 0; a < 3; ++a)
	{
		AxisWalk& w = axes[a];
		if (w.step == 0)
		{
			w.c = static_cast<int>(std::floor(w.o));
			w.updateNext();
			continue;
		}
		if (outside[a] && sIn[a] == sEnter)
		{
			// one step short of the grid; its crossing in (at sEnter) is the walk loop's to make
			w.c = w.step > 0 ? w.lo - 1 : w.hi;
			w.updateNext();
			continue;
		}
		// Inside the grid's range by sEnter: the cell after every crossing with s < sEnter. Start from an estimate
		// (clamped to the cells the walk can be in) and settle it with the walk's own crossing values.
		const int first = outside[a] ? (w.step > 0 ? w.lo : w.hi - 1) : static_cast<int>(std::floor(w.o));
		const int last = w.step > 0 ? w.hi - 1 : w.lo;
		const double guess = std::floor(w.o + sEnter * w.d);
		const double low = (std::min)(first, last), high = (std::max)(first, last);
		int c = static_cast<int>((std::min)((std::max)(guess, low), high));
		while (c != first && w.crossing(w.entryBoundary(c)) >= sEnter)
			c -= w.step;
		// bounded by `last`: its exit is sOut, which is >= sEnter
		while (w.crossing(w.exitBoundary(c)) < sEnter)
			c += w.step;
		w.c = c;
		w.updateNext();
	}
	return true;
}

} // namespace

std::optional<RayHit> raycast(const World& world, Vec3 origin, Vec3 dir, float maxDistance, const RayFilter& filter)
{
	const float inputs[7] = {origin.x, origin.y, origin.z, dir.x, dir.y, dir.z, maxDistance};
	for (const float v : inputs)
		if (!std::isfinite(v))
			return std::nullopt;
	if (maxDistance < 0.0f)
		return std::nullopt; // no t satisfies 0 <= t <= maxDistance
	if (dir.x == 0.0f && dir.y == 0.0f && dir.z == 0.0f)
		return std::nullopt;

	const GridSpec& spec = world.spec();
	const double o[3] = {origin.x, origin.y, origin.z};
	const double d[3] = {dir.x, dir.y, dir.z};
	const int lo[3] = {spec.origin.x, spec.origin.y, spec.origin.z};
	const int size[3] = {spec.size.x, spec.size.y, spec.size.z};
	// in double, the squares of float components can neither overflow nor underflow
	const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
	const double maxDist = maxDistance;

	AxisWalk axes[3];
	bool startsInGrid = true;
	for (int a = 0; a < 3; ++a)
	{
		AxisWalk& w = axes[a];
		w.o = o[a];
		w.d = d[a];
		w.lo = lo[a];
		w.hi = lo[a] + size[a];
		w.step = d[a] > 0.0 ? 1 : (d[a] < 0.0 ? -1 : 0);
		if (!(w.o >= w.lo && w.o < w.hi))
			startsInGrid = false;
	}
	if (startsInGrid)
	{
		for (AxisWalk& w : axes)
		{
			w.c = static_cast<int>(std::floor(w.o));
			w.updateNext();
		}
	}
	else if (!skipToGrid(axes))
		return std::nullopt;

	double s = -kInfinity; // s of the last step
	double dist = 0.0;     // distance (in blocks) of the last step
	int axis = -1;         // the axis of the last step; -1: still in the start cell
	for (;;)
	{
		bool inside = true;
		for (const AxisWalk& w : axes)
		{
			if (w.inRange(w.c))
				continue;
			inside = false;
			// each coordinate only moves one way, so once past the grid on some axis the ray never comes back
			if (w.step == 0 || (w.step > 0) == (w.c >= w.hi))
				return std::nullopt;
		}
		if (inside)
		{
			const Int3 cell{axes[0].c, axes[1].c, axes[2].c};
			const BlockId block = world.blocks()[world.indexOf(cell)];
			// air never stops the ray (outside the grid everything is air, so skipping it there is exact)
			if (block != kAir && (!filter || filter(block, cell)))
			{
				RayHit hit;
				hit.cell = cell;
				hit.block = block;
				if (axis >= 0)
				{
					// the entered face looks back along the step
					hit.normal = Int3{axis == 0 ? -axes[0].step : 0, axis == 1 ? -axes[1].step : 0,
						axis == 2 ? -axes[2].step : 0};
					hit.distance = static_cast<float>(dist);
				}
				return hit;
			}
		}

		// The next crossing; on a tie the strict comparisons keep the lower axis: x before y before z.
		int a = 0;
		if (axes[1].next < axes[a].next)
			a = 1;
		if (axes[2].next < axes[a].next)
			a = 2;
		AxisWalk& w = axes[a];
		if (w.step == 0)
			return std::nullopt; // unreachable: some axis has a finite crossing
		if (axis < 0 || w.next != s)
		{
			// The distance along the normalised ray, with no normalised direction (and its rounding) in between: for
			// an axis-aligned ray (len == |d|) or one like (3, 4, 0) whose crossing distance is a float, every step is
			// exact, so the inclusive maxDistance test has no rounding at its boundary.
			dist = (w.exitBoundary(w.c) - w.o) * len / w.d;
			if (!(dist > 0.0))
				dist = 0.0; // -0 for an origin on the boundary, moving back across it
		}
		// else: the second (or third) step through the same edge or corner point: the same distance, so cells entered
		// through one point never disagree about maxDistance
		if (dist > maxDist)
			return std::nullopt;
		s = w.next;
		axis = a;
		w.c += w.step;
		w.updateNext();
	}
}

} // namespace vox
