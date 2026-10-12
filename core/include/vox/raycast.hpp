// Finding the block a player looks at: a ray walked through the grid cell by cell.
#pragma once

#include "vox/types.hpp"
#include "vox/world.hpp"

#include <functional>
#include <optional>

namespace vox {

struct RayHit
{
	Int3 cell;           ///< the block hit
	Int3 normal;         ///< unit axis vector of the face the ray entered through; (0,0,0) if it started inside
	float distance = 0;  ///< along the ray, in blocks, to where it entered `cell` (0 if it started inside)
	BlockId block = kAir;
};

/// Which blocks stop the ray. The default (an empty function) stops at any non-air block.
using RayFilter = std::function<bool(BlockId block, Int3 cell)>;

/// Walks the ray `origin + t * dir` (dir need not be unit length; it is normalised, so distances are in blocks) for
/// 0 <= t <= maxDistance and returns the first cell whose block passes `filter`. The origin may be outside the grid;
/// cells outside the grid are air. Returns nothing for a zero direction, a non-finite input (an infinite maxDistance
/// included), a negative maxDistance or a miss. Air never stops the ray and is never offered to `filter`.
/// The start cell is floor(origin) on each axis. Exactly matches the order the cells are crossed in; when the ray
/// passes exactly through an edge or corner, it steps x before y before z. (MCPassthrough.fx's trace_voxels breaks
/// such exact ties the other way, z before y before x; camera rays practically never tie exactly, and the GPU's float
/// stepping would blur an exact tie anyway, so the block targeted and the face drawn agree in practice.)
std::optional<RayHit> raycast(const World& world, Vec3 origin, Vec3 dir, float maxDistance, const RayFilter& filter = {});

} // namespace vox
