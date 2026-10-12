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
/// cells outside the grid are air. Returns nothing for a zero direction, a non-finite input or a miss. Exactly
/// matches the order the cells are crossed in; when the ray passes exactly through an edge or corner, it steps
/// x before y before z.
std::optional<RayHit> raycast(const World& world, Vec3 origin, Vec3 dir, float maxDistance, const RayFilter& filter = {});

} // namespace vox
