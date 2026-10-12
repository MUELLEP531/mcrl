#include "rl/arena.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// Ground layering (cells outside the arena, i.e. with none of their 8 corners inside it). The floor's level is the
// cell layer L = floor(mapping.floorY): cells with y >= L are not entirely below RL's floor and get outsideBlock
// (behind the walls, above the ceiling); cells L - topsoilDepth <= y < L get topsoilBlock; cells y < L - topsoilDepth
// get groundBlock. With an integer floorY (the default 64) the top topsoil layer is the floor's own shell layer
// (y = floorY - 1): under the pitch it is shell, around the arena it is topsoil, so the ground continues level with
// RL's floor. All of this is decided by y alone; shell and open cells never hold ground.

namespace rl {

namespace {

void validate(const vox::BlockRegistry& registry, const ArenaOptions& options)
{
	if (options.shellBlock == vox::kAir)
		throw std::invalid_argument("buildSoccarArena: shellBlock must not be air");
	if (!registry.contains(options.shellBlock))
		throw std::invalid_argument("buildSoccarArena: shellBlock is not in the registry");
	if (!registry.contains(options.topsoilBlock))
		throw std::invalid_argument("buildSoccarArena: topsoilBlock is not in the registry");
	if (!registry.contains(options.groundBlock))
		throw std::invalid_argument("buildSoccarArena: groundBlock is not in the registry");
	// not named by the header, but an unknown id written into the world would reach the shader's MCTYPES lookup
	if (!registry.contains(options.outsideBlock))
		throw std::invalid_argument("buildSoccarArena: outsideBlock is not in the registry");
	if (options.topsoilDepth < 0)
		throw std::invalid_argument("buildSoccarArena: topsoilDepth must not be negative");
	if (!std::isfinite(options.mapping.uuPerBlock) || !(options.mapping.uuPerBlock > 0.0f))
		throw std::invalid_argument("buildSoccarArena: mapping.uuPerBlock must be positive and finite");
	if (!std::isfinite(options.mapping.floorY))
		throw std::invalid_argument("buildSoccarArena: mapping.floorY must be finite");
}

} // namespace

vox::Vec3 toVoxel(const RlVec& p, const ArenaMapping& map)
{
	const double uu = map.uuPerBlock;
	vox::Vec3 v;
	v.x = static_cast<float>(static_cast<double>(p.x) / uu);
	v.y = static_cast<float>(static_cast<double>(map.floorY) + static_cast<double>(p.z) / uu);
	v.z = static_cast<float>(static_cast<double>(p.y) / uu);
	return v;
}

RlVec toRl(const vox::Vec3& p, const ArenaMapping& map)
{
	const double uu = map.uuPerBlock;
	RlVec r;
	r.x = static_cast<float>(static_cast<double>(p.x) * uu);
	r.y = static_cast<float>(static_cast<double>(p.z) * uu);
	r.z = static_cast<float>((static_cast<double>(p.y) - static_cast<double>(map.floorY)) * uu);
	return r;
}

bool insideSoccar(const RlVec& p, const SoccarDimensions& dims)
{
	// In double: every float and the sum of two of them are exact, so a point on a surface is decided exactly.
	// Written as positive conditions so a NaN coordinate is outside.
	const double ax = std::fabs(static_cast<double>(p.x));
	const double ay = std::fabs(static_cast<double>(p.y));
	const double z = p.z;
	const double halfLength = dims.halfLength;
	const bool pitch = ax <= dims.halfWidth && ay <= halfLength && z >= 0.0 && z <= dims.height
		&& ax + ay <= dims.cornerSum;
	if (pitch)
		return true;
	return ax <= dims.goalHalfWidth && ay >= halfLength && ay <= halfLength + static_cast<double>(dims.goalDepth)
		&& z >= 0.0 && z <= dims.goalHeight;
}

void buildSoccarArena(vox::World& world, const vox::BlockRegistry& registry, const ArenaOptions& options)
{
	validate(registry, options);

	const vox::GridSpec& spec = world.spec();
	const std::size_t sx = static_cast<std::size_t>(spec.size.x);
	const std::size_t sy = static_cast<std::size_t>(spec.size.y);
	const std::size_t sz = static_cast<std::size_t>(spec.size.z);

	// Each lattice point is shared by up to 8 cells: test it once. Index i + cx * (k + cz * j) for the point at
	// origin + (i, j, k).
	const std::size_t cx = sx + 1, cy = sy + 1, cz = sz + 1;
	std::vector<unsigned char> corner(cx * cy * cz);
	for (std::size_t j = 0; j < cy; ++j)
		for (std::size_t k = 0; k < cz; ++k)
			for (std::size_t i = 0; i < cx; ++i)
			{
				vox::Vec3 p;
				p.x = static_cast<float>(spec.origin.x + static_cast<int>(i));
				p.y = static_cast<float>(spec.origin.y + static_cast<int>(j));
				p.z = static_cast<float>(spec.origin.z + static_cast<int>(k));
				corner[i + cx * (k + cz * j)] = insideSoccar(toRl(p, options.mapping), options.dims) ? 1 : 0;
			}

	// What an outside cell holds depends only on its layer (see the comment at the top).
	const double floorLevel = std::floor(static_cast<double>(options.mapping.floorY));
	const double topsoilBottom = floorLevel - static_cast<double>(options.topsoilDepth);
	std::vector<vox::BlockId> outsideByLayer(sy);
	for (std::size_t j = 0; j < sy; ++j)
	{
		const double y = static_cast<double>(spec.origin.y + static_cast<int>(j));
		outsideByLayer[j] = y >= floorLevel ? options.outsideBlock
			: (y >= topsoilBottom ? options.topsoilBlock : options.groundBlock);
	}

	// Fresh vectors (every cell's state reset, Dug included), installed at once: World::assign makes readers resync.
	const std::size_t count = sx * sy * sz;
	std::vector<vox::BlockId> blocks(count, vox::kAir);
	std::vector<vox::CellState> cells(count, vox::CellState::Open);
	const std::size_t dk = cx, dj = cx * cz; // lattice strides for +z and +y
	std::size_t index = 0;
	for (std::size_t j = 0; j < sy; ++j)
		for (std::size_t k = 0; k < sz; ++k)
			for (std::size_t i = 0; i < sx; ++i, ++index)
			{
				const std::size_t c = i + cx * (k + cz * j);
				const int inside = corner[c] + corner[c + 1] + corner[c + dk] + corner[c + dk + 1] + corner[c + dj]
					+ corner[c + dj + 1] + corner[c + dj + dk] + corner[c + dj + dk + 1];
				if (inside == 8)
					continue; // open air inside the arena
				if (inside > 0)
				{
					blocks[index] = options.shellBlock;
					cells[index] = vox::CellState::Shell;
				}
				else
					blocks[index] = outsideByLayer[j];
			}

	world.assign(std::move(blocks), std::move(cells));
}

} // namespace rl
