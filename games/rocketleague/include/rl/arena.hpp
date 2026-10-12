// Rocket League's standard soccar arena as voxels: the shell (barrier blocks in the cells holding RL's floor, walls,
// corners, goals and ceiling, so a player can walk on RL's surfaces and dig into them) and the ground under it.
// This is the only Rocket League specific part of the voxel world; it replaces mcrl/host/arena.py.
#pragma once

#include "vox/block_registry.hpp"
#include "vox/types.hpp"
#include "vox/world.hpp"

namespace rl {

/// RL's world axes: x across the pitch, y along it (towards the goals), z up; Unreal units (uu).
struct RlVec
{
	float x = 0.0f, y = 0.0f, z = 0.0f;
};

/// How RL's units map to the voxel world. Must match the plugin's camera conversion (HostCamRow / HostCamPos):
/// minecraft x = rl.x / uuPerBlock, minecraft z = rl.y / uuPerBlock, minecraft y = floorY + rl.z / uuPerBlock.
/// (Swapping two axes turns RL's left-handed axes into Minecraft's right-handed ones.) The defaults fit the arena into
/// the shader's 128 x 40 x 128 grid with 16 blocks of ground below the floor; they are NOT yet checked against the
/// plugin's own values (MCBridge.cpp / arena.py are not in this repo yet).
struct ArenaMapping
{
	float uuPerBlock = 100.0f;
	float floorY = 64.0f;
};

vox::Vec3 toVoxel(const RlVec& p, const ArenaMapping& map);
RlVec toRl(const vox::Vec3& p, const ArenaMapping& map);

/// Standard soccar dimensions (uu). Sources: RLBot wiki "Useful game values" and RocketSim's arena.
struct SoccarDimensions
{
	float halfWidth = 4096.0f;    ///< side walls at |x| = 4096
	float halfLength = 5120.0f;   ///< back walls at |y| = 5120
	float height = 2044.0f;       ///< ceiling
	float cornerSum = 8064.0f;    ///< 45-degree corner walls: |x| + |y| <= 8064
	float goalHalfWidth = 893.0f; ///< goal mouth |x| <= 893
	float goalHeight = 642.775f;  ///< goal mouth z <= 642.775
	float goalDepth = 880.0f;     ///< goals reach |y| = 5120 + 880
};

/// Whether an RL point is inside the playable volume (the pitch plus both goals). Closed: points on a surface count
/// as inside. Ignores the curved wall-to-floor ramps (radius ~256 uu, under 3 blocks).
bool insideSoccar(const RlVec& p, const SoccarDimensions& dims = {});

struct ArenaOptions
{
	ArenaMapping mapping;
	SoccarDimensions dims;
	/// The block placed in shell cells: solid, invisible, breakable (digging it makes the cell Dug).
	vox::BlockId shellBlock = vox::kAir;
	/// Cells outside the arena below the floor's level: the top `topsoilDepth` layers get `topsoilBlock`, the rest
	/// `groundBlock`. kAir leaves them empty.
	vox::BlockId topsoilBlock = vox::kAir;
	vox::BlockId groundBlock = vox::kAir;
	int topsoilDepth = 3;
	/// Cells outside the arena at or above the floor's level (behind the walls and above the ceiling).
	vox::BlockId outsideBlock = vox::kAir;
};

/// Rebuilds the whole world as the arena: every cell is classified by testing insideSoccar() at its 8 corners.
/// All corners inside: open air inside the arena. Some inside and some outside: a Shell cell holding shellBlock.
/// None inside: outside the arena (ground or outsideBlock). So a surface exactly on a cell boundary puts the shell in
/// the cell behind it (the floor at y = floorY gives shell cells at y = floorY - 1), as the shader expects.
/// Every cell's state is reset first (Dug cells become Shell again). Throws std::invalid_argument if shellBlock is
/// air or not in `registry`, or a ground block is not in `registry`.
void buildSoccarArena(vox::World& world, const vox::BlockRegistry& registry, const ArenaOptions& options);

} // namespace rl
