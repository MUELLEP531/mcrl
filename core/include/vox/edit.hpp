// Breaking and placing blocks, with Minecraft's creative-mode rules and timing.
#pragma once

#include "vox/block_registry.hpp"
#include "vox/raycast.hpp"
#include "vox/types.hpp"
#include "vox/world.hpp"

#include <optional>
#include <vector>

namespace vox {

enum class EditResult
{
	Ok,
	NoTarget,        ///< nothing within reach
	OutOfBounds,     ///< the cell is outside the grid
	Unbreakable,     ///< air, or a type with breakable = false
	Occupied,        ///< placing into a cell that is not air
	BlockedByEntity, ///< a solid block would overlap a player, car or ball
	InvalidBlock,    ///< placing air or an id the registry doesn't have
};

/// Breaks the block at `cell` (sets it to air). Breaking a block in a Shell cell digs it: the cell becomes Dug.
EditResult breakBlock(World& world, const BlockRegistry& registry, Int3 cell);

/// Places `block` against the face `hit` points at (into hit.cell + hit.normal). A hit with a zero normal (the ray
/// started inside the block) places nothing (NoTarget). The target must be inside the grid and air; a solid block must
/// not overlap any of `entities` (touching faces don't count). Placing into a Dug cell is allowed and leaves it Dug.
EditResult placeBlock(World& world, const BlockRegistry& registry, const RayHit& hit, BlockId block,
	const std::vector<Aabb>& entities = {});

/// Turns held mouse buttons into breaks and places at Minecraft's creative-mode pace, one tick at a time (20 ticks per
/// second): pressing acts at once, holding repeats every `breakRepeatTicks` / `placeRepeatTicks`.
class Interactor
{
public:
	struct Config
	{
		float reach = 5.0f;        ///< blocks; Minecraft's creative reach
		int breakRepeatTicks = 6;  ///< ticks between breaks while attack is held
		int placeRepeatTicks = 4;  ///< ticks between places while use is held
	};

	struct Input
	{
		bool attack = false; ///< break (left mouse)
		bool use = false;    ///< place (right mouse)
	};

	struct Action
	{
		enum class Kind { None, Break, Place } kind = Kind::None;
		Int3 cell;                         ///< the cell broken or placed into
		BlockId before = kAir, after = kAir;
	};

	Interactor() = default;
	explicit Interactor(Config config) : config_(config) {}

	/// One tick. `eye` and `look` are the player's eye position and view direction; `held` the block in hand (air:
	/// nothing to place). If both buttons are held, attack wins. Returns what happened (None if nothing changed,
	/// including when an attempt failed: a failed press is retried on the next tick while the button is held).
	Action tick(World& world, const BlockRegistry& registry, Vec3 eye, Vec3 look, Input input, BlockId held,
		const std::vector<Aabb>& entities = {});

	/// The block the player looks at (for drawing a selection outline), if any.
	std::optional<RayHit> target(const World& world, Vec3 eye, Vec3 look) const;

	const Config& config() const { return config_; }

private:
	Config config_;
	int attackCooldown_ = 0;
	int useCooldown_ = 0;
	bool attackWasDown_ = false;
	bool useWasDown_ = false;
};

} // namespace vox
