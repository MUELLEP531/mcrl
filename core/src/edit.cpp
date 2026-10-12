#include "vox/edit.hpp"

namespace vox {

namespace {

// Whether the unit cube of `cell` and `box` share a positive volume: their open intervals overlap on every axis.
// Touching faces, a flat box and an inverted (or NaN) box don't count.
bool overlaps(Int3 cell, const Aabb& box)
{
	const double lo[3] = {static_cast<double>(cell.x), static_cast<double>(cell.y), static_cast<double>(cell.z)};
	const double bmin[3] = {box.min.x, box.min.y, box.min.z};
	const double bmax[3] = {box.max.x, box.max.y, box.max.z};
	for (int a = 0; a < 3; ++a)
		if (!(bmin[a] < bmax[a] && bmin[a] < lo[a] + 1.0 && bmax[a] > lo[a]))
			return false;
	return true;
}

// A button's repeat countdown at the start of a tick: up resets it, a press (down now, up last tick) acts at once,
// holding counts down towards the next repeat. An attempt is made when it is <= 0; a success sets it to the repeat
// interval (so a held button acts every that many ticks; an interval below 1 means every tick), a failure leaves it
// as it is, so the next tick tries again.
void countDown(bool down, bool wasDown, int& cooldown)
{
	if (!down || !wasDown)
		cooldown = 0;
	else if (cooldown > 0)
		--cooldown;
}

} // namespace

EditResult breakBlock(World& world, const BlockRegistry& registry, Int3 cell)
{
	if (!world.contains(cell))
		return EditResult::OutOfBounds;
	const BlockId block = world.get(cell);
	if (block == kAir)
		return EditResult::Unbreakable;
	// An id the registry doesn't know (a world saved with another block set) is breakable: better than a block nobody
	// can ever remove.
	if (registry.contains(block) && !registry.get(block).breakable)
		return EditResult::Unbreakable;
	world.set(cell, kAir);
	// Digging the host's surface opens it; Open and Dug cells keep their state.
	if (world.cell(cell) == CellState::Shell)
		world.setCell(cell, CellState::Dug);
	return EditResult::Ok;
}

EditResult placeBlock(World& world, const BlockRegistry& registry, const RayHit& hit, BlockId block,
	const std::vector<Aabb>& entities)
{
	// Checked in this order, so the first reason that applies is the one reported.
	if (hit.normal == Int3{})
		return EditResult::NoTarget;
	const Int3 target = hit.cell + hit.normal;
	if (!world.contains(target))
		return EditResult::OutOfBounds;
	if (world.get(target) != kAir)
		return EditResult::Occupied;
	if (block == kAir || !registry.contains(block))
		return EditResult::InvalidBlock;
	if (registry.get(block).solid)
		for (const Aabb& entity : entities)
			if (overlaps(target, entity))
				return EditResult::BlockedByEntity;
	// The cell state is left alone: a Dug cell stays Dug (the host's surface stays open around the new block).
	world.set(target, block);
	return EditResult::Ok;
}

std::optional<RayHit> Interactor::target(const World& world, Vec3 eye, Vec3 look) const
{
	// The default filter stops at any non-air block, the invisible arena barrier included: that is what lets a player
	// dig into the host's floor and build on it.
	return raycast(world, eye, look, config_.reach);
}

Interactor::Action Interactor::tick(World& world, const BlockRegistry& registry, Vec3 eye, Vec3 look, Input input,
	BlockId held, const std::vector<Aabb>& entities)
{
	countDown(input.attack, attackWasDown_, attackCooldown_);
	countDown(input.use, useWasDown_, useCooldown_);
	attackWasDown_ = input.attack;
	useWasDown_ = input.use;

	Action action;
	if (input.attack)
	{
		// Attack wins while it is held, whether or not it acts this tick: use does nothing. The use button's state
		// is still tracked above (its countdown keeps running, a press resets it), so a use held through an attack
		// acts on the first tick attack is up once its own countdown has run out, and a use pressed during an
		// attack is not lost. Tapping attack therefore never speeds up placing.
		if (attackCooldown_ <= 0)
		{
			const std::optional<RayHit> hit = target(world, eye, look);
			if (hit)
			{
				const BlockId before = world.get(hit->cell);
				if (breakBlock(world, registry, hit->cell) == EditResult::Ok)
				{
					action.kind = Action::Kind::Break;
					action.cell = hit->cell;
					action.before = before;
					action.after = kAir;
					attackCooldown_ = config_.breakRepeatTicks;
				}
			}
		}
		return action;
	}

	if (input.use && useCooldown_ <= 0)
	{
		const std::optional<RayHit> hit = target(world, eye, look);
		if (hit && placeBlock(world, registry, *hit, held, entities) == EditResult::Ok)
		{
			action.kind = Action::Kind::Place;
			action.cell = hit->cell + hit->normal;
			action.before = kAir;
			action.after = held;
			useCooldown_ = config_.placeRepeatTicks;
		}
	}
	return action;
}

} // namespace vox
