#include "check.hpp"

#include "vox/edit.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using vox::Aabb;
using vox::BlockId;
using vox::BlockRegistry;
using vox::BlockType;
using vox::CellState;
using vox::EditResult;
using vox::Int3;
using vox::Interactor;
using vox::RayHit;
using vox::Vec3;
using vox::World;
using vox::kAir;

namespace {

struct Blocks
{
	BlockRegistry registry;
	BlockId stone = kAir;   ///< solid, visible, breakable
	BlockId barrier = kAir; ///< the arena shell: solid, invisible, breakable
	BlockId bedrock = kAir; ///< unbreakable
	BlockId flower = kAir;  ///< not solid
};

Blocks makeBlocks()
{
	Blocks b;
	BlockType t;
	t.name = "stone";
	b.stone = b.registry.add(t);
	t = BlockType{};
	t.name = "barrier";
	t.visible = false;
	b.barrier = b.registry.add(t);
	t = BlockType{};
	t.name = "bedrock";
	t.breakable = false;
	b.bedrock = b.registry.add(t);
	t = BlockType{};
	t.name = "poppy";
	t.solid = false;
	b.flower = b.registry.add(t);
	return b;
}

RayHit hitAt(Int3 cell, Int3 normal)
{
	RayHit hit;
	hit.cell = cell;
	hit.normal = normal;
	return hit;
}

const char* name(EditResult r)
{
	switch (r)
	{
	case EditResult::Ok:
		return "Ok";
	case EditResult::NoTarget:
		return "NoTarget";
	case EditResult::OutOfBounds:
		return "OutOfBounds";
	case EditResult::Unbreakable:
		return "Unbreakable";
	case EditResult::Occupied:
		return "Occupied";
	case EditResult::BlockedByEntity:
		return "BlockedByEntity";
	case EditResult::InvalidBlock:
		return "InvalidBlock";
	}
	return "?";
}

std::string str(const Int3& p)
{
	return "(" + std::to_string(p.x) + "," + std::to_string(p.y) + "," + std::to_string(p.z) + ")";
}

// Placing stone (or `block`) on top of a stone at (0, 60, 0), into the cube [0, 1] x [61, 62] x [0, 1], next to
// `entities`; in a fresh world each time. Checks that a refused place changes nothing.
EditResult placeNear(const std::vector<Aabb>& entities, bool flower = false)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 60, 0}, b.stone);
	const std::uint64_t seq = w.changeSeq();
	const BlockId block = flower ? b.flower : b.stone;
	const EditResult r = vox::placeBlock(w, b.registry, hitAt({0, 60, 0}, {0, 1, 0}), block, entities);
	if (r == EditResult::Ok)
		CHECK_EQ(w.get({0, 61, 0}), block);
	else
	{
		CHECK_EQ(w.get({0, 61, 0}), kAir);
		CHECK_EQ(w.changeSeq(), seq);
	}
	return r;
}

// The Interactor's test stage: the player at (0.5, 60.5, 0.5) looking +x at a stone wall block at (3, 60, 0), so a
// break hits `wall` and a place goes into `front`.
struct Scene
{
	Blocks b = makeBlocks();
	World world;
	Vec3 eye{0.5f, 60.5f, 0.5f};
	Vec3 look{1.0f, 0.0f, 0.0f};
	Int3 wall{3, 60, 0};
	Int3 front{2, 60, 0};
	std::vector<Aabb> entities;

	Scene() { world.set(wall, b.stone); }
};

// Plays one tick per character of `buttons` ('a' attack, 'u' use, 'b' both, '-' neither) and returns one character
// per tick: 'B' a break, 'P' a place, '.' nothing. With `restore`, the wall is put back and the placed block removed
// after every tick, so each tick sees the same target.
std::string play(Interactor& in, Scene& sc, const std::string& buttons, BlockId held, bool restore = true)
{
	std::string out;
	for (const char c : buttons)
	{
		Interactor::Input input;
		input.attack = c == 'a' || c == 'b';
		input.use = c == 'u' || c == 'b';
		const Interactor::Action action = in.tick(sc.world, sc.b.registry, sc.eye, sc.look, input, held, sc.entities);
		out += action.kind == Interactor::Action::Kind::Break ? 'B' : (action.kind == Interactor::Action::Kind::Place ? 'P' : '.');
		if (restore)
		{
			sc.world.set(sc.wall, sc.b.stone);
			sc.world.set(sc.front, kAir);
		}
	}
	return out;
}

Interactor withRepeats(int breakTicks, int placeTicks)
{
	Interactor::Config config;
	config.breakRepeatTicks = breakTicks;
	config.placeRepeatTicks = placeTicks;
	return Interactor(config);
}

} // namespace

// --- breakBlock ---

TEST(edit_break_sets_air)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({1, 60, 2}, b.stone);
	const std::uint64_t seq = w.changeSeq();
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {1, 60, 2})), std::string("Ok"));
	CHECK_EQ(w.get({1, 60, 2}), kAir);
	std::vector<vox::BlockChange> changes;
	REQUIRE(w.changesSince(seq, changes));
	REQUIRE(changes.size() == 1);
	CHECK_EQ(changes[0].before, b.stone);
	CHECK_EQ(changes[0].after, kAir);
	CHECK(w.cell({1, 60, 2}) == CellState::Open);
}

TEST(edit_break_outside_grid)
{
	const Blocks b = makeBlocks();
	World w;
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {64, 60, 0})), std::string("OutOfBounds"));
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 47, 0})), std::string("OutOfBounds"));
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 88, 0})), std::string("OutOfBounds"));
}

TEST(edit_break_air_is_unbreakable)
{
	const Blocks b = makeBlocks();
	World w;
	w.setCell({0, 63, 0}, CellState::Shell);
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 63, 0})), std::string("Unbreakable"));
	CHECK(w.cell({0, 63, 0}) == CellState::Shell); // nothing was dug
	CHECK_EQ(w.changeSeq(), static_cast<std::uint64_t>(0));
}

TEST(edit_break_unbreakable_type)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 60, 0}, b.bedrock);
	w.set({0, 63, 0}, b.bedrock);
	w.setCell({0, 63, 0}, CellState::Shell);
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 60, 0})), std::string("Unbreakable"));
	CHECK_EQ(w.get({0, 60, 0}), b.bedrock);
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 63, 0})), std::string("Unbreakable"));
	CHECK(w.cell({0, 63, 0}) == CellState::Shell);
}

TEST(edit_break_in_shell_cell_digs_it)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({5, 63, -7}, b.barrier);
	w.setCell({5, 63, -7}, CellState::Shell);
	w.takeDirtyCells();
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {5, 63, -7})), std::string("Ok"));
	CHECK_EQ(w.get({5, 63, -7}), kAir);
	CHECK(w.cell({5, 63, -7}) == CellState::Dug);
	CHECK(w.takeDirtyCells().has_value()); // the holes texture needs the update
}

TEST(edit_break_leaves_open_and_dug_cells_alone)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 70, 0}, b.stone);
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 70, 0})), std::string("Ok"));
	CHECK(w.cell({0, 70, 0}) == CellState::Open);
	w.setCell({0, 63, 0}, CellState::Dug);
	w.set({0, 63, 0}, b.stone); // built back into a hole
	w.takeDirtyCells();
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 63, 0})), std::string("Ok"));
	CHECK(w.cell({0, 63, 0}) == CellState::Dug);
	CHECK(!w.takeDirtyCells());
}

TEST(edit_break_any_breakable_block_in_shell_digs)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 63, 0}, b.flower);
	w.setCell({0, 63, 0}, CellState::Shell);
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 63, 0})), std::string("Ok"));
	CHECK(w.cell({0, 63, 0}) == CellState::Dug);
}

TEST(edit_break_unknown_id_is_breakable)
{
	const Blocks b = makeBlocks();
	World w;
	const BlockId unknown = static_cast<BlockId>(b.registry.size() + 10);
	w.set({0, 60, 0}, unknown);
	CHECK_EQ(name(vox::breakBlock(w, b.registry, {0, 60, 0})), std::string("Ok"));
	CHECK_EQ(w.get({0, 60, 0}), kAir);
}

// --- placeBlock ---

TEST(edit_place_against_each_face)
{
	const Blocks b = makeBlocks();
	const Int3 normals[] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
	for (const Int3& n : normals)
	{
		World w;
		w.set({0, 60, 0}, b.stone);
		CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 60, 0}, n), b.flower)), std::string("Ok"));
		CHECK_EQ(w.get(Int3{0, 60, 0} + n), b.flower);
		CHECK_EQ(w.get({0, 60, 0}), b.stone);
	}
}

TEST(edit_place_with_zero_normal_is_no_target)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 60, 0}, b.stone);
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 60, 0}, {0, 0, 0}), b.stone)), std::string("NoTarget"));
	CHECK_EQ(w.changeSeq(), static_cast<std::uint64_t>(1));
}

TEST(edit_place_outside_grid)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 87, 0}, b.stone);
	w.set({63, 60, 0}, b.stone);
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 87, 0}, {0, 1, 0}), b.stone)), std::string("OutOfBounds"));
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({63, 60, 0}, {1, 0, 0}), b.stone)), std::string("OutOfBounds"));
	CHECK_EQ(w.changeSeq(), static_cast<std::uint64_t>(2));
}

TEST(edit_place_into_occupied_cell)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 60, 0}, b.stone);
	w.set({0, 61, 0}, b.flower);
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 60, 0}, {0, 1, 0}), b.stone)), std::string("Occupied"));
	CHECK_EQ(w.get({0, 61, 0}), b.flower);
}

TEST(edit_place_invalid_block)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 60, 0}, b.stone);
	const RayHit top = hitAt({0, 60, 0}, {0, 1, 0});
	CHECK_EQ(name(vox::placeBlock(w, b.registry, top, kAir)), std::string("InvalidBlock"));
	CHECK_EQ(name(vox::placeBlock(w, b.registry, top, static_cast<BlockId>(b.registry.size()))), std::string("InvalidBlock"));
	CHECK_EQ(name(vox::placeBlock(w, b.registry, top, static_cast<BlockId>(65535))), std::string("InvalidBlock"));
	CHECK_EQ(w.get({0, 61, 0}), kAir);
}

TEST(edit_place_checks_in_documented_order)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 60, 0}, b.stone);
	w.set({0, 87, 0}, b.stone);
	w.set({1, 60, 0}, b.stone);
	const std::vector<Aabb> everywhere{{{-100, 0, -100}, {100, 100, 100}}};
	// NoTarget before everything else
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 60, 0}, {0, 0, 0}), kAir, everywhere)), std::string("NoTarget"));
	// OutOfBounds before Occupied / InvalidBlock / BlockedByEntity
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 87, 0}, {0, 1, 0}), kAir, everywhere)), std::string("OutOfBounds"));
	// Occupied before InvalidBlock / BlockedByEntity
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 60, 0}, {1, 0, 0}), kAir, everywhere)), std::string("Occupied"));
	// InvalidBlock before BlockedByEntity
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 60, 0}, {0, 1, 0}), kAir, everywhere)), std::string("InvalidBlock"));
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 60, 0}, {0, 1, 0}), b.stone, everywhere)), std::string("BlockedByEntity"));
}

TEST(edit_place_blocked_by_overlapping_entity)
{
	CHECK_EQ(name(placeNear({{{0.2f, 61.5f, 0.2f}, {0.8f, 63.3f, 0.8f}}})), std::string("BlockedByEntity"));
	CHECK_EQ(name(placeNear({{{-5, 50, -5}, {5, 70, 5}}})), std::string("BlockedByEntity"));           // contains the cube
	CHECK_EQ(name(placeNear({{{0.4f, 61.4f, 0.4f}, {0.6f, 61.6f, 0.6f}}})), std::string("BlockedByEntity")); // inside it
	CHECK_EQ(name(placeNear({{{0, 61, 0}, {1, 62, 1}}})), std::string("BlockedByEntity"));            // exactly it
	CHECK_EQ(name(placeNear({{{0.999f, 61, 0}, {2, 62, 1}}})), std::string("BlockedByEntity"));       // a sliver
	CHECK_EQ(name(placeNear({{{-1, 60, -1}, {0.001f, 61.001f, 0.001f}}})), std::string("BlockedByEntity")); // a corner
}

TEST(edit_place_touching_entities_dont_block)
{
	// each face, an edge and a corner of the cube [0, 1] x [61, 62] x [0, 1]
	const Aabb touching[] = {
		{{0, 62, 0}, {1, 64, 1}},
		{{0, 59, 0}, {1, 61, 1}},
		{{-1, 61, 0}, {0, 62, 1}},
		{{1, 61, 0}, {2, 62, 1}},
		{{0, 61, -1}, {1, 62, 0}},
		{{0, 61, 1}, {1, 62, 2}},
		{{1, 62, 0}, {2, 63, 1}},
		{{1, 62, 1}, {2, 63, 2}},
	};
	for (const Aabb& box : touching)
		CHECK_EQ(name(placeNear({box})), std::string("Ok"));
	CHECK_EQ(name(placeNear({{{0.2f, 61.2f, 1.5f}, {0.8f, 61.8f, 2.0f}}})), std::string("Ok")); // apart along z only
	CHECK_EQ(name(placeNear({{{0.2f, 61.5f, 0.2f}, {0.8f, 61.5f, 0.8f}}})), std::string("Ok")); // flat: no volume
	CHECK_EQ(name(placeNear({{{0.8f, 61.8f, 0.8f}, {0.2f, 61.2f, 0.2f}}})), std::string("Ok")); // inverted: empty
	CHECK_EQ(name(placeNear({})), std::string("Ok"));
}

TEST(edit_place_checks_every_entity)
{
	const std::vector<Aabb> entities{{{5, 60, 5}, {6, 62, 6}}, {{0.5f, 60, 0.5f}, {1.5f, 61.5f, 1.5f}}};
	CHECK_EQ(name(placeNear(entities)), std::string("BlockedByEntity"));
}

TEST(edit_place_non_solid_ignores_entities)
{
	CHECK_EQ(name(placeNear({{{-5, 50, -5}, {5, 70, 5}}}, true)), std::string("Ok"));
}

TEST(edit_place_keeps_cell_state)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({0, 62, 0}, b.stone);
	w.setCell({0, 63, 0}, CellState::Dug);
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 62, 0}, {0, 1, 0}), b.stone)), std::string("Ok"));
	CHECK_EQ(w.get({0, 63, 0}), b.stone);
	CHECK(w.cell({0, 63, 0}) == CellState::Dug);
	CHECK_EQ(name(vox::placeBlock(w, b.registry, hitAt({0, 63, 0}, {0, 1, 0}), b.stone)), std::string("Ok"));
	CHECK(w.cell({0, 64, 0}) == CellState::Open);
}

// --- Interactor ---

TEST(interactor_default_config)
{
	const Interactor in;
	CHECK_EQ(in.config().reach, 5.0f);
	CHECK_EQ(in.config().breakRepeatTicks, 6);
	CHECK_EQ(in.config().placeRepeatTicks, 4);
	const Interactor fast = withRepeats(2, 3);
	CHECK_EQ(fast.config().breakRepeatTicks, 2);
	CHECK_EQ(fast.config().placeRepeatTicks, 3);
}

TEST(interactor_target_includes_invisible_barrier)
{
	Scene sc;
	sc.world.set(sc.front, sc.b.barrier);
	const Interactor in;
	const auto hit = in.target(sc.world, sc.eye, sc.look);
	REQUIRE(hit.has_value());
	CHECK(hit->cell == sc.front);
	CHECK_EQ(hit->block, sc.b.barrier);
	CHECK(hit->normal == (Int3{-1, 0, 0}));
	CHECK_NEAR(hit->distance, 1.5, 1e-6);
}

TEST(interactor_target_within_reach_inclusive)
{
	const Blocks b = makeBlocks();
	World w;
	w.set({6, 60, 0}, b.stone);
	const Interactor in;
	const auto hit = in.target(w, {1.0f, 60.5f, 0.5f}, {1, 0, 0}); // the face at x = 6 is exactly 5 away
	REQUIRE(hit.has_value());
	CHECK_EQ(hit->distance, 5.0f);
	CHECK(!in.target(w, {std::nextafter(1.0f, 0.0f), 60.5f, 0.5f}, {1, 0, 0}));
	CHECK(!in.target(w, {1.0f, 60.5f, 0.5f}, {-1, 0, 0}));
	Interactor::Config config;
	config.reach = 6.0f;
	CHECK(Interactor(config).target(w, {0.5f, 60.5f, 0.5f}, {1, 0, 0}));
	config.reach = 5.4f;
	CHECK(!Interactor(config).target(w, {0.5f, 60.5f, 0.5f}, {1, 0, 0}));
}

TEST(interactor_idle_does_nothing)
{
	Scene sc;
	Interactor in;
	const std::uint64_t seq = sc.world.changeSeq();
	CHECK_EQ(play(in, sc, "-----", sc.b.stone, false), std::string("....."));
	CHECK_EQ(sc.world.changeSeq(), seq);
}

TEST(interactor_break_repeats_every_six_ticks)
{
	Scene sc;
	Interactor in;
	CHECK_EQ(play(in, sc, "aaaaaaaaaaaaaaaaaaaa", sc.b.stone), std::string("B.....B.....B.....B."));
}

TEST(interactor_place_repeats_every_four_ticks)
{
	Scene sc;
	Interactor in;
	CHECK_EQ(play(in, sc, "uuuuuuuuuuuuuuuuuuuu", sc.b.stone), std::string("P...P...P...P...P..."));
}

TEST(interactor_press_acts_at_once)
{
	Scene sc;
	Interactor in;
	CHECK_EQ(play(in, sc, "--a", sc.b.stone), std::string("..B"));
	Scene sc2;
	Interactor in2;
	CHECK_EQ(play(in2, sc2, "-u", sc2.b.stone), std::string(".P"));
}

TEST(interactor_release_resets_the_repeat)
{
	Scene sc;
	Interactor in;
	CHECK_EQ(play(in, sc, "a-a-a", sc.b.stone), std::string("B.B.B"));
	CHECK_EQ(play(in, sc, "-aaa-aaaaaaa", sc.b.stone), std::string(".B...B.....B"));
	CHECK_EQ(play(in, sc, "-uu-uuuuu", sc.b.stone), std::string(".P..P...P"));
}

TEST(interactor_break_and_place_actions)
{
	Scene sc;
	Interactor in;
	Interactor::Input use;
	use.use = true;
	auto action = in.tick(sc.world, sc.b.registry, sc.eye, sc.look, use, sc.b.flower);
	CHECK(action.kind == Interactor::Action::Kind::Place);
	CHECK(action.cell == sc.front);
	CHECK_EQ(action.before, kAir);
	CHECK_EQ(action.after, sc.b.flower);
	CHECK_EQ(sc.world.get(sc.front), sc.b.flower);

	Interactor::Input attack;
	attack.attack = true;
	action = in.tick(sc.world, sc.b.registry, sc.eye, sc.look, attack, sc.b.flower);
	CHECK(action.kind == Interactor::Action::Kind::Break);
	CHECK(action.cell == sc.front); // the flower is now the nearest block
	CHECK_EQ(action.before, sc.b.flower);
	CHECK_EQ(action.after, kAir);
	CHECK_EQ(sc.world.get(sc.front), kAir);

	action = in.tick(sc.world, sc.b.registry, sc.eye, sc.look, Interactor::Input{}, sc.b.flower);
	CHECK(action.kind == Interactor::Action::Kind::None);
	CHECK_EQ(action.before, kAir);
	CHECK_EQ(action.after, kAir);
}

TEST(interactor_failed_break_is_retried_next_tick)
{
	Scene sc;
	sc.world.set(sc.wall, kAir);
	Interactor in;
	CHECK_EQ(play(in, sc, "aaa", sc.b.stone, false), std::string("...")); // nothing in reach
	sc.world.set(sc.wall, sc.b.stone);
	CHECK_EQ(play(in, sc, "aaaaaaa", sc.b.stone), std::string("B.....B"));
}

TEST(interactor_unbreakable_target_is_retried_each_tick)
{
	Scene sc;
	sc.world.set(sc.front, sc.b.bedrock);
	Interactor in;
	CHECK_EQ(play(in, sc, "aa", sc.b.stone, false), std::string(".."));
	CHECK_EQ(sc.world.get(sc.front), sc.b.bedrock);
	sc.world.set(sc.front, kAir);
	CHECK_EQ(play(in, sc, "aaaaaaa", sc.b.stone), std::string("B.....B"));
}

TEST(interactor_failed_place_is_retried_next_tick)
{
	Scene sc;
	sc.entities.push_back({{2.25f, 60.0f, 0.25f}, {2.75f, 61.8f, 0.75f}}); // someone standing in `front`
	Interactor in;
	CHECK_EQ(play(in, sc, "uuu", sc.b.stone), std::string("..."));
	sc.entities.clear();
	CHECK_EQ(play(in, sc, "uuuuu", sc.b.stone), std::string("P...P"));
	// a non-solid block goes in regardless
	sc.entities.push_back({{2.25f, 60.0f, 0.25f}, {2.75f, 61.8f, 0.75f}});
	Interactor in2;
	CHECK_EQ(play(in2, sc, "u", sc.b.flower), std::string("P"));
}

TEST(interactor_empty_hand_places_nothing_until_a_block_is_held)
{
	Scene sc;
	Interactor in;
	CHECK_EQ(play(in, sc, "uu", kAir), std::string(".."));
	CHECK_EQ(sc.world.get(sc.front), kAir);
	CHECK_EQ(play(in, sc, "uuuuu", sc.b.stone), std::string("P...P")); // still held: acts on the first tick it can
}

TEST(interactor_attack_wins_when_both_are_held)
{
	Scene sc;
	Interactor in;
	CHECK_EQ(play(in, sc, "b", sc.b.stone, false), std::string("B"));
	CHECK_EQ(sc.world.get(sc.wall), kAir);
	CHECK_EQ(sc.world.get(sc.front), kAir);
	Scene sc2;
	Interactor in2;
	CHECK_EQ(play(in2, sc2, "bbbbbbbbbbbbb", sc2.b.stone), std::string("B.....B.....B"));
}

TEST(interactor_attack_blocks_use_even_while_cooling_down)
{
	Scene sc;
	Interactor in;
	// use pressed at tick 1 while attack is held: swallowed until attack is up, then placed at once
	CHECK_EQ(play(in, sc, "abbbbbu", sc.b.stone), std::string("B.....P"));
}

TEST(interactor_use_pressed_during_attack_acts_after_release)
{
	Scene sc;
	Interactor in;
	CHECK_EQ(play(in, sc, "abuuuuu", sc.b.stone), std::string("B.P...P"));
}

TEST(interactor_use_held_through_attack_keeps_its_countdown)
{
	Scene sc;
	Interactor in;
	// placed at tick 0; the attack at tick 1 doesn't restart use's countdown, so the next place is at tick 4
	CHECK_EQ(play(in, sc, "ubuuuuuuu", sc.b.stone), std::string("PB..P...P"));
	// and an attack held for longer than the countdown lets use act on the first tick it is up
	Scene sc2;
	Interactor in2;
	CHECK_EQ(play(in2, sc2, "ubbbbbbuuuu", sc2.b.stone), std::string("PB.....P..."));
}

TEST(interactor_attack_press_ignores_use_cooldown)
{
	Scene sc;
	Interactor in;
	CHECK_EQ(play(in, sc, "uub", sc.b.stone), std::string("P.B"));
}

TEST(interactor_repeat_of_one_or_less_acts_every_tick)
{
	for (const int ticks : {1, 0, -3})
	{
		Scene sc;
		Interactor in = withRepeats(ticks, ticks);
		CHECK_EQ(play(in, sc, "aaaa", sc.b.stone), std::string("BBBB"));
		CHECK_EQ(play(in, sc, "-uuuu", sc.b.stone), std::string(".PPPP"));
	}
	Scene sc;
	Interactor in = withRepeats(2, 3);
	CHECK_EQ(play(in, sc, "aaaaa-uuuuuu", sc.b.stone), std::string("B.B.B.P..P.."));
}

TEST(interactor_out_of_reach_does_nothing)
{
	Scene sc;
	Interactor::Config config;
	config.reach = 2.0f; // the wall's face is 2.5 away
	Interactor in(config);
	CHECK_EQ(play(in, sc, "aaa-uuu", sc.b.stone), std::string("......."));
	CHECK_EQ(sc.world.get(sc.wall), sc.b.stone);
}

TEST(interactor_digs_the_arena_floor_and_builds_on_it)
{
	// RL's floor at y = 64: barrier blocks in the shell cells at y = 63, ground below
	const Blocks b = makeBlocks();
	World w;
	w.fill({-3, 62, -3}, {3, 62, 3}, b.stone);
	w.fill({-3, 63, -3}, {3, 63, 3}, b.barrier);
	for (int x = -3; x <= 3; ++x)
		for (int z = -3; z <= 3; ++z)
			w.setCell({x, 63, z}, CellState::Shell);
	const Vec3 eye{0.5f, 65.62f, 0.5f};
	const Vec3 down{0.0f, -1.0f, 0.0f};
	Interactor in;
	Interactor::Input attack, use, none;
	attack.attack = true;
	use.use = true;

	const auto target = in.target(w, eye, down);
	REQUIRE(target.has_value());
	CHECK(target->cell == (Int3{0, 63, 0})); // the invisible floor is targetable

	// build on RL's floor
	auto action = in.tick(w, b.registry, eye, down, use, b.stone);
	CHECK(action.kind == Interactor::Action::Kind::Place);
	CHECK(action.cell == (Int3{0, 64, 0}));
	CHECK(w.cell({0, 64, 0}) == CellState::Open);

	// break it again, then dig through the floor
	in.tick(w, b.registry, eye, down, none, b.stone);
	action = in.tick(w, b.registry, eye, down, attack, b.stone);
	CHECK(action.kind == Interactor::Action::Kind::Break);
	CHECK(action.cell == (Int3{0, 64, 0}));
	for (int t = 1; t < 6; ++t)
		CHECK(in.tick(w, b.registry, eye, down, attack, b.stone).kind == Interactor::Action::Kind::None);
	action = in.tick(w, b.registry, eye, down, attack, b.stone);
	CHECK(action.kind == Interactor::Action::Kind::Break);
	CHECK(action.cell == (Int3{0, 63, 0}));
	CHECK_EQ(action.before, b.barrier);
	CHECK(w.cell({0, 63, 0}) == CellState::Dug);
	CHECK_EQ(w.get({0, 63, 0}), kAir);

	// the ground below the hole is the target now; building into the hole keeps it dug
	in.tick(w, b.registry, eye, down, none, b.stone);
	action = in.tick(w, b.registry, eye, down, use, b.flower);
	CHECK(action.kind == Interactor::Action::Kind::Place);
	CHECK(action.cell == (Int3{0, 63, 0}));
	CHECK_EQ(w.get({0, 63, 0}), b.flower);
	CHECK(w.cell({0, 63, 0}) == CellState::Dug);
	CHECK_EQ(str(target->cell), str(Int3{0, 63, 0}));
}
