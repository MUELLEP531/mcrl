#include "check.hpp"

#include "vox/serialize.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using vox::BlockId;
using vox::BlockRegistry;
using vox::BlockType;
using vox::CellState;
using vox::DirtyRect;
using vox::GridSpec;
using vox::Int3;
using vox::LoadReport;
using vox::World;

namespace {

BlockType named(const std::string& name)
{
	BlockType t;
	t.name = name;
	return t;
}

/// stone 1, dirt 2, barrier 3 (invisible), glass 4, plank 5.
BlockRegistry registryA()
{
	BlockRegistry r;
	r.add(named("stone"));
	r.add(named("dirt"));
	BlockType barrier = named("barrier");
	barrier.visible = false;
	r.add(barrier);
	r.add(named("glass"));
	r.add(named("plank"));
	return r;
}

/// The same names in another order, plus one more: glass 1, plank 2, sand 3, barrier 4, dirt 5, stone 6.
BlockRegistry registryReordered()
{
	BlockRegistry r;
	r.add(named("glass"));
	r.add(named("plank"));
	r.add(named("sand"));
	BlockType barrier = named("barrier");
	barrier.visible = false;
	r.add(barrier);
	r.add(named("dirt"));
	r.add(named("stone"));
	return r;
}

std::string save(const World& w, const BlockRegistry& r)
{
	std::ostringstream out(std::ios::binary);
	vox::saveWorld(w, r, out);
	return out.str();
}

World load(const std::string& bytes, const BlockRegistry& r, GridSpec spec = {}, LoadReport* report = nullptr)
{
	std::istringstream in(bytes, std::ios::binary);
	return vox::loadWorld(in, r, spec, report);
}

template <typename F>
bool throwsRuntimeError(F f)
{
	try
	{
		f();
	}
	catch (const std::runtime_error&)
	{
		return true;
	}
	catch (...)
	{
		return false;
	}
	return false;
}

bool loadFails(const std::string& bytes, const BlockRegistry& r, GridSpec spec = {})
{
	return throwsRuntimeError([&] { load(bytes, r, spec); });
}

std::uint32_t lcg(std::uint32_t& state)
{
	state = state * 1664525u + 1013904223u;
	return state >> 8;
}

/// Bitwise CRC-32 (IEEE 802.3, as zlib), independent of the table version in serialize.cpp.
std::uint32_t crc32(const std::string& bytes)
{
	std::uint32_t crc = 0xffffffffu;
	for (const char ch : bytes)
	{
		crc ^= static_cast<unsigned char>(ch);
		for (int k = 0; k < 8; ++k)
			crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
	}
	return ~crc;
}

std::uint32_t readU32(const std::string& s, std::size_t at)
{
	std::uint32_t v = 0;
	for (int b = 3; b >= 0; --b)
		v = (v << 8) | static_cast<unsigned char>(s[at + static_cast<std::size_t>(b)]);
	return v;
}

std::uint16_t readU16(const std::string& s, std::size_t at)
{
	return static_cast<std::uint16_t>(static_cast<unsigned char>(s[at]) | (static_cast<unsigned char>(s[at + 1]) << 8));
}

int readI32(const std::string& s, std::size_t at)
{
	const std::uint32_t u = readU32(s, at);
	return u <= 0x7fffffffu ? static_cast<int>(u) : -static_cast<int>(~u) - 1;
}

/// Builds a save byte by byte, as the format comment in serialize.cpp lays it out.
struct Bytes
{
	std::string s;

	Bytes& u8(std::uint32_t v)
	{
		s.push_back(static_cast<char>(v & 0xffu));
		return *this;
	}
	Bytes& u16(std::uint32_t v) { return u8(v).u8(v >> 8); }
	Bytes& u32(std::uint32_t v) { return u16(v & 0xffffu).u16(v >> 16); }
	Bytes& i32(int v) { return u32(static_cast<std::uint32_t>(v)); }
	Bytes& name(const std::string& n)
	{
		u16(static_cast<std::uint32_t>(n.size()));
		s += n;
		return *this;
	}
	Bytes& header(GridSpec g, std::uint32_t paletteSize)
	{
		s += "MCRLVOX1";
		i32(g.origin.x).i32(g.origin.y).i32(g.origin.z).i32(g.size.x).i32(g.size.y).i32(g.size.z);
		return u32(paletteSize);
	}
	Bytes& blockRun(std::uint32_t count, std::uint32_t index) { return u32(count).u16(index); }
	Bytes& cellRun(std::uint32_t count, std::uint32_t state) { return u32(count).u8(state); }
	std::string withCrc() const
	{
		Bytes b = *this;
		b.u32(crc32(s));
		return b.s;
	}
};

/// 2 x 3 x 2 = 12 cells.
const GridSpec kTiny{{1, 2, 3}, {2, 3, 2}};

/// A valid tiny save: palette air, stone, dirt; 5 air, 4 stone, 3 dirt; 6 open, 3 shell, 3 dug.
Bytes tinyBody()
{
	Bytes b;
	b.header(kTiny, 3).name("air").name("stone").name("dirt");
	b.blockRun(5, 0).blockRun(4, 1).blockRun(3, 2);
	b.cellRun(6, 0).cellRun(3, 1).cellRun(3, 2);
	return b;
}

/// A small grid where neighbouring cells mostly differ: many runs.
World patterned(const BlockRegistry& r)
{
	World w(GridSpec{{-3, 5, 2}, {6, 4, 5}});
	std::vector<BlockId> blocks(w.blocks().size());
	std::vector<CellState> cells(w.cells().size());
	std::uint32_t seed = 7;
	for (std::size_t i = 0; i < blocks.size(); ++i)
	{
		blocks[i] = static_cast<BlockId>(lcg(seed) % r.size());
		cells[i] = static_cast<CellState>(lcg(seed) % 3);
	}
	w.assign(blocks, cells);
	return w;
}

/// The default grid with some building and dug arena cells.
World built()
{
	World w;
	w.fill({-64, 48, -64}, {63, 63, 63}, 1);  // stone floor
	w.fill({-10, 64, -10}, {10, 64, 10}, 2);  // dirt
	w.fill({-64, 64, -64}, {-64, 87, 63}, 3); // barrier wall
	w.set({0, 65, 0}, 4);
	w.set({63, 87, 63}, 5);
	w.set({-64, 48, -64}, 0);
	for (int x = -64; x < 64; ++x)
		for (int z = -64; z < 64; ++z)
			w.setCell({x, 63, z}, CellState::Shell);
	for (int x = -2; x <= 2; ++x)
		for (int z = -2; z <= 2; ++z)
			w.setCell({x, 63, z}, CellState::Dug);
	w.setCell({63, 87, 63}, CellState::Dug);
	return w;
}

std::vector<std::string> paletteOf(const std::string& bytes)
{
	std::vector<std::string> names;
	const std::uint32_t count = readU32(bytes, 32);
	std::size_t at = 36;
	for (std::uint32_t i = 0; i < count; ++i)
	{
		const std::uint16_t length = readU16(bytes, at);
		names.push_back(bytes.substr(at + 2, length));
		at += 2 + static_cast<std::size_t>(length);
	}
	return names;
}

} // namespace

// ---- the format ----

TEST(serialize_test_crc_matches_zlib)
{
	CHECK_EQ(crc32("123456789"), 0xcbf43926u);
	CHECK_EQ(crc32(""), 0u);
}

TEST(serialize_header_layout)
{
	const BlockRegistry reg = registryA();
	const World w = patterned(reg);
	const std::string bytes = save(w, reg);
	REQUIRE(bytes.size() > 40);
	CHECK_EQ(bytes.substr(0, 8), std::string("MCRLVOX1"));
	CHECK_EQ(readI32(bytes, 8), -3);
	CHECK_EQ(readI32(bytes, 12), 5);
	CHECK_EQ(readI32(bytes, 16), 2);
	CHECK_EQ(readI32(bytes, 20), 6);
	CHECK_EQ(readI32(bytes, 24), 4);
	CHECK_EQ(readI32(bytes, 28), 5);
	// every type is used by the pattern; names in id order
	const std::vector<std::string> names = paletteOf(bytes);
	CHECK((names == std::vector<std::string>{"air", "stone", "dirt", "barrier", "glass", "plank"}));
	// the trailer is the CRC-32 of everything before it
	CHECK_EQ(readU32(bytes, bytes.size() - 4), crc32(bytes.substr(0, bytes.size() - 4)));
}

TEST(serialize_tiny_save_is_byte_exact)
{
	// pins the documented layout: saving the world the hand-built bytes describe gives those bytes
	BlockRegistry reg;
	reg.add(named("stone"));
	reg.add(named("dirt"));
	reg.add(named("unused"));
	World w(kTiny);
	std::vector<BlockId> blocks = {0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2};
	std::vector<CellState> cells = {CellState::Open, CellState::Open, CellState::Open, CellState::Open, CellState::Open,
		CellState::Open, CellState::Shell, CellState::Shell, CellState::Shell, CellState::Dug, CellState::Dug,
		CellState::Dug};
	w.assign(blocks, cells);
	CHECK(save(w, reg) == tinyBody().withCrc());
	const World back = load(tinyBody().withCrc(), reg, kTiny);
	CHECK(back.blocks() == blocks);
	CHECK(back.cells() == cells);
}

TEST(serialize_palette_only_names_in_use)
{
	const BlockRegistry reg = registryA();
	World w(kTiny);
	w.fill({1, 2, 3}, {2, 4, 4}, 4); // glass everywhere: no air
	w.set({1, 2, 3}, 2);
	CHECK((paletteOf(save(w, reg)) == std::vector<std::string>{"dirt", "glass"}));
	const World air(kTiny);
	CHECK((paletteOf(save(air, reg)) == std::vector<std::string>{"air"}));
}

TEST(serialize_all_air_world_is_small)
{
	const BlockRegistry reg = registryA();
	const std::string bytes = save(World(), reg);
	// header 36 + "air" 5 + one block run 6 + one cell run 5 + crc 4
	CHECK_EQ(bytes.size(), static_cast<std::size_t>(56));
	const World back = load(bytes, reg);
	CHECK(back.blocks() == World().blocks());
}

// ---- round trips ----

TEST(serialize_round_trip_default_grid)
{
	const BlockRegistry reg = registryA();
	const World w = built();
	LoadReport report;
	report.unknownBlocks.push_back("stale");
	report.cellsDropped = 99;
	const World back = load(save(w, reg), reg, GridSpec{}, &report);
	CHECK(back.spec() == w.spec());
	CHECK(back.blocks() == w.blocks());
	CHECK(back.cells() == w.cells());
	CHECK(report.unknownBlocks.empty()); // overwritten
	CHECK_EQ(report.cellsDropped, static_cast<std::size_t>(0));
	CHECK(back.cell({0, 63, 0}) == CellState::Dug);
	CHECK(back.cell({30, 63, 0}) == CellState::Shell);
	CHECK(back.cell({0, 70, 0}) == CellState::Open);
	CHECK_EQ(back.get({-64, 70, 0}), 3); // the invisible barrier is saved like any block
}

TEST(serialize_round_trip_many_runs)
{
	const BlockRegistry reg = registryA();
	const World w = patterned(reg);
	const World back = load(save(w, reg), reg, w.spec());
	CHECK(back.blocks() == w.blocks());
	CHECK(back.cells() == w.cells());
}

TEST(serialize_round_trip_extreme_grid_numbers)
{
	const BlockRegistry reg = registryA();
	const GridSpec spec{{INT_MIN, INT_MAX, -1}, {1, 2, 3}};
	World w(spec);
	std::vector<BlockId> blocks = {1, 2, 3, 4, 5, 0};
	std::vector<CellState> cells = {CellState::Dug, CellState::Open, CellState::Shell, CellState::Shell,
		CellState::Dug, CellState::Open};
	w.assign(blocks, cells);
	const std::string bytes = save(w, reg);
	CHECK_EQ(readI32(bytes, 8), INT_MIN);
	CHECK_EQ(readI32(bytes, 12), INT_MAX);
	CHECK_EQ(readI32(bytes, 16), -1);
	const World back = load(bytes, reg, spec);
	CHECK(back.blocks() == blocks);
	CHECK(back.cells() == cells);
}

TEST(serialize_reordered_registry_loads_by_name)
{
	const BlockRegistry a = registryA();
	const BlockRegistry b = registryReordered();
	const World w = built();
	LoadReport report;
	const World back = load(save(w, a), b, GridSpec{}, &report);
	CHECK(report.unknownBlocks.empty());
	const BlockId map[] = {0, 6, 5, 4, 1, 2}; // registryA id -> registryReordered id
	bool same = true;
	for (std::size_t i = 0; i < w.blocks().size(); ++i)
		same = same && back.blocks()[i] == map[w.blocks()[i]];
	CHECK(same);
	CHECK(back.cells() == w.cells());
	CHECK_EQ(back.get({0, 50, 0}), 6); // stone
	CHECK_EQ(back.get({0, 65, 0}), 1); // glass
}

TEST(serialize_unknown_names_load_as_air_and_are_reported)
{
	const BlockRegistry a = registryA();
	BlockRegistry onlyStone;
	onlyStone.add(named("stone"));
	World w(kTiny);
	w.set({1, 2, 3}, 5); // plank
	w.set({2, 2, 3}, 2); // dirt
	w.set({1, 3, 3}, 2); // dirt
	w.set({2, 3, 3}, 1); // stone
	w.set({1, 4, 4}, 5); // plank
	w.set({2, 4, 4}, 4); // glass
	w.setCell({2, 4, 4}, CellState::Dug);
	LoadReport report;
	const World back = load(save(w, a), onlyStone, kTiny, &report);
	// palette order (id order of registryA), each once
	CHECK((report.unknownBlocks == std::vector<std::string>{"dirt", "glass", "plank"}));
	CHECK_EQ(report.cellsDropped, static_cast<std::size_t>(5));
	CHECK_EQ(back.get({2, 3, 3}), 1);
	CHECK_EQ(back.get({1, 2, 3}), 0);
	CHECK_EQ(back.get({2, 2, 3}), 0);
	CHECK_EQ(back.get({2, 4, 4}), 0);
	CHECK(back.cell({2, 4, 4}) == CellState::Dug); // cell states don't depend on names
}

TEST(serialize_unknown_ids_are_saved_as_air)
{
	const BlockRegistry reg = registryA();
	World w(kTiny);
	w.set({1, 2, 3}, 77);    // not in the registry
	w.set({2, 2, 3}, 65535); // not in the registry
	w.fill({1, 3, 3}, {2, 4, 4}, 1);
	const std::string bytes = save(w, reg);
	CHECK((paletteOf(bytes) == std::vector<std::string>{"air", "stone"}));
	LoadReport report;
	const World back = load(bytes, reg, kTiny, &report);
	CHECK_EQ(back.get({1, 2, 3}), 0);
	CHECK_EQ(back.get({2, 2, 3}), 0);
	CHECK_EQ(back.get({1, 3, 3}), 1);
	CHECK(report.unknownBlocks.empty());
	CHECK_EQ(report.cellsDropped, static_cast<std::size_t>(0));
}

TEST(serialize_bad_cell_states_are_saved_as_open)
{
	const BlockRegistry reg = registryA();
	World w(kTiny);
	w.setCell({1, 2, 3}, static_cast<CellState>(7));
	w.setCell({2, 2, 3}, CellState::Shell);
	const World back = load(save(w, reg), reg, kTiny);
	CHECK(back.cell({1, 2, 3}) == CellState::Open);
	CHECK(back.cell({2, 2, 3}) == CellState::Shell);
}

TEST(serialize_air_is_always_air)
{
	BlockRegistry reg;
	reg.add(named("stone"));
	LoadReport report;
	Bytes b;
	b.header(kTiny, 2).name("stone").name("air").blockRun(6, 1).blockRun(6, 0).cellRun(12, 0);
	const World back = load(b.withCrc(), reg, kTiny, &report);
	CHECK(report.unknownBlocks.empty());
	CHECK_EQ(back.get({1, 2, 3}), 0);
	CHECK_EQ(back.get({2, 4, 4}), 1);
}

TEST(serialize_load_forces_resync_and_is_all_dirty)
{
	const BlockRegistry reg = registryA();
	World w = built();
	World back = load(save(w, reg), reg);
	CHECK(back.changeSeq() > 0);
	std::vector<vox::BlockChange> changes;
	CHECK(!back.changesSince(0, changes)); // a reader that saw nothing yet still has to resync
	CHECK(changes.empty());
	CHECK(back.changesSince(back.changeSeq(), changes)); // after the resync it is up to date
	const DirtyRect all{0, 0, 127, 5119};
	const auto blocks = back.takeDirtyBlocks();
	const auto cells = back.takeDirtyCells();
	REQUIRE(blocks.has_value());
	REQUIRE(cells.has_value());
	CHECK((*blocks == all));
	CHECK((*cells == all));
	// edits after loading journal normally
	const std::uint64_t seq = back.changeSeq();
	CHECK(back.set({0, 80, 0}, 2));
	CHECK(back.changesSince(seq, changes));
	CHECK_EQ(changes.size(), static_cast<std::size_t>(1));
}

TEST(serialize_null_report_is_fine)
{
	const BlockRegistry reg = registryA();
	BlockRegistry empty;
	const World back = load(save(built(), reg), empty, GridSpec{}, nullptr);
	CHECK_EQ(back.get({0, 50, 0}), 0);
}

TEST(serialize_consecutive_saves_in_one_stream)
{
	// loadWorld reads exactly one save and leaves the rest of the stream
	const BlockRegistry reg = registryA();
	const World w1 = patterned(reg);
	World w2(w1.spec());
	w2.fill({-3, 5, 2}, {2, 6, 6}, 4);
	std::istringstream in(save(w1, reg) + save(w2, reg) + "tail", std::ios::binary);
	const World b1 = vox::loadWorld(in, reg, w1.spec());
	const World b2 = vox::loadWorld(in, reg, w1.spec());
	CHECK(b1.blocks() == w1.blocks());
	CHECK(b2.blocks() == w2.blocks());
	std::string rest;
	in >> rest;
	CHECK_EQ(rest, std::string("tail"));
}

// ---- saving errors ----

TEST(serialize_save_to_failed_stream_throws)
{
	const BlockRegistry reg = registryA();
	const World w(kTiny);
	std::ofstream notOpen;
	CHECK(throwsRuntimeError([&] { vox::saveWorld(w, reg, notOpen); }));
	std::ostringstream bad(std::ios::binary);
	bad.setstate(std::ios::badbit);
	CHECK(throwsRuntimeError([&] { vox::saveWorld(w, reg, bad); }));
}

TEST(serialize_save_handles_the_longest_names)
{
	// the registry refuses names over 1024 bytes (the format's limit), so every registered name can be saved
	BlockRegistry reg;
	CHECK_THROWS(reg.add(named(std::string(1025, 'a'))));
	reg.add(named(std::string(1024, 'b')));
	World w(kTiny);
	w.set({1, 2, 3}, 1);
	const std::string bytes = save(w, reg);
	CHECK(!bytes.empty());
	std::istringstream in(bytes, std::ios::binary);
	const World back = vox::loadWorld(in, reg, kTiny);
	CHECK_EQ(back.get({1, 2, 3}), 1);
}

// ---- loading errors ----

TEST(serialize_tiny_body_loads)
{
	// the hand-built bytes the corruption tests below start from are valid
	BlockRegistry reg;
	reg.add(named("stone"));
	reg.add(named("dirt"));
	LoadReport report;
	const World w = load(tinyBody().withCrc(), reg, kTiny, &report);
	CHECK_EQ(w.get({1, 2, 3}), 0);
	CHECK_EQ(w.get({2, 4, 4}), 2);
	CHECK(w.cell({2, 4, 4}) == CellState::Dug);
	CHECK(report.unknownBlocks.empty());
}

TEST(serialize_rejects_bad_magic)
{
	const BlockRegistry reg = registryA();
	CHECK(loadFails("", reg));
	CHECK(loadFails("MCRLVOX", reg));
	std::string bytes = tinyBody().withCrc();
	bytes[7] = '2';
	CHECK(loadFails(bytes, reg, kTiny));
	bytes = tinyBody().withCrc();
	bytes[0] = 'm';
	CHECK(loadFails(bytes, reg, kTiny));
}

TEST(serialize_rejects_grid_mismatch)
{
	const BlockRegistry reg = registryA();
	const std::string bytes = save(built(), reg);
	CHECK(loadFails(bytes, reg, GridSpec{{-64, 47, -64}, {128, 40, 128}}));
	CHECK(loadFails(bytes, reg, GridSpec{{-64, 48, -64}, {128, 41, 128}}));
	CHECK(loadFails(bytes, reg, GridSpec{{-64, 48, -64}, {40, 128, 128}}));
	CHECK(loadFails(tinyBody().withCrc(), reg)); // the default grid
}

TEST(serialize_invalid_spec_is_a_caller_error)
{
	const BlockRegistry reg = registryA();
	const std::string bytes = tinyBody().withCrc();
	std::istringstream in(bytes, std::ios::binary);
	CHECK_THROWS(vox::loadWorld(in, reg, GridSpec{{0, 0, 0}, {0, 1, 1}}));
}

TEST(serialize_rejects_bad_checksum)
{
	const BlockRegistry reg = registryA();
	std::string bytes = tinyBody().withCrc();
	bytes[bytes.size() - 1] = static_cast<char>(bytes[bytes.size() - 1] ^ 0x01);
	CHECK(loadFails(bytes, reg, kTiny));
	CHECK(loadFails(tinyBody().s, reg, kTiny)); // no checksum at all
}

TEST(serialize_rejects_bad_palette_size)
{
	const BlockRegistry reg = registryA();
	const std::uint32_t sizes[] = {32769u, 0x7fffffffu, 0xffffffffu};
	for (const std::uint32_t size : sizes)
	{
		Bytes b;
		b.header(kTiny, size).name("air").blockRun(12, 0).cellRun(12, 0);
		CHECK(loadFails(b.withCrc(), reg, kTiny));
	}
	Bytes empty;
	empty.header(kTiny, 0).blockRun(12, 0).cellRun(12, 0);
	CHECK(loadFails(empty.withCrc(), reg, kTiny));
}

TEST(serialize_largest_palette_loads)
{
	// 32768 names (the registry's capacity) is allowed; all unknown here
	const BlockRegistry reg = registryA();
	Bytes b;
	b.header(kTiny, 32768);
	for (int i = 0; i < 32768; ++i)
		b.name("n" + std::to_string(i));
	b.blockRun(11, 32767).blockRun(1, 0).cellRun(12, 1);
	LoadReport report;
	const World w = load(b.withCrc(), reg, kTiny, &report);
	CHECK_EQ(report.unknownBlocks.size(), static_cast<std::size_t>(32768));
	CHECK_EQ(report.unknownBlocks.back(), std::string("n32767"));
	CHECK_EQ(report.cellsDropped, static_cast<std::size_t>(12));
	CHECK(w.cell({1, 2, 3}) == CellState::Shell);
}

TEST(serialize_rejects_bad_name_lengths)
{
	const BlockRegistry reg = registryA();
	{
		Bytes b;
		b.header(kTiny, 1).u16(0).blockRun(12, 0).cellRun(12, 0);
		CHECK(loadFails(b.withCrc(), reg, kTiny));
	}
	{
		Bytes b;
		b.header(kTiny, 1).name(std::string(1025, 'x')).blockRun(12, 0).cellRun(12, 0);
		CHECK(loadFails(b.withCrc(), reg, kTiny));
	}
	{
		Bytes b; // claims 65535 bytes, has 3
		b.header(kTiny, 1).u16(0xffff);
		b.s += "air";
		CHECK(loadFails(b.withCrc(), reg, kTiny));
	}
	{
		Bytes b; // 1024 is the longest allowed
		b.header(kTiny, 1).name(std::string(1024, 'x')).blockRun(12, 0).cellRun(12, 0);
		LoadReport report;
		load(b.withCrc(), reg, kTiny, &report);
		CHECK_EQ(report.unknownBlocks.size(), static_cast<std::size_t>(1));
	}
}

TEST(serialize_rejects_duplicate_names)
{
	const BlockRegistry reg = registryA();
	Bytes b;
	b.header(kTiny, 3).name("stone").name("dirt").name("stone").blockRun(12, 0).cellRun(12, 0);
	CHECK(loadFails(b.withCrc(), reg, kTiny));
}

TEST(serialize_rejects_bad_block_runs)
{
	const BlockRegistry reg = registryA();
	const auto file = [](std::uint32_t c0, std::uint32_t i0, std::uint32_t c1, std::uint32_t i1) {
		Bytes b;
		b.header(kTiny, 2).name("air").name("stone").blockRun(c0, i0);
		if (c1 != 0 || i1 != 0)
			b.blockRun(c1, i1);
		b.cellRun(12, 0);
		return b.withCrc();
	};
	CHECK(!loadFails(file(6, 0, 6, 1), reg, kTiny)); // the baseline is valid
	CHECK(loadFails(file(0, 0, 12, 1), reg, kTiny)); // zero-length run
	CHECK(loadFails(file(13, 0, 0, 0), reg, kTiny)); // past the grid
	CHECK(loadFails(file(6, 0, 7, 1), reg, kTiny));  // the second run overflows
	CHECK(loadFails(file(0xffffffffu, 0, 0, 0), reg, kTiny));
	CHECK(loadFails(file(6, 0, 6, 2), reg, kTiny));      // index == palette size
	CHECK(loadFails(file(6, 0xffff, 6, 1), reg, kTiny)); // index far out of range
	CHECK(loadFails(file(6, 0, 5, 1), reg, kTiny));      // too few blocks: the cell run is read as a block run
}

TEST(serialize_rejects_bad_cell_runs)
{
	const BlockRegistry reg = registryA();
	const auto file = [](std::uint32_t c0, std::uint32_t s0, std::uint32_t c1, std::uint32_t s1) {
		Bytes b;
		b.header(kTiny, 1).name("air").blockRun(12, 0).cellRun(c0, s0);
		if (c1 != 0 || s1 != 0)
			b.cellRun(c1, s1);
		return b.withCrc();
	};
	CHECK(!loadFails(file(4, 1, 8, 2), reg, kTiny)); // the baseline is valid
	CHECK(loadFails(file(0, 0, 12, 1), reg, kTiny)); // zero-length run
	CHECK(loadFails(file(13, 0, 0, 0), reg, kTiny)); // past the grid
	CHECK(loadFails(file(4, 1, 9, 2), reg, kTiny));
	CHECK(loadFails(file(0xffffffffu, 1, 0, 0), reg, kTiny));
	CHECK(loadFails(file(4, 3, 8, 2), reg, kTiny)); // state 3
	CHECK(loadFails(file(4, 1, 8, 255), reg, kTiny));
}

TEST(serialize_failed_load_leaves_report_alone)
{
	const BlockRegistry reg = registryA();
	LoadReport report;
	report.unknownBlocks.push_back("before");
	report.cellsDropped = 3;
	std::string bytes = save(built(), reg);
	bytes.resize(bytes.size() - 1);
	CHECK(throwsRuntimeError([&] { load(bytes, reg, GridSpec{}, &report); }));
	CHECK((report.unknownBlocks == std::vector<std::string>{"before"}));
	CHECK_EQ(report.cellsDropped, static_cast<std::size_t>(3));
}

// ---- hostile input ----

TEST(serialize_truncation_at_every_offset_throws)
{
	const BlockRegistry reg = registryA();
	const World small = patterned(reg);
	const std::string a = save(small, reg);
	int notThrown = 0;
	for (std::size_t n = 0; n < a.size(); ++n)
		if (!loadFails(a.substr(0, n), reg, small.spec()))
			++notThrown;
	// the default grid with a few blocks and cells (every truncated load allocates the whole grid)
	World big;
	big.set({0, 60, 0}, 1);
	big.fill({-3, 70, -3}, {3, 70, 3}, 4);
	big.set({63, 87, 63}, 3);
	big.setCell({0, 63, 0}, CellState::Shell);
	big.setCell({1, 63, 0}, CellState::Dug);
	const std::string b = save(big, reg);
	CHECK(b.size() < 400);
	for (std::size_t n = 0; n < b.size(); ++n)
		if (!loadFails(b.substr(0, n), reg))
			++notThrown;
	CHECK_EQ(notThrown, 0);
	CHECK(load(b, reg).blocks() == big.blocks());
}

TEST(serialize_every_byte_and_bit_flip_throws)
{
	// a single damaged byte always breaks the structure or the CRC
	const BlockRegistry reg = registryA();
	const World small = patterned(reg);
	const std::string bytes = save(small, reg);
	int notThrown = 0;
	for (std::size_t i = 0; i < bytes.size(); ++i)
		for (int mask = 1; mask <= 0x100; mask <<= 1)
		{
			std::string damaged = bytes;
			damaged[i] = static_cast<char>(damaged[i] ^ (mask == 0x100 ? 0xff : mask));
			if (!loadFails(damaged, reg, small.spec()))
				++notThrown;
		}
	CHECK_EQ(notThrown, 0);
}

TEST(serialize_fuzz_with_valid_checksums)
{
	// damage past the CRC: every mutation must either fail cleanly or load a valid world
	const BlockRegistry reg = registryA();
	const World small = patterned(reg);
	const std::string bytes = save(small, reg);
	const std::string body = bytes.substr(0, bytes.size() - 4);
	std::uint32_t seed = 2024;
	int failed = 0, loaded = 0, invalid = 0;
	for (int iteration = 0; iteration < 4000; ++iteration)
	{
		Bytes b;
		b.s = body;
		const int edits = 1 + static_cast<int>(lcg(seed) % 4);
		for (int e = 0; e < edits; ++e)
		{
			const std::size_t at = lcg(seed) % b.s.size();
			switch (lcg(seed) % 4)
			{
			case 0:
				b.s[at] = static_cast<char>(lcg(seed) & 0xff);
				break;
			case 1:
				b.s[at] = static_cast<char>(b.s[at] ^ (1 << (lcg(seed) % 8)));
				break;
			case 2:
				b.s.resize(at);
				break;
			default:
				b.s.insert(at, 1, static_cast<char>(lcg(seed) & 0xff));
				break;
			}
			if (b.s.empty())
				b.s = body;
		}
		try
		{
			LoadReport report;
			const World w = load(b.withCrc(), reg, small.spec(), &report);
			++loaded;
			for (std::size_t i = 0; i < w.blocks().size(); ++i)
				if (!reg.contains(w.blocks()[i]) || static_cast<int>(w.cells()[i]) > 2)
					++invalid;
		}
		catch (const std::runtime_error&)
		{
			++failed;
		}
	}
	CHECK_EQ(invalid, 0);
	CHECK(failed > 0);
	CHECK(loaded > 0); // some damage (a palette index or state changed in range) is indistinguishable from data
}
