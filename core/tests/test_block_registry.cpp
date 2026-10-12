#include "check.hpp"

#include "vox/block_registry.hpp"

#include <sstream>

using vox::BlockRegistry;
using vox::BlockType;

namespace {

BlockType named(const char* name)
{
	BlockType t;
	t.name = name;
	return t;
}

} // namespace

TEST(registry_starts_with_air)
{
	BlockRegistry r;
	CHECK_EQ(r.size(), static_cast<std::size_t>(1));
	CHECK_EQ(r.get(vox::kAir).name, std::string("air"));
	CHECK(!r.get(vox::kAir).solid);
	CHECK(!r.get(vox::kAir).visible);
	CHECK(!r.get(vox::kAir).breakable);
	CHECK(r.find("air") == std::optional<vox::BlockId>(vox::kAir));
}

TEST(registry_add_and_find)
{
	BlockRegistry r;
	const vox::BlockId stone = r.add(named("stone"));
	const vox::BlockId dirt = r.add(named("dirt"));
	CHECK_EQ(stone, 1);
	CHECK_EQ(dirt, 2);
	CHECK(r.find("dirt") == std::optional<vox::BlockId>(dirt));
	CHECK(!r.find("grass"));
	CHECK_THROWS(r.get(3));
}

TEST(registry_rejects_bad_types)
{
	BlockRegistry r;
	r.add(named("stone"));
	CHECK_THROWS(r.add(named("stone")));
	CHECK_THROWS(r.add(named("")));
	CHECK_THROWS(r.add(named("two words")));
	CHECK_THROWS(r.add(named("air")));
	BlockType t = named("bad_slot");
	t.texSide = 4096;
	CHECK_THROWS(r.add(t));
	t.texSide = -2;
	CHECK_THROWS(r.add(t));
}

TEST(registry_text_round_trip)
{
	BlockRegistry r;
	BlockType grass = named("grass_block");
	grass.texTop = 7;
	grass.texSide = 8;
	grass.texBottom = 9;
	grass.tintFaces = vox::kTintTop;
	grass.tint = vox::Rgb{0x91, 0xbd, 0x59};
	grass.color = vox::Rgb{0x5d, 0x8c, 0x3a};
	r.add(grass);
	BlockType barrier = named("barrier");
	barrier.visible = false;
	r.add(barrier);

	std::stringstream text;
	r.write(text);
	const BlockRegistry back = BlockRegistry::parse(text);
	REQUIRE(back.size() == r.size());
	const BlockType& g = back.get(1);
	CHECK_EQ(g.name, std::string("grass_block"));
	CHECK_EQ(g.texTop, 7);
	CHECK_EQ(g.texSide, 8);
	CHECK_EQ(g.texBottom, 9);
	CHECK_EQ(static_cast<int>(g.tintFaces), 1);
	CHECK(g.tint == (vox::Rgb{0x91, 0xbd, 0x59}));
	CHECK(g.color == (vox::Rgb{0x5d, 0x8c, 0x3a}));
	CHECK(!back.get(2).visible);
	CHECK(back.get(2).solid);
}

TEST(registry_parse_accepts_comments_and_crlf)
{
	std::istringstream text("# header\r\n\r\n   # indented comment\n1 stone 0 0 0 1 1 1 0 #ffffff #7F7F7F\r\n");
	const BlockRegistry r = BlockRegistry::parse(text);
	REQUIRE(r.size() == 2);
	CHECK(r.get(1).color == (vox::Rgb{0x7f, 0x7f, 0x7f}));
}

TEST(registry_parse_rejects_malformed_lines)
{
	const char* bad[] = {
		"2 stone 0 0 0 1 1 1 0 #ffffff #ffffff\n",          // ids must start at 1
		"1 stone 0 0 1 1 1 0 #ffffff #ffffff\n",            // 10 fields
		"1 stone 0 0 4096 1 1 1 0 #ffffff #ffffff\n",       // slot out of range
		"1 stone 0 0 x 1 1 1 0 #ffffff #ffffff\n",          // not a number
		"1 stone 0 0 0 2 1 1 0 #ffffff #ffffff\n",          // flag not 0/1
		"1 stone 0 0 0 1 1 1 8 #ffffff #ffffff\n",          // tint mask > 7
		"1 stone 0 0 0 1 1 1 0 ffffff #ffffff\n",           // colour without #
		"1 stone 0 0 0 1 1 1 0 #fffffg #ffffff\n",          // bad hex digit
		"1 stone 0 0 0 1 1 1 0 #ffffff #ffffff\n1 dirt 0 0 0 1 1 1 0 #ffffff #ffffff\n", // repeated id
		"1 stone 0 0 0 1 1 1 0 #ffffff #ffffff\n2 stone 0 0 0 1 1 1 0 #ffffff #ffffff\n", // repeated name
	};
	for (const char* text : bad)
	{
		std::istringstream in(text);
		CHECK_THROWS(BlockRegistry::parse(in));
	}
}

TEST(registry_load_missing_file_throws) { CHECK_THROWS(BlockRegistry::loadFile("/nonexistent/mcrl_blocks.txt")); }
