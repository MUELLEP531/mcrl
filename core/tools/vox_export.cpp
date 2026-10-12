// Builds a demo world (Rocket League's arena plus a few structures and a dug hole) and writes what the ReShade effect
// would be given, for tools/preview/render.py:
//
//     vox_export --blocks assets/mcrl_blocks.txt --out <dir> [--atlas assets/mcrl_atlas.png] [--skin assets/mcrl_steve.png]
//
// <dir>/vox.r16, cells.r8, types.rgba32f (docs/SHADER_CONTRACT.md), scene.json (camera, Steve, arena) and
// world.mcrlvox (the save format; reloaded and compared as a check).
#include "rl/arena.hpp"
#include "vox/block_registry.hpp"
#include "vox/edit.hpp"
#include "vox/gpu_layout.hpp"
#include "vox/serialize.hpp"
#include "vox/world.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options
{
	std::string blocks = "assets/mcrl_blocks.txt";
	std::string out = "tools/preview/out/demo";
	std::string atlas = "assets/mcrl_atlas.png";
	std::string skin = "assets/mcrl_steve.png";
};

Options parseArgs(int argc, char** argv)
{
	Options o;
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		const auto value = [&]() -> std::string {
			if (i + 1 >= argc)
				throw std::runtime_error("missing value after " + arg);
			return argv[++i];
		};
		if (arg == "--blocks")
			o.blocks = value();
		else if (arg == "--out")
			o.out = value();
		else if (arg == "--atlas")
			o.atlas = value();
		else if (arg == "--skin")
			o.skin = value();
		else
			throw std::runtime_error("unknown argument " + arg);
	}
	return o;
}

void writeFile(const std::string& path, const std::vector<unsigned char>& bytes)
{
	std::ofstream out(path, std::ios::binary);
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	if (!out)
		throw std::runtime_error("cannot write " + path);
}

std::vector<unsigned char> littleEndian(const std::vector<std::uint16_t>& values)
{
	std::vector<unsigned char> bytes;
	bytes.reserve(values.size() * 2);
	for (const std::uint16_t v : values)
	{
		bytes.push_back(static_cast<unsigned char>(v & 0xff));
		bytes.push_back(static_cast<unsigned char>(v >> 8));
	}
	return bytes;
}

std::vector<unsigned char> littleEndian(const std::vector<float>& values)
{
	std::vector<unsigned char> bytes;
	bytes.reserve(values.size() * 4);
	for (const float f : values)
	{
		std::uint32_t u = 0;
		static_assert(sizeof(u) == sizeof(f), "float must be 32-bit");
		std::memcpy(&u, &f, sizeof(u));
		for (int b = 0; b < 4; ++b)
			bytes.push_back(static_cast<unsigned char>((u >> (8 * b)) & 0xff));
	}
	return bytes;
}

std::string jsonString(const std::string& s)
{
	std::string out = "\"";
	for (const char c : s)
	{
		if (c == '"' || c == '\\')
			out += '\\';
		out += c;
	}
	return out + "\"";
}

/// The first of `names` the registry has (block lists may name things differently); air if none.
vox::BlockId pick(const vox::BlockRegistry& registry, std::initializer_list<const char*> names)
{
	for (const char* name : names)
		if (const auto id = registry.find(name))
			return *id;
	std::fprintf(stderr, "warning: none of the blocks %s... exist; leaving them out\n", *names.begin());
	return vox::kAir;
}

void column(vox::World& world, vox::Int3 base, int height, vox::BlockId id)
{
	for (int i = 0; i < height; ++i)
		world.set(base + vox::Int3{0, i, 0}, id);
}

} // namespace

int main(int argc, char** argv)
{
	try
	{
		const Options o = parseArgs(argc, argv);
		std::filesystem::create_directories(o.out);
		const vox::BlockRegistry registry = vox::BlockRegistry::loadFile(o.blocks);

		const vox::BlockId barrier = pick(registry, {"barrier"});
		const vox::BlockId dirt = pick(registry, {"dirt"});
		const vox::BlockId stone = pick(registry, {"stone"});

		vox::World world;
		rl::ArenaOptions arena;
		arena.shellBlock = barrier;
		arena.topsoilBlock = dirt;
		arena.groundBlock = stone;
		rl::buildSoccarArena(world, registry, arena);
		const int floorY = static_cast<int>(arena.mapping.floorY);

		// a little build on the pitch, in front of the camera
		const vox::BlockId planks = pick(registry, {"oak_planks", "planks"});
		const vox::BlockId log = pick(registry, {"oak_log", "log"});
		const vox::BlockId leaves = pick(registry, {"oak_leaves", "leaves"});
		const vox::BlockId glass = pick(registry, {"glass"});
		const vox::BlockId grass = pick(registry, {"grass_block", "grass"});
		const vox::BlockId cobble = pick(registry, {"cobblestone"});
		const vox::BlockId tnt = pick(registry, {"tnt"});
		const vox::BlockId bricks = pick(registry, {"bricks", "brick"});
		const vox::BlockId wool = pick(registry, {"red_wool", "wool_red"});
		const vox::BlockId gold = pick(registry, {"gold_block"});

		// a hut: cobblestone corners, planks walls, a glass window, brick roof
		for (int x = -6; x <= -2; ++x)
			for (int z = 4; z <= 8; ++z)
				for (int y = floorY; y < floorY + 4; ++y)
				{
					const bool wall = x == -6 || x == -2 || z == 4 || z == 8;
					if (!wall)
						continue;
					const bool corner = (x == -6 || x == -2) && (z == 4 || z == 8);
					const bool door = z == 4 && x == -4 && y < floorY + 2;
					const bool window = z == 4 && x == -3 && y == floorY + 2;
					if (door)
						continue;
					world.set({x, y, z}, corner ? cobble : (window ? glass : planks));
				}
		world.fill({-6, floorY + 4, 4}, {-2, floorY + 4, 8}, bricks);

		// a tree on a patch of grass
		world.fill({3, floorY, 9}, {5, floorY, 11}, grass);
		column(world, {4, floorY + 1, 10}, 4, log);
		for (int x = 2; x <= 6; ++x)
			for (int z = 8; z <= 12; ++z)
				for (int y = floorY + 4; y <= floorY + 5; ++y)
					if (world.get({x, y, z}) == vox::kAir && !((x == 2 || x == 6) && (z == 8 || z == 12)))
						world.set({x, y, z}, leaves);
		world.fill({3, floorY + 6, 9}, {5, floorY + 6, 11}, leaves);

		// odds and ends: TNT, a wool line, a gold block on a cobble pillar
		world.set({1, floorY, 3}, tnt);
		world.fill({-1, floorY, 12}, {3, floorY, 12}, wool);
		column(world, {7, floorY, 5}, 2, cobble);
		world.set({7, floorY + 2, 5}, gold);

		// dig a 3 x 3 hole two blocks deep into RL's floor (the shell is at floorY - 1), as a player would
		for (int x = -1; x <= 1; ++x)
			for (int z = -6; z <= -4; ++z)
				for (int y = floorY - 1; y >= floorY - 2; --y)
				{
					const vox::EditResult r = vox::breakBlock(world, registry, {x, y, z});
					if (r != vox::EditResult::Ok)
						std::fprintf(stderr, "warning: digging (%d,%d,%d) failed (%d)\n", x, y, z, static_cast<int>(r));
				}

		std::vector<std::uint16_t> vox;
		std::vector<std::uint8_t> cells;
		std::vector<float> types;
		vox::exportVoxels(world, registry, vox);
		vox::exportCells(world, cells);
		vox::exportTypes(registry, types);
		writeFile(o.out + "/vox.r16", littleEndian(vox));
		writeFile(o.out + "/cells.r8", std::vector<unsigned char>(cells.begin(), cells.end()));
		writeFile(o.out + "/types.rgba32f", littleEndian(types));

		// the save format, reloaded and compared
		{
			std::ofstream save(o.out + "/world.mcrlvox", std::ios::binary);
			vox::saveWorld(world, registry, save);
		}
		{
			std::ifstream load(o.out + "/world.mcrlvox", std::ios::binary);
			const vox::World back = vox::loadWorld(load, registry);
			if (back.blocks() != world.blocks() || back.cells() != world.cells())
				throw std::runtime_error("world.mcrlvox did not reload identically");
		}

		const rl::SoccarDimensions& d = arena.dims;
		std::ofstream scene(o.out + "/scene.json", std::ios::binary);
		scene << "{\n"
			  << "  \"width\": 640, \"height\": 360, \"fovY\": 70,\n"
			  << "  \"camera\": {\"pos\": [0.5, " << floorY + 2.6 << ", -14.0], \"yaw\": 0, \"pitch\": 12},\n"
			  << "  \"steve\": {\"pos\": [2.5, " << floorY << ", -8.0, 1], \"look\": [200, 190, 5, 0], \"anim\": [3.0, 0.6, 0.0, 0]"
			  << ", \"item\": [-1, " << planks << ", 0, 0]},\n"
			  << "  \"arena\": {\"uuPerBlock\": " << arena.mapping.uuPerBlock << ", \"floorY\": " << arena.mapping.floorY
			  << ", \"halfWidth\": " << d.halfWidth << ", \"halfLength\": " << d.halfLength << ", \"height\": " << d.height
			  << ", \"cornerSum\": " << d.cornerSum << ", \"goalHalfWidth\": " << d.goalHalfWidth
			  << ", \"goalHeight\": " << d.goalHeight << ", \"goalDepth\": " << d.goalDepth << "},\n"
			  << "  \"atlas\": " << jsonString(o.atlas) << ",\n"
			  << "  \"skin\": " << jsonString(o.skin) << "\n"
			  << "}\n";
		if (!scene)
			throw std::runtime_error("cannot write scene.json");

		std::size_t shell = 0, dug = 0, solid = 0;
		for (std::size_t i = 0; i < world.blocks().size(); ++i)
		{
			shell += world.cells()[i] == vox::CellState::Shell;
			dug += world.cells()[i] == vox::CellState::Dug;
			solid += world.blocks()[i] != vox::kAir;
		}
		std::printf("wrote %s: %zu blocks registered, %zu non-air cells, %zu shell, %zu dug\n", o.out.c_str(),
			registry.size(), solid, shell, dug);
		return 0;
	}
	catch (const std::exception& e)
	{
		std::fprintf(stderr, "vox_export: %s\n", e.what());
		return 1;
	}
}
