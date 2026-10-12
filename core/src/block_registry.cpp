#include "vox/block_registry.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vox {

namespace {

bool validName(const std::string& name)
{
	if (name.empty())
		return false;
	for (const char c : name)
		if (std::isspace(static_cast<unsigned char>(c)) || c == '#')
			return false;
	return true;
}

bool validSlot(int slot) { return slot >= -1 && slot < BlockRegistry::kAtlasSlots; }

std::string hexColor(const Rgb& c)
{
	char buffer[8];
	std::snprintf(buffer, sizeof(buffer), "#%02x%02x%02x", c.r, c.g, c.b);
	return buffer;
}

bool parseHexColor(const std::string& text, Rgb& out)
{
	if (text.size() != 7 || text[0] != '#')
		return false;
	unsigned value = 0;
	for (std::size_t i = 1; i < 7; ++i)
	{
		const char c = text[i];
		unsigned digit = 0;
		if (c >= '0' && c <= '9')
			digit = static_cast<unsigned>(c - '0');
		else if (c >= 'a' && c <= 'f')
			digit = static_cast<unsigned>(c - 'a' + 10);
		else if (c >= 'A' && c <= 'F')
			digit = static_cast<unsigned>(c - 'A' + 10);
		else
			return false;
		value = value * 16 + digit;
	}
	out.r = static_cast<std::uint8_t>((value >> 16) & 0xff);
	out.g = static_cast<std::uint8_t>((value >> 8) & 0xff);
	out.b = static_cast<std::uint8_t>(value & 0xff);
	return true;
}

bool parseFlag(const std::string& text, bool& out)
{
	if (text == "0")
		out = false;
	else if (text == "1")
		out = true;
	else
		return false;
	return true;
}

bool parseInt(const std::string& text, long min, long max, long& out)
{
	if (text.empty())
		return false;
	std::size_t used = 0;
	long value = 0;
	try
	{
		value = std::stol(text, &used, 10);
	}
	catch (const std::exception&)
	{
		return false;
	}
	if (used != text.size() || value < min || value > max)
		return false;
	out = value;
	return true;
}

} // namespace

BlockRegistry::BlockRegistry()
{
	BlockType air;
	air.name = "air";
	air.solid = false;
	air.visible = false;
	air.breakable = false;
	air.color = Rgb{0, 0, 0};
	byName_.emplace(air.name, kAir);
	types_.push_back(air);
}

BlockId BlockRegistry::add(BlockType type)
{
	if (!validName(type.name))
		throw std::invalid_argument("block name must be non-empty with no whitespace or '#': '" + type.name + "'");
	if (type.name.size() > kMaxNameLength)
		throw std::invalid_argument("block name longer than 1024 bytes");
	if (find(type.name))
		throw std::invalid_argument("block name already registered: " + type.name);
	if (!validSlot(type.texTop) || !validSlot(type.texSide) || !validSlot(type.texBottom))
		throw std::invalid_argument("atlas slot out of range (-1..4095) for block " + type.name);
	if (types_.size() >= kMaxBlocks)
		throw std::invalid_argument("block registry is full");
	const BlockId id = static_cast<BlockId>(types_.size());
	byName_.emplace(type.name, id);
	types_.push_back(std::move(type));
	return id;
}

const BlockType& BlockRegistry::get(BlockId id) const
{
	if (id >= types_.size())
		throw std::out_of_range("unknown block id " + std::to_string(id));
	return types_[id];
}

std::optional<BlockId> BlockRegistry::find(std::string_view name) const
{
	const auto it = byName_.find(std::string(name));
	if (it == byName_.end())
		return std::nullopt;
	return it->second;
}

BlockRegistry BlockRegistry::parse(std::istream& in)
{
	BlockRegistry registry;
	std::string line;
	int lineNumber = 0;
	while (std::getline(in, line))
	{
		++lineNumber;
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		const std::size_t first = line.find_first_not_of(" \t");
		if (first == std::string::npos || line[first] == '#')
			continue;

		const auto fail = [&](const std::string& why) {
			throw std::runtime_error("blocks file line " + std::to_string(lineNumber) + ": " + why);
		};

		std::istringstream fields(line);
		std::vector<std::string> f;
		for (std::string field; fields >> field;)
			f.push_back(field);
		if (f.size() != 11)
			fail("expected 11 fields, got " + std::to_string(f.size()));

		long id = 0, top = 0, side = 0, bottom = 0, tintFaces = 0;
		if (!parseInt(f[0], 1, static_cast<long>(kMaxBlocks) - 1, id))
			fail("bad id '" + f[0] + "'");
		if (static_cast<std::size_t>(id) != registry.size())
			fail("ids must be consecutive from 1: expected " + std::to_string(registry.size()) + ", got " + f[0]);
		if (!parseInt(f[2], -1, kAtlasSlots - 1, top) || !parseInt(f[3], -1, kAtlasSlots - 1, side)
			|| !parseInt(f[4], -1, kAtlasSlots - 1, bottom))
			fail("atlas slots must be integers in -1..4095");

		BlockType type;
		type.name = f[1];
		type.texTop = static_cast<int>(top);
		type.texSide = static_cast<int>(side);
		type.texBottom = static_cast<int>(bottom);
		if (!parseFlag(f[5], type.solid) || !parseFlag(f[6], type.visible) || !parseFlag(f[7], type.breakable))
			fail("solid / visible / breakable must be 0 or 1");
		if (!parseInt(f[8], 0, 7, tintFaces))
			fail("tint_faces must be 0..7");
		type.tintFaces = static_cast<std::uint8_t>(tintFaces);
		if (!parseHexColor(f[9], type.tint) || !parseHexColor(f[10], type.color))
			fail("colours must be #rrggbb");
		try
		{
			registry.add(std::move(type));
		}
		catch (const std::invalid_argument& e)
		{
			fail(e.what());
		}
	}
	return registry;
}

BlockRegistry BlockRegistry::loadFile(const std::string& path)
{
	std::ifstream in(path, std::ios::binary);
	if (!in)
		throw std::runtime_error("cannot open blocks file " + path);
	return parse(in);
}

void BlockRegistry::write(std::ostream& out) const
{
	out << "# mcrl blocks v1\n";
	out << "# id name top side bottom solid visible breakable tint_faces tint color\n";
	for (std::size_t i = 1; i < types_.size(); ++i)
	{
		const BlockType& t = types_[i];
		out << i << ' ' << t.name << ' ' << t.texTop << ' ' << t.texSide << ' ' << t.texBottom << ' ' << (t.solid ? 1 : 0)
			<< ' ' << (t.visible ? 1 : 0) << ' ' << (t.breakable ? 1 : 0) << ' ' << static_cast<int>(t.tintFaces) << ' '
			<< hexColor(t.tint) << ' ' << hexColor(t.color) << '\n';
	}
}

} // namespace vox
