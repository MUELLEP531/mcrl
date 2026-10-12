// The "MCRLVOX1" save format. Every integer is little-endian, written and read byte by byte; int32 is two's
// complement. Field by field:
//
//     offset  size  field
//     0       8     magic: the ASCII bytes "MCRLVOX1" (no terminator)
//     8       4     grid origin x  (int32)
//     12      4     grid origin y  (int32)
//     16      4     grid origin z  (int32)
//     20      4     grid size x    (int32)
//     24      4     grid size y    (int32)
//     28      4     grid size z    (int32)
//     32      4     palette count P (uint32, 1..32768 = kMaxPalette)
//     36      ...   P palette entries, each:
//                       2  name length L (uint16, 1..1024 = kMaxNameBytes)
//                       L  the block type's name (raw bytes, no terminator); no name appears twice
//     ...     ...   block runs, until their counts add up to the cell count N = size.x * size.y * size.z:
//                       4  count (uint32, 1..cells still to cover)
//                       2  palette index (uint16, < P)
//     ...     ...   cell-state runs, the same way:
//                       4  count (uint32, 1..cells still to cover)
//                       1  state (uint8: 0 open, 1 shell, 2 dug)
//     ...     4     CRC-32 (IEEE 802.3 / zlib, uint32) of every byte before it, from the magic on
//
// Runs go through the cells in storage order (World::indexOf: x fastest, then z, then y), which is the GPU layout.
// A run never spans more than 2^32 - 1 cells (longer ones are split).
//
// Saving: the palette holds only the names of the types the world uses, in id order. Ids the registry doesn't have
// are saved as air, and cell states other than open / shell / dug as open. A used name longer than kMaxNameBytes
// cannot be saved (std::runtime_error, nothing written).
//
// Loading: names map to ids through the registry by name ("air" always to air); names it doesn't have load as air and
// are reported. The grid must equal the GridSpec passed in. Anything structurally wrong (bad magic, grid mismatch,
// truncation, a count or length outside its range, a zero-length run or one past the grid, a palette index or state
// out of range, a duplicate name, a bad checksum) throws std::runtime_error; an invalid GridSpec is the caller's error
// (std::invalid_argument, from World). Memory is bounded by the GridSpec and the caps above, never by a size read from
// the file: the grid's arrays are sized from the GridSpec, and a name is allocated (at most kMaxNameBytes) only once
// its length has been read. loadWorld reads exactly one save: bytes after the checksum stay in the stream. The
// LoadReport is only written when loading succeeds.
#include "vox/serialize.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vox {

namespace {

constexpr char kMagic[8] = {'M', 'C', 'R', 'L', 'V', 'O', 'X', '1'};
/// A save never names more types than a registry can hold.
constexpr std::uint32_t kMaxPalette = static_cast<std::uint32_t>(BlockRegistry::kMaxBlocks);
/// Far longer than any real block name, short enough that a hostile palette stays small.
constexpr std::size_t kMaxNameBytes = 1024;
constexpr std::uint32_t kMaxRun = 0xffffffffu;
constexpr std::uint8_t kStateCount = 3;

[[noreturn]] void corrupt(const std::string& why) { throw std::runtime_error("MCRLVOX1: " + why); }

const std::array<std::uint32_t, 256>& crcTable()
{
	static const std::array<std::uint32_t, 256> table = [] {
		std::array<std::uint32_t, 256> t{};
		for (std::uint32_t i = 0; i < 256; ++i)
		{
			std::uint32_t c = i;
			for (int k = 0; k < 8; ++k)
				c = (c & 1u) ? 0xedb88320u ^ (c >> 1) : c >> 1;
			t[i] = c;
		}
		return t;
	}();
	return table;
}

/// Running CRC-32 without the final inversion: start from 0xffffffff, finish with ~.
std::uint32_t crcUpdate(std::uint32_t crc, const unsigned char* data, std::size_t size)
{
	const std::array<std::uint32_t, 256>& table = crcTable();
	for (std::size_t i = 0; i < size; ++i)
		crc = table[(crc ^ data[i]) & 0xffu] ^ (crc >> 8);
	return crc;
}

class Writer
{
public:
	void u8(std::uint8_t v) { bytes_.push_back(v); }
	void u16(std::uint16_t v)
	{
		bytes_.push_back(static_cast<unsigned char>(v & 0xffu));
		bytes_.push_back(static_cast<unsigned char>((v >> 8) & 0xffu));
	}
	void u32(std::uint32_t v)
	{
		for (int b = 0; b < 4; ++b)
			bytes_.push_back(static_cast<unsigned char>((v >> (8 * b)) & 0xffu));
	}
	void i32(int v) { u32(static_cast<std::uint32_t>(v)); } // modulo 2^32: two's complement
	void raw(const char* data, std::size_t size)
	{
		for (std::size_t i = 0; i < size; ++i)
			bytes_.push_back(static_cast<unsigned char>(data[i]));
	}
	void crc() { u32(~crcUpdate(0xffffffffu, bytes_.data(), bytes_.size())); }
	const std::vector<unsigned char>& bytes() const { return bytes_; }

private:
	std::vector<unsigned char> bytes_;
};

class Reader
{
public:
	explicit Reader(std::istream& in)
		: in_(in)
	{
	}

	void raw(unsigned char* data, std::size_t size)
	{
		in_.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
		if (static_cast<std::size_t>(in_.gcount()) != size)
			corrupt("unexpected end of data");
		crc_ = crcUpdate(crc_, data, size);
	}
	std::uint8_t u8()
	{
		unsigned char b[1];
		raw(b, 1);
		return b[0];
	}
	std::uint16_t u16()
	{
		unsigned char b[2];
		raw(b, 2);
		return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
	}
	std::uint32_t u32()
	{
		unsigned char b[4];
		raw(b, 4);
		return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8)
			| (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
	}
	int i32()
	{
		// no implementation-defined unsigned -> signed conversion
		const std::uint32_t u = u32();
		return u <= 0x7fffffffu ? static_cast<int>(u) : -static_cast<int>(~u) - 1;
	}
	/// The CRC of everything read so far.
	std::uint32_t crc() const { return ~crc_; }

private:
	std::istream& in_;
	std::uint32_t crc_ = 0xffffffffu;
};

/// Run-length encodes `count` values given by `value(i)`, calling `emit(runLength, value)` per run.
template <typename Value, typename Emit>
void encodeRuns(std::size_t count, Value value, Emit emit)
{
	std::size_t i = 0;
	while (i < count)
	{
		const auto v = value(i);
		std::size_t end = i + 1;
		while (end < count && end - i < kMaxRun && value(end) == v)
			++end;
		emit(static_cast<std::uint32_t>(end - i), v);
		i = end;
	}
}

/// A run's count, checked against the cells still to cover.
std::size_t readRun(Reader& r, std::size_t remaining)
{
	const std::uint32_t count = r.u32();
	if (count == 0)
		corrupt("zero-length run");
	if (count > remaining)
		corrupt("run reaches past the end of the grid");
	return count;
}

std::uint8_t stateByte(CellState state)
{
	const std::uint8_t v = static_cast<std::uint8_t>(state);
	return v < kStateCount ? v : static_cast<std::uint8_t>(CellState::Open);
}

} // namespace

void saveWorld(const World& world, const BlockRegistry& registry, std::ostream& out)
{
	const std::vector<BlockId>& blocks = world.blocks();
	const std::vector<CellState>& cells = world.cells();
	const auto known = [&registry](BlockId id) { return registry.contains(id) ? id : kAir; };

	// the palette: the ids in use (unknown ones as air), in id order
	std::vector<bool> used(registry.size(), false);
	for (const BlockId id : blocks)
		used[known(id)] = true;
	std::vector<std::uint16_t> paletteIndex(registry.size(), 0);
	std::vector<BlockId> palette;
	for (std::size_t id = 0; id < used.size(); ++id)
		if (used[id])
		{
			paletteIndex[id] = static_cast<std::uint16_t>(palette.size());
			palette.push_back(static_cast<BlockId>(id));
		}

	Writer w;
	w.raw(kMagic, sizeof(kMagic));
	const GridSpec& spec = world.spec();
	w.i32(spec.origin.x);
	w.i32(spec.origin.y);
	w.i32(spec.origin.z);
	w.i32(spec.size.x);
	w.i32(spec.size.y);
	w.i32(spec.size.z);
	w.u32(static_cast<std::uint32_t>(palette.size()));
	for (const BlockId id : palette)
	{
		const std::string& name = registry.get(id).name;
		if (name.size() > kMaxNameBytes)
			throw std::runtime_error("saveWorld: block name longer than " + std::to_string(kMaxNameBytes) + " bytes: "
				+ name.substr(0, 32) + "...");
		w.u16(static_cast<std::uint16_t>(name.size()));
		w.raw(name.data(), name.size());
	}
	encodeRuns(
		blocks.size(), [&](std::size_t i) { return paletteIndex[known(blocks[i])]; },
		[&w](std::uint32_t run, std::uint16_t index) {
			w.u32(run);
			w.u16(index);
		});
	encodeRuns(
		cells.size(), [&cells](std::size_t i) { return stateByte(cells[i]); },
		[&w](std::uint32_t run, std::uint8_t state) {
			w.u32(run);
			w.u8(state);
		});
	w.crc();

	const std::vector<unsigned char>& bytes = w.bytes();
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	out.flush();
	if (!out)
		throw std::runtime_error("saveWorld: writing the stream failed");
}

World loadWorld(std::istream& in, const BlockRegistry& registry, GridSpec spec, LoadReport* report)
{
	// validates the spec (std::invalid_argument) and fixes the cell count before anything is read
	World world(spec);
	const std::size_t count = world.blocks().size();

	Reader r(in);
	unsigned char magic[sizeof(kMagic)];
	r.raw(magic, sizeof(magic));
	for (std::size_t i = 0; i < sizeof(kMagic); ++i)
		if (magic[i] != static_cast<unsigned char>(kMagic[i]))
			corrupt("not a world save (bad magic)");

	GridSpec file;
	file.origin.x = r.i32();
	file.origin.y = r.i32();
	file.origin.z = r.i32();
	file.size.x = r.i32();
	file.size.y = r.i32();
	file.size.z = r.i32();
	if (!(file == spec))
		corrupt("the save's grid doesn't match");

	const std::uint32_t paletteSize = r.u32();
	if (paletteSize == 0 || paletteSize > kMaxPalette)
		corrupt("bad palette size " + std::to_string(paletteSize));

	std::unordered_map<std::string_view, BlockId> byName;
	for (std::size_t id = 0; id < registry.size(); ++id)
		byName.emplace(registry.get(static_cast<BlockId>(id)).name, static_cast<BlockId>(id));

	LoadReport result;
	std::vector<BlockId> paletteIds;
	std::vector<bool> paletteKnown;
	std::unordered_set<std::string> seen;
	std::string name;
	for (std::uint32_t p = 0; p < paletteSize; ++p)
	{
		const std::uint16_t length = r.u16();
		if (length == 0 || length > kMaxNameBytes)
			corrupt("bad block name length " + std::to_string(length));
		name.assign(length, '\0');
		r.raw(reinterpret_cast<unsigned char*>(&name[0]), length);
		if (!seen.insert(name).second)
			corrupt("block name listed twice: " + name);
		BlockId id = kAir;
		bool known = true;
		if (name != "air")
		{
			const auto found = byName.find(name);
			known = found != byName.end();
			if (known)
				id = found->second;
			else
				result.unknownBlocks.push_back(name);
		}
		paletteIds.push_back(id);
		paletteKnown.push_back(known);
	}

	std::vector<BlockId> blocks(count, kAir);
	for (std::size_t i = 0; i < count;)
	{
		const std::size_t run = readRun(r, count - i);
		const std::uint16_t index = r.u16();
		if (index >= paletteSize)
			corrupt("palette index " + std::to_string(index) + " out of range");
		if (!paletteKnown[index])
			result.cellsDropped += run;
		std::fill(blocks.begin() + static_cast<std::ptrdiff_t>(i), blocks.begin() + static_cast<std::ptrdiff_t>(i + run),
			paletteIds[index]);
		i += run;
	}

	std::vector<CellState> cells(count, CellState::Open);
	for (std::size_t i = 0; i < count;)
	{
		const std::size_t run = readRun(r, count - i);
		const std::uint8_t state = r.u8();
		if (state >= kStateCount)
			corrupt("bad cell state " + std::to_string(state));
		std::fill(cells.begin() + static_cast<std::ptrdiff_t>(i), cells.begin() + static_cast<std::ptrdiff_t>(i + run),
			static_cast<CellState>(state));
		i += run;
	}

	const std::uint32_t expected = r.crc();
	if (r.u32() != expected)
		corrupt("checksum mismatch (the save is damaged)");

	// assign(), not set(): readers' journals must resync, and everything is dirty for the upload
	world.assign(std::move(blocks), std::move(cells));
	if (report)
		*report = std::move(result);
	return world;
}

} // namespace vox
