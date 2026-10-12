// The block types a world can hold: their textures (atlas slots), tint and behaviour.
#pragma once

#include "vox/types.hpp"

#include <cstddef>
#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vox {

/// Bits of BlockType::tintFaces (the shader's face indices: 0 top, 1 side, 2 bottom).
enum TintFace : std::uint8_t
{
	kTintTop = 1,
	kTintSide = 2,
	kTintBottom = 4,
};

struct BlockType
{
	/// Unique name, lowercase with underscores ("stone", "grass_block"). Saved worlds store blocks by name.
	std::string name;
	/// Atlas slots (0..4095) of the top, side and bottom faces; -1: no texture (drawn in `color`).
	int texTop = -1, texSide = -1, texBottom = -1;
	/// Collides with players, cars and the ball.
	bool solid = true;
	/// Drawn. False for the arena barrier: exported to the shader as air.
	bool visible = true;
	/// Can be broken by a player.
	bool breakable = true;
	/// Faces multiplied by `tint` (TintFace bits): grass tops, leaves.
	std::uint8_t tintFaces = 0;
	Rgb tint{255, 255, 255};
	/// The block's own colour, for faces without a texture.
	Rgb color{128, 128, 128};
};

/// Block types by id. Id 0 is always air (not solid, not visible, not breakable).
class BlockRegistry
{
public:
	/// The shader's MCTYPES texture holds 256 x 128 ids.
	static constexpr std::size_t kMaxBlocks = 32768;
	/// The atlas holds 64 x 64 slots.
	static constexpr int kAtlasSlots = 4096;

	/// A registry holding only air.
	BlockRegistry();

	/// Adds a type and returns its id (the next free one). Throws std::invalid_argument if the name is empty, contains
	/// whitespace or is taken, if an atlas slot is outside -1..4095, or if the registry is full.
	BlockId add(BlockType type);

	/// The type of `id`; throws std::out_of_range for an id that was never added.
	const BlockType& get(BlockId id) const;
	/// The id of the type named `name`, if any.
	std::optional<BlockId> find(std::string_view name) const;
	/// Number of ids in use, including air.
	std::size_t size() const { return types_.size(); }
	bool contains(BlockId id) const { return id < types_.size(); }

	/// Reads the text format written by tools/assets/build_assets.py (`mcrl_blocks.txt`):
	///
	///     # comment (also blank lines)
	///     <id> <name> <top> <side> <bottom> <solid> <visible> <breakable> <tint_faces> <tint> <color>
	///
	/// with ids consecutive from 1 (air is implicit), slots as integers (-1 none), flags as 0/1, tint_faces as a
	/// decimal bit mask and colours as #rrggbb. Throws std::runtime_error naming the line on any malformed input.
	static BlockRegistry parse(std::istream& in);
	/// parse() from a file; throws std::runtime_error if it cannot be opened.
	static BlockRegistry loadFile(const std::string& path);
	/// Writes the format parse() reads (round-trips exactly).
	void write(std::ostream& out) const;

private:
	std::vector<BlockType> types_;
};

} // namespace vox
