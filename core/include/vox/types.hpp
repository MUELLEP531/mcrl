// Basic types shared by the voxel core. No game code, no platform headers: this compiles into a BakkesMod plugin
// (where windows.h defines min/max macros, so never write std::min(...) unparenthesised: use (std::min)(...)).
#pragma once

#include <cmath>
#include <cstdint>

namespace vox {

/// A block type's index in the BlockRegistry; also the value written into the shader's MCVOX texture.
using BlockId = std::uint16_t;
constexpr BlockId kAir = 0;

/// A cell (block) position in world coordinates: Minecraft's axes, x east, y up, z south.
struct Int3
{
	int x = 0, y = 0, z = 0;

	friend bool operator==(const Int3& a, const Int3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
	friend bool operator!=(const Int3& a, const Int3& b) { return !(a == b); }
	friend Int3 operator+(const Int3& a, const Int3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
	friend Int3 operator-(const Int3& a, const Int3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
};

/// A point or direction in world coordinates (1 unit = 1 block = 1 metre).
struct Vec3
{
	float x = 0.0f, y = 0.0f, z = 0.0f;

	friend Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
	friend Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
	friend Vec3 operator*(const Vec3& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
	friend Vec3 operator*(float s, const Vec3& a) { return {a.x * s, a.y * s, a.z * s}; }
};

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }

/// The cell containing a point.
inline Int3 cellAt(const Vec3& p)
{
	return {static_cast<int>(std::floor(p.x)), static_cast<int>(std::floor(p.y)), static_cast<int>(std::floor(p.z))};
}

/// An axis-aligned box in world coordinates (a player's or car's bounds), min <= max.
struct Aabb
{
	Vec3 min, max;
};

/// An 8-bit sRGB colour.
struct Rgb
{
	std::uint8_t r = 255, g = 255, b = 255;

	friend bool operator==(const Rgb& p, const Rgb& q) { return p.r == q.r && p.g == q.g && p.b == q.b; }
	friend bool operator!=(const Rgb& p, const Rgb& q) { return !(p == q); }
};

} // namespace vox
