// A minimal test harness (no dependencies). In a test file:
//
//     #include "check.hpp"
//     TEST(world_starts_empty) { vox::World w; CHECK(w.get({0, 60, 0}) == vox::kAir); }
//
// CHECK records a failure and continues; REQUIRE stops the current test. CHECK_THROWS expects an exception.
// test_main.cpp runs every registered test; pass a substring to run only matching tests.
#pragma once

#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace check {

struct Case
{
	const char* name;
	std::function<void()> fn;
};

inline std::vector<Case>& registry()
{
	static std::vector<Case> cases;
	return cases;
}

struct Registrar
{
	Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

struct Abort
{
};

inline int& failures()
{
	static int count = 0;
	return count;
}

void report(const char* file, int line, const std::string& message);

} // namespace check

#define CHECK_CONCAT_INNER(a, b) a##b
#define CHECK_CONCAT(a, b) CHECK_CONCAT_INNER(a, b)

#define TEST(name)                                                                                                     \
	static void name();                                                                                                \
	static const check::Registrar CHECK_CONCAT(check_registrar_, name)(#name, name);                                  \
	static void name()

#define CHECK(cond)                                                                                                    \
	do                                                                                                                 \
	{                                                                                                                  \
		if (!(cond))                                                                                                   \
			check::report(__FILE__, __LINE__, "CHECK(" #cond ")");                                                    \
	} while (0)

#define REQUIRE(cond)                                                                                                  \
	do                                                                                                                 \
	{                                                                                                                  \
		if (!(cond))                                                                                                   \
		{                                                                                                              \
			check::report(__FILE__, __LINE__, "REQUIRE(" #cond ")");                                                  \
			throw check::Abort{};                                                                                      \
		}                                                                                                              \
	} while (0)

#define CHECK_EQ(a, b)                                                                                                 \
	do                                                                                                                 \
	{                                                                                                                  \
		const auto& check_a_ = (a);                                                                                    \
		const auto& check_b_ = (b);                                                                                    \
		if (!(check_a_ == check_b_))                                                                                   \
		{                                                                                                              \
			std::ostringstream check_s_;                                                                               \
			check_s_ << "CHECK_EQ(" #a ", " #b "): " << check_a_ << " != " << check_b_;                               \
			check::report(__FILE__, __LINE__, check_s_.str());                                                        \
		}                                                                                                              \
	} while (0)

#define CHECK_NEAR(a, b, eps)                                                                                          \
	do                                                                                                                 \
	{                                                                                                                  \
		const double check_a_ = static_cast<double>(a);                                                                \
		const double check_b_ = static_cast<double>(b);                                                                \
		if (!(std::fabs(check_a_ - check_b_) <= static_cast<double>(eps)))                                            \
		{                                                                                                              \
			std::ostringstream check_s_;                                                                               \
			check_s_ << "CHECK_NEAR(" #a ", " #b "): " << check_a_ << " vs " << check_b_;                             \
			check::report(__FILE__, __LINE__, check_s_.str());                                                        \
		}                                                                                                              \
	} while (0)

#define CHECK_THROWS(expr)                                                                                             \
	do                                                                                                                 \
	{                                                                                                                  \
		bool check_threw_ = false;                                                                                     \
		try                                                                                                            \
		{                                                                                                              \
			(void)(expr);                                                                                              \
		}                                                                                                              \
		catch (const check::Abort&)                                                                                    \
		{                                                                                                              \
			throw;                                                                                                     \
		}                                                                                                              \
		catch (...)                                                                                                    \
		{                                                                                                              \
			check_threw_ = true;                                                                                       \
		}                                                                                                              \
		if (!check_threw_)                                                                                             \
			check::report(__FILE__, __LINE__, "CHECK_THROWS(" #expr ")");                                             \
	} while (0)
