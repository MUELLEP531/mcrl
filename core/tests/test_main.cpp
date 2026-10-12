#include "check.hpp"

#include <cstdio>
#include <cstring>
#include <exception>

void check::report(const char* file, int line, const std::string& message)
{
	++failures();
	std::printf("  FAIL %s:%d: %s\n", file, line, message.c_str());
}

int main(int argc, char** argv)
{
	const char* filter = argc > 1 ? argv[1] : nullptr;
	int run = 0, failed = 0;
	for (const check::Case& c : check::registry())
	{
		if (filter && !std::strstr(c.name, filter))
			continue;
		++run;
		const int before = check::failures();
		try
		{
			c.fn();
		}
		catch (const check::Abort&)
		{
		}
		catch (const std::exception& e)
		{
			check::report(__FILE__, __LINE__, std::string("unexpected exception: ") + e.what());
		}
		catch (...)
		{
			check::report(__FILE__, __LINE__, "unexpected non-standard exception");
		}
		if (check::failures() != before)
		{
			++failed;
			std::printf("FAILED %s\n", c.name);
		}
	}
	std::printf("%d test(s), %d failed\n", run, failed);
	return failed == 0 && run > 0 ? 0 : 1;
}
