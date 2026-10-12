cmake_minimum_required(VERSION 3.16)
project(mcrl_voxel LANGUAGES CXX)

# The voxel core builds on its own (tests run on Linux and Windows); the BakkesMod plugin links vox_core and rl_arena.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
	add_compile_options(/W4 /permissive- /utf-8)
	add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
else()
	add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
endif()

option(MCRL_WERROR "Treat warnings as errors" ON)
if(MCRL_WERROR)
	if(MSVC)
		add_compile_options(/WX)
	else()
		add_compile_options(-Werror)
	endif()
endif()

# --- game-agnostic core ---
file(GLOB VOX_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/core/src/*.cpp)
add_library(vox_core STATIC ${VOX_CORE_SOURCES})
target_include_directories(vox_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)

# --- Rocket League adapter pieces that need no game SDK ---
file(GLOB RL_ARENA_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/games/rocketleague/src/*.cpp)
add_library(rl_arena STATIC ${RL_ARENA_SOURCES})
target_include_directories(rl_arena PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/games/rocketleague/include)
target_link_libraries(rl_arena PUBLIC vox_core)

# --- tests (one executable; every *.cpp in the test folders registers its own cases) ---
file(GLOB VOX_TEST_SOURCES CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/core/tests/*.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/games/rocketleague/tests/*.cpp)
add_executable(vox_tests ${VOX_TEST_SOURCES})
target_include_directories(vox_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core/tests)
target_link_libraries(vox_tests PRIVATE vox_core rl_arena)

enable_testing()
add_test(NAME vox_tests COMMAND vox_tests)

# --- tools ---
file(GLOB VOX_TOOLS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/core/tools/*.cpp)
foreach(tool_source ${VOX_TOOLS})
	get_filename_component(tool_name ${tool_source} NAME_WE)
	add_executable(${tool_name} ${tool_source})
	target_link_libraries(${tool_name} PRIVATE vox_core rl_arena)
endforeach()
