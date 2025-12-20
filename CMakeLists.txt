cmake_minimum_required(VERSION 3.14)
project(HelixDrone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")
endif()

find_package(pybind11 REQUIRED)

include_directories(cpp_core/include)

set(PHYSICS_SOURCES
    cpp_core/src/Quadrotor.cpp
    cpp_core/src/PhysicsEngine.cpp
)

pybind11_add_module(drone_core 
    cpp_core/src/bindings.cpp
    ${PHYSICS_SOURCES}
)

target_compile_definitions(drone_core PRIVATE _USE_MATH_DEFINES)

add_library(helix_physics STATIC ${PHYSICS_SOURCES})
target_include_directories(helix_physics PUBLIC cpp_core/include)
target_compile_definitions(helix_physics PRIVATE _USE_MATH_DEFINES)

add_executable(sota_test_physics tests/test_physics.cpp cpp_core/src/Quadrotor.cpp cpp_core/src/PhysicsEngine.cpp)
target_include_directories(sota_test_physics PRIVATE cpp_core/include)
target_compile_definitions(sota_test_physics PRIVATE _USE_MATH_DEFINES)
