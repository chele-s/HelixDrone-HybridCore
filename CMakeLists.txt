cmake_minimum_required(VERSION 3.14)
project(HelixDrone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /arch:AVX2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
find_package(pybind11 REQUIRED)

include_directories(cpp_core/include)

set(PHYSICS_SOURCES
    cpp_core/src/Quadrotor.cpp
    cpp_core/src/PhysicsEngine.cpp
    cpp_core/src/SOTAActuator.cpp
    cpp_core/src/StateEstimator.cpp
    cpp_core/src/ActuatorFailure.cpp
    cpp_core/src/CollisionWorld.cpp
    cpp_core/src/PayloadDynamics.cpp
)

pybind11_add_module(drone_core 
    cpp_core/src/bindings.cpp
    ${PHYSICS_SOURCES}
)

target_compile_definitions(drone_core PRIVATE _USE_MATH_DEFINES)

add_library(helix_physics STATIC ${PHYSICS_SOURCES})
target_include_directories(helix_physics PUBLIC cpp_core/include)
target_compile_definitions(helix_physics PRIVATE _USE_MATH_DEFINES)

add_executable(sota_test_physics tests/test_physics.cpp cpp_core/src/Quadrotor.cpp cpp_core/src/PhysicsEngine.cpp cpp_core/src/SOTAActuator.cpp)
target_include_directories(sota_test_physics PRIVATE cpp_core/include)
target_compile_definitions(sota_test_physics PRIVATE _USE_MATH_DEFINES)
