cmake_minimum_required(VERSION 3.18)
project(relationallib)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "RELEASE")

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# Optimisation comes from the Release build type above, which is -O3 on GCC
# and Clang and /O2 on MSVC. Assigning CMAKE_CXX_FLAGS directly would replace
# the toolchain's own defaults rather than add to them, and on MSVC that
# drops /EHsc, which pybind11 needs to propagate C++ exceptions.

# The macOS deployment target is deliberately left to the environment. It
# decides both what the compiler targets and what platform tag the wheel
# carries, so pinning it here while the build environment says otherwise
# would produce a wheel whose tag promises more than the binary delivers.

pybind11_add_module(relationallib src/kumo_relational_engine/csrc/neighbor_sampler.cpp)

# When building in editable mode, place the compiled extension next to the
# Python sources so `pip install -e .` works as expected.
set_target_properties(relationallib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/kumo_relational_engine"
)

# Install the extension into the kumo_relational_engine package when building a wheel.
install(TARGETS relationallib DESTINATION kumo_relational_engine)
