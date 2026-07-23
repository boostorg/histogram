# Copyright Peter Dimov, Hans Dembinski 2018-2019
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

# We support CMake 3.10, but prefer 4.4 policies and behavior
cmake_minimum_required(VERSION 3.10...4.4)

project(boost_histogram VERSION "${BOOST_SUPERPROJECT_VERSION}" LANGUAGES CXX)

add_library(boost_histogram INTERFACE)
add_library(Boost::histogram ALIAS boost_histogram)

target_include_directories(boost_histogram INTERFACE include)

target_link_libraries(boost_histogram
  INTERFACE
    Boost::config
    Boost::core
    Boost::mp11
    Boost::throw_exception
    Boost::variant2
)

target_compile_features(boost_histogram INTERFACE cxx_std_14)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Standalone build, fetch dependencies

  # Fetch support files

  # BoostFetch.cmake still uses FetchContent_Populate, an error under CMP0169 NEW
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()

  message(STATUS "Fetching BoostFetch.cmake")

  file(DOWNLOAD
    "https://raw.githubusercontent.com/boostorg/cmake/develop/include/BoostFetch.cmake"
    "${CMAKE_BINARY_DIR}/BoostFetch.cmake"
  )

  include("${CMAKE_BINARY_DIR}/BoostFetch.cmake")

  boost_fetch(boostorg/cmake TAG develop NO_ADD_SUBDIR)

  FetchContent_GetProperties(boostorg_cmake)

  list(APPEND CMAKE_MODULE_PATH ${boostorg_cmake_SOURCE_DIR}/include)

  # Enable testing

  include(CTest)
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG>)
  add_dependencies(check tests) # needed to build the "run" tests

  if(BUILD_TESTING)

    set(BUILD_TESTING OFF) # do not build tests of dependencies

    boost_fetch(boostorg/config TAG develop EXCLUDE_FROM_ALL)
    boost_fetch(boostorg/core TAG develop EXCLUDE_FROM_ALL)
    boost_fetch(boostorg/mp11 TAG develop EXCLUDE_FROM_ALL)
    boost_fetch(boostorg/throw_exception TAG develop EXCLUDE_FROM_ALL)
    boost_fetch(boostorg/variant2 TAG develop EXCLUDE_FROM_ALL)

    boost_fetch(boostorg/assert TAG develop EXCLUDE_FROM_ALL) # needed by core

    # this takes forever :(
    boost_fetch(boostorg/math TAG develop EXCLUDE_FROM_ALL) # needed by some tests

    ## skipping serialization, range, accumulators, units;
    ## the tests pull in too many dependencies
    # boost_fetch(boostorg/serialization TAG develop EXCLUDE_FROM_ALL)
    # boost_fetch(boostorg/units TAG develop EXCLUDE_FROM_ALL)
    # boost_fetch(boostorg/range TAG develop EXCLUDE_FROM_ALL)
    # boost_fetch(boostorg/accumulators TAG develop EXCLUDE_FROM_ALL)

    set(BUILD_TESTING ON)

  endif()

endif()

if (BUILD_TESTING)

  add_subdirectory(test)

  # do not pollute the superproject with the benchmarks
  if(NOT BOOST_SUPERPROJECT_VERSION)

    # google/benchmark requires CMake 3.13
    if(CMAKE_VERSION VERSION_LESS 3.13)
      message(STATUS "CMake 3.13+ required by google/benchmark; skipping benchmarks")
    else()
      add_subdirectory(benchmark)
    endif()

  endif()

endif()
