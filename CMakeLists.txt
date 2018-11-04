# Copyright (c) 2017 Hans Dembinski
#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)
#
cmake_minimum_required (VERSION 3.5)

project(histogram CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MIN_BOOST_VERSION 1.66)

if ("$ENV{BOOST_ROOT}" STREQUAL "")
  set(BOOST_INCLUDEDIR "${PROJECT_SOURCE_DIR}/../..")
endif()

# setup build
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TEST_SERIALIZATION "Test serialization code" OFF)

# serialization is optional
if (TEST_SERIALIZATION)
  find_package(Boost ${MIN_BOOST_VERSION} REQUIRED COMPONENTS serialization)
  # needed if Boost version is newer than CMake version
  if(NOT TARGET Boost::serialization)
    add_library(Boost::serialization IMPORTED INTERFACE)
    set_property(TARGET Boost::serialization PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
    set_property(TARGET Boost::serialization PROPERTY
        INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
  endif()
else ()
  find_package(Boost ${MIN_BOOST_VERSION} REQUIRED)
endif()

# tests
enable_testing()

function(compiled_test SRC)
  get_filename_component(BASENAME ${SRC} NAME_WE)
  add_executable(${BASENAME} ${SRC})
  target_include_directories(${BASENAME} PUBLIC include ${Boost_INCLUDE_DIR})

  if(CMAKE_BUILD_TYPE MATCHES coverage)
    target_compile_options(${BASENAME} PRIVATE --coverage)
    target_link_libraries(${BASENAME} PRIVATE --coverage)
  endif()

  # max warnings
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${BASENAME} PRIVATE -D_SCL_SECURE_NO_WARNINGS)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${BASENAME} PRIVATE -Wall -Wextra -g -O0)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6)
      # -Wno-noexcept-type needed for callable_traits
      target_compile_options(${BASENAME} PRIVATE -Wno-noexcept-type)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${BASENAME} PRIVATE -Wall -Wextra -g -O0
                           -D__STRICT_ANSI__)
  endif()

  # activate sanitizers for clang builds
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${BASENAME} PRIVATE
                           -fsanitize=address,undefined
                           -fno-omit-frame-pointer)
    target_link_libraries(${BASENAME} PRIVATE -fsanitize=address,undefined)
  endif()

  if (${BASENAME} MATCHES "_fail")
    if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      # fail tests stall infinitely on appveyor, needs proper solution
      add_test(${BASENAME} python ${PROJECT_SOURCE_DIR}/tools/pass_on_fail.py ${BASENAME})
    endif()
  else()
    add_test(${BASENAME} ${BASENAME})
  endif()
endfunction()

compiled_test(test/adaptive_storage_test.cpp)
compiled_test(test/storage_adaptor_test.cpp)
compiled_test(test/axis_regular_test.cpp)
compiled_test(test/axis_circular_test.cpp)
compiled_test(test/axis_variable_test.cpp)
compiled_test(test/axis_integer_test.cpp)
compiled_test(test/axis_category_test.cpp)
compiled_test(test/axis_variant_test.cpp)
compiled_test(test/detail_test.cpp)
compiled_test(test/histogram_dynamic_reduce_wrong_order_fail.cpp)
compiled_test(test/histogram_dynamic_test.cpp)
compiled_test(test/histogram_mixed_test.cpp)
compiled_test(test/histogram_test.cpp)
compiled_test(test/index_mapper_test.cpp)
compiled_test(test/meta_test.cpp)
compiled_test(test/utility_test.cpp)
compiled_test(test/weight_counter_test.cpp)

compiled_test(examples/getting_started_listing_01.cpp)
compiled_test(examples/getting_started_listing_02.cpp)
compiled_test(examples/guide_access_bin_counts.cpp)
compiled_test(examples/guide_axis_with_labels.cpp)
compiled_test(examples/guide_axis_with_uoflow_off.cpp)
compiled_test(examples/guide_custom_modified_axis.cpp)
compiled_test(examples/guide_custom_minimal_axis.cpp)
compiled_test(examples/guide_custom_storage.cpp)
compiled_test(examples/guide_fill_histogram.cpp)
compiled_test(examples/guide_histogram_operators.cpp)
compiled_test(examples/guide_histogram_reduction.cpp)
compiled_test(examples/guide_histogram_streaming.cpp)
compiled_test(examples/guide_make_dynamic_histogram.cpp)
compiled_test(examples/guide_make_static_histogram.cpp)

if (TEST_SERIALIZATION)
  compiled_test(examples/guide_histogram_serialization.cpp Boost::serialization)
  target_link_libraries(guide_histogram_serialization PUBLIC Boost::serialization)

  compiled_test(test/adaptive_storage_serialization_test.cpp Boost::serialization)
  target_link_libraries(adaptive_storage_serialization_test PUBLIC Boost::serialization)

  compiled_test(test/storage_adaptor_serialization_test.cpp)
  target_link_libraries(storage_adaptor_serialization_test PUBLIC Boost::serialization)

  compiled_test(test/histogram_serialization_test.cpp Boost::serialization)
  target_link_libraries(histogram_serialization_test PUBLIC Boost::serialization)
endif()

if (BUILD_BENCHMARKS)
  add_executable(speed_cpp test/speed_cpp.cpp)
  target_include_directories(speed_cpp PRIVATE include ${Boost_INCLUDE_DIR})
  target_compile_definitions(speed_cpp PRIVATE -DBOOST_DISABLE_ASSERTS)
  target_compile_options(speed_cpp PRIVATE -O3)
endif()
