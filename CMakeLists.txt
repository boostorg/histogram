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
set(CMAKE_CXX_EXTENSIONS OFF)

set(MIN_BOOST_VERSION 1.66)

if ("$ENV{BOOST_ROOT}" STREQUAL "")
  set(BOOST_INCLUDEDIR "${PROJECT_SOURCE_DIR}/../..")
endif()

# setup build
option(TEST_SERIALIZATION "Test serialization code (requires boost.serialization)" ON)
option(TEST_ACCUMULATORS_SUPPORT "Test support for boost.accumulators" ON)
option(TEST_UNITS_SUPPORT "Test support for boost.units" ON)
option(TEST_RANGE_SUPPORT "Test support for boost.range" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

# serialization support is optional
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
  target_include_directories(${BASENAME} PRIVATE include ${Boost_INCLUDE_DIR})

  if(CMAKE_BUILD_TYPE MATCHES coverage)
    target_compile_options(${BASENAME} PRIVATE --coverage)
    target_link_libraries(${BASENAME} PRIVATE --coverage)
  endif()

  # max warnings and activate sanitizers for clang
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${BASENAME} PRIVATE -D_SCL_SECURE_NO_WARNINGS)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # cannot use sanitizers with gcc < 8, causes linker errors
    target_compile_options(${BASENAME} PRIVATE -Wall -Wextra -g -O0)
    if (${BASENAME} MATCHES "parallel")
      target_compile_options(${BASENAME} PRIVATE -pthread)
      target_link_libraries(${BASENAME} PRIVATE -pthread)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${BASENAME} PRIVATE
      -Wall -Wextra -g -O0 -D__STRICT_ANSI__
      -fsanitize=address,undefined
      -fsanitize-address-use-after-scope
      -fno-omit-frame-pointer
      -ftemplate-backtrace-limit=0)
    target_link_libraries(${BASENAME} PRIVATE
      -fsanitize=address,undefined
      -fsanitize-address-use-after-scope)
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
compiled_test(test/algorithm_project_test.cpp)
compiled_test(test/algorithm_reduce_test.cpp)
compiled_test(test/algorithm_sum_test.cpp)
compiled_test(test/axis_category_test.cpp)
compiled_test(test/axis_integer_test.cpp)
compiled_test(test/axis_option_test.cpp)
compiled_test(test/axis_regular_test.cpp)
compiled_test(test/axis_variable_test.cpp)
compiled_test(test/axis_variant_test.cpp)
compiled_test(test/detail_test.cpp)
compiled_test(test/histogram_dynamic_test.cpp)
compiled_test(test/histogram_growing_test.cpp)
compiled_test(test/histogram_mixed_test.cpp)
compiled_test(test/histogram_operators_test.cpp)
compiled_test(test/histogram_test.cpp)
compiled_test(test/indexed_test.cpp)
compiled_test(test/internal_accumulators_test.cpp)
compiled_test(test/meta_test.cpp)
compiled_test(test/linearize_test.cpp)
compiled_test(test/storage_adaptor_test.cpp)
compiled_test(test/utility_test.cpp)

compiled_test(examples/getting_started_listing_01.cpp)
compiled_test(examples/getting_started_listing_02.cpp)
compiled_test(examples/getting_started_listing_03.cpp)
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
compiled_test(examples/guide_parallel_filling.cpp)

if (TEST_SERIALIZATION)
  compiled_test(examples/guide_histogram_serialization.cpp Boost::serialization)
  target_link_libraries(guide_histogram_serialization PRIVATE Boost::serialization)

  compiled_test(test/adaptive_storage_serialization_test.cpp Boost::serialization)
  target_link_libraries(adaptive_storage_serialization_test PRIVATE Boost::serialization)

  compiled_test(test/storage_adaptor_serialization_test.cpp)
  target_link_libraries(storage_adaptor_serialization_test PRIVATE Boost::serialization)

  compiled_test(test/histogram_serialization_test.cpp Boost::serialization)
  target_link_libraries(histogram_serialization_test PRIVATE Boost::serialization)
endif()

if (TEST_UNITS_SUPPORT)
  compiled_test(test/boost_units_support_test.cpp)
endif()

if (TEST_ACCUMULATORS_SUPPORT) # test support for external boost::accumulators
  compiled_test(test/boost_accumulators_support_test.cpp)
  target_compile_definitions(boost_accumulators_support_test
    PRIVATE -DBOOST_HISTOGRAM_WITH_ACCUMULATORS_SUPPORT)
endif()

if (TEST_RANGE_SUPPORT) # test support for external boost::range
  compiled_test(test/boost_range_support_test.cpp)
endif()

if (cxx_std_17 IN_LIST CMAKE_CXX_COMPILE_FEATURES) # test C++17 features
  # check above is not enough: cxx_std_17 is set for gcc-5.5, but there is no support for
  # deduction guides until gcc-7
  file(WRITE ${CMAKE_BINARY_DIR}/feature_detect/deduction_guide_support.cpp
    "template <class T> struct A { A(T) {} }; A(int) -> A<int>; int main() {}")
  try_compile(COMPILER_HAS_DEDUCTION_GUIDE_SUPPORT
              ${CMAKE_BINARY_DIR}/feature_detect
              ${CMAKE_BINARY_DIR}/feature_detect/deduction_guide_support.cpp
              CXX_STANDARD 17)
  if (COMPILER_HAS_DEDUCTION_GUIDE_SUPPORT)
    compiled_test(test/deduction_guides_test.cpp)
    target_compile_features(deduction_guides_test PRIVATE cxx_std_17)
  endif()
endif()

if (BUILD_BENCHMARKS)
  find_package(benchmark) # uses Google Benchmark

  if (NOT TARGET benchmark::benchmark_main)
    message(FATAL_ERROR "Building benchmarks is requested, but Google Benchmark could not"
    " be found. Please install Google Benchmark from github or toggle BUILD_BENCHMARKS=OFF.")
  endif()

  function(benchmark SRC)
    get_filename_component(BASENAME ${SRC} NAME_WE)
    add_executable(${BASENAME} ${SRC})
    target_include_directories(${BASENAME} PRIVATE include ${Boost_INCLUDE_DIR} benchmark::benchmark)
    target_compile_definitions(${BASENAME} PRIVATE -DBOOST_DISABLE_ASSERTS)
    target_compile_options(${BASENAME} PRIVATE -O3)
    target_link_libraries(${BASENAME} PRIVATE benchmark::benchmark_main)
  endfunction()

  benchmark(test/iteration_bench.cpp)
  benchmark(test/axis_bench.cpp)
  benchmark(test/parallel_filling_bench.cpp)

  add_executable(speed_cpp test/speed_cpp.cpp)
  target_include_directories(speed_cpp PRIVATE include ${Boost_INCLUDE_DIR})
  target_compile_definitions(speed_cpp PRIVATE -DBOOST_DISABLE_ASSERTS)
  target_compile_options(speed_cpp PRIVATE -O3)

  add_executable(axis_size test/axis_size.cpp)
  target_include_directories(axis_size PRIVATE include ${Boost_INCLUDE_DIR})
  target_compile_options(axis_size PRIVATE -O3)
endif()
