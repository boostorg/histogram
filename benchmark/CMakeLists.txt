# Copyright 2019 Hans Dembinski
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

if(CMAKE_VERSION VERSION_LESS 3.6)
  FetchContent_Declare(google_benchmark QUIET GIT_REPOSITORY "https://github.com/google/benchmark" GIT_TAG v1.4.1)
else()
  FetchContent_Declare(google_benchmark QUIET GIT_REPOSITORY "https://github.com/google/benchmark" GIT_TAG v1.4.1 GIT_SHALLOW 1)
endif()

FetchContent_GetProperties(google_benchmark)

if(NOT google_benchmark_POPULATED)
    message(STATUS "Fetching Google::benchmark")
    FetchContent_Populate(google_benchmark)
    # options should be set with CMAKE_ARGS, but doesn't seem to work
    option(BENCHMARK_ENABLE_TESTING OFF)
    set(CMAKE_BUILD_TYPE_saved ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)
    add_subdirectory(${google_benchmark_SOURCE_DIR} ${google_benchmark_BINARY_DIR})
    set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_saved})
endif()

macro(add_benchmark name)
  add_executable(${name} "${name}.cpp")
  target_compile_options(${name} PRIVATE -O3 -march=native -fno-exceptions -fno-rtti -funsafe-math-optimizations)
  target_link_libraries(${name} PRIVATE Boost::histogram benchmark_main)
endmacro()

add_benchmark(axis_index)
add_benchmark(histogram_filling)
add_benchmark(histogram_filling_experiments)
add_benchmark(histogram_iteration)
if (Threads_FOUND)
  add_benchmark(histogram_parallel_filling)
endif()

find_package(ROOT)
if (ROOT_FOUND)
  add_benchmark(histogram_filling_root)
  target_include_directories(histogram_filling_root PRIVATE ${ROOT_INCLUDE_DIRS})
  target_link_libraries(histogram_filling_root
    PRIVATE ${ROOT_LIBRARIES} benchmark_main)
  target_compile_options(histogram_filling_root PRIVATE -frtti -fexceptions)
  # target_link_options(histogram_filling_root
  #   PRIVATE ${ROOT_EXE_LINKER_FLAGS})
endif()

find_package(GSL)
if (GSL_FOUND)
  add_benchmark(histogram_filling_gsl)
  target_include_directories(histogram_filling_gsl PRIVATE ${GSL_INCLUDE_DIRS})
  target_link_libraries(histogram_filling_gsl
    PRIVATE ${GSL_LIBRARIES} benchmark_main)
endif()
