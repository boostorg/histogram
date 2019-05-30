# Copyright 2019 Hans Dembinski
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

set(CMAKE_BUILD_TYPE Release) # ok, only set in local scope

option(BENCHMARK_ENABLE_TESTING "" OFF)
boost_fetch(hdembinski/benchmark)

macro(add_benchmark name)
  add_executable(${name} "${name}.cpp")
  target_compile_options(${name} PRIVATE -DNDEBUG -O3 -fno-exceptions -fno-rtti -funsafe-math-optimizations)
  target_link_libraries(${name} PRIVATE Boost::histogram benchmark_main)
endmacro()

add_benchmark(axis_index)
add_benchmark(histogram_filling)
add_benchmark(histogram_filling_experiments)
add_benchmark(histogram_iteration)
if (Threads_FOUND)
  add_benchmark(histogram_parallel_filling)
endif()

find_package(GSL)
if (GSL_FOUND)
  add_benchmark(histogram_filling_gsl)
  target_include_directories(histogram_filling_gsl PRIVATE ${GSL_INCLUDE_DIRS})
  target_link_libraries(histogram_filling_gsl
    PRIVATE ${GSL_LIBRARIES} benchmark_main)
endif()

find_package(ROOT QUIET)
if (ROOT_FOUND)
  add_benchmark(histogram_filling_root)
  target_include_directories(histogram_filling_root PRIVATE ${ROOT_INCLUDE_DIRS})
  target_link_libraries(histogram_filling_root
    PRIVATE ${ROOT_LIBRARIES} benchmark_main)
  target_compile_options(histogram_filling_root PRIVATE -frtti -fexceptions)
  # target_link_options(histogram_filling_root
  #   PRIVATE ${ROOT_EXE_LINKER_FLAGS})
endif()
