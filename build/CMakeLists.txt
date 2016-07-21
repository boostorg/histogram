cmake_minimum_required (VERSION 2.8)
# cmake_policy(SET CMP0042 NEW)

project(histogram)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# setup build
option(BUILD_CHECKS "Build auxillary checks" OFF)

if(${CMAKE_BUILD_TYPE})
  STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

add_definitions(-Wall -std=c++11)
add_definitions(-DBOOST_TEST_DYN_LINK) # for unit_test_framework

find_package(Boost 1.55 REQUIRED
             COMPONENTS iostreams serialization unit_test_framework)

set(LIBRARIES stdc++ m ${Boost_LIBRARIES})

include_directories(../include ${Boost_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE STREQUAL "debug")
  add_definitions(-O0 -g)
  message(STATUS "Build type: DEBUG [optimizations off]")
elseif(CMAKE_BUILD_TYPE STREQUAL "cov")
  add_definitions(-O0 -g)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} --coverage)
  message(STATUS "Build type: COVERAGE [optimizations off]")
elseif(CMAKE_BUILD_TYPE STREQUAL "perf")
  message(STATUS "Build type: PERF [optimizations on]")
  add_definitions(-O3 -g -fno-omit-frame-pointer)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DBOOST_DISABLE_ASSERTS)
  set(BUILD_CHECKS ON)
else()
  message(STATUS "Build type: RELEASE [optimizations on]")
  add_definitions(-O3 -fomit-frame-pointer)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DBOOST_DISABLE_ASSERTS)
endif()

# checks
if(BUILD_CHECKS)
  add_executable(speed_cpp
                 ../test/check/speed_cpp.cpp)
  target_link_libraries(speed_cpp ${LIBRARIES})
endif()

# tests
enable_testing()
add_executable(histogram_test
               ../test/histogram_test.cpp)
target_link_libraries(histogram_test ${LIBRARIES})
add_test(histogram_test histogram_test)

add_executable(axis_test
               ../test/axis_test.cpp)
target_link_libraries(axis_test ${LIBRARIES})
add_test(axis_test axis_test)

add_executable(zero_suppression_test
               ../test/zero_suppression_test.cpp)
target_link_libraries(zero_suppression_test ${LIBRARIES})
add_test(zero_suppression_test zero_suppression_test)

add_executable(dynamic_storage_test
               ../test/dynamic_storage_test.cpp)
target_link_libraries(dynamic_storage_test ${LIBRARIES})
add_test(dynamic_storage_test dynamic_storage_test)
