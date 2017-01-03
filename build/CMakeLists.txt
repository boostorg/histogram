cmake_minimum_required (VERSION 2.8)

project(histogram CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# setup build
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_NUMPY_SUPPORT "Build numpy support" ON)
option(BUILD_CHECKS "Build auxillary checks" OFF)

if(${CMAKE_BUILD_TYPE})
  STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

add_definitions(-Wall -pedantic -std=c++11)
add_definitions(-DBOOST_TEST_DYN_LINK) # for unit_test_framework

set(STD_LIBRARIES stdc++ m)

if(BUILD_PYTHON)
  find_package(Boost 1.55 REQUIRED
               COMPONENTS iostreams serialization python unit_test_framework)
  if(PYTHON_VERSION)
    find_package(PythonLibs ${PYTHON_VERSION} EXACT REQUIRED)
    find_package(PythonInterp ${PYTHON_VERSION} EXACT REQUIRED) # used by python_suite_test and FindNumpy
  else()
    find_package(PythonLibs REQUIRED)
    find_package(PythonInterp REQUIRED) # used by python_suite_test and FindNumpy
  endif()

  message(STATUS "Python lib: ${PYTHON_LIBRARIES}")
  message(STATUS "Python include: ${PYTHON_INCLUDE_DIRS}")
  set(HAVE_PYTHON 1)
  include_directories(${PYTHON_INCLUDE_DIRS})

  if(BUILD_NUMPY_SUPPORT)
    find_package(Numpy 1.7 REQUIRED)
    set(HAVE_NUMPY 1)
    include_directories(${NUMPY_INCLUDE_DIR})
    add_definitions(-DHAVE_NUMPY)
  endif()

  add_library(histogram SHARED
    ../src/module.cpp
    ../src/axis.cpp
    ../src/histogram.cpp
  )

  set(LIBRARIES ${STD_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
  target_link_libraries(histogram ${LIBRARIES})
  set_target_properties(histogram PROPERTIES PREFIX "" SUFFIX ".so")
else()
  find_package(Boost 1.55 REQUIRED
             COMPONENTS iostreams serialization unit_test_framework)
  set(LIBRARIES ${STD_LIBRARIES} ${Boost_LIBRARIES})
endif()

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
  add_executable(speed_cpp_static
                 ../test/check/speed_cpp.cpp)
  target_link_libraries(speed_cpp_static ${LIBRARIES})
  target_compile_definitions(speed_cpp_static PUBLIC HISTOGRAM_TYPE=1)

  add_executable(speed_cpp_dynamic1
                 ../test/check/speed_cpp.cpp)
  target_link_libraries(speed_cpp_dynamic1 ${LIBRARIES})
  target_compile_definitions(speed_cpp_dynamic1 PUBLIC HISTOGRAM_TYPE=2)

  add_executable(speed_cpp_dynamic2
                 ../test/check/speed_cpp.cpp)
  target_link_libraries(speed_cpp_dynamic2 ${LIBRARIES})
  target_compile_definitions(speed_cpp_dynamic2 PUBLIC HISTOGRAM_TYPE=3)
endif()

# tests
enable_testing()
add_executable(dynamic_histogram_test
               ../test/dynamic_histogram_test.cpp)
target_link_libraries(dynamic_histogram_test ${LIBRARIES})
add_test(dynamic_histogram_test dynamic_histogram_test)

add_executable(axis_test
               ../test/axis_test.cpp)
target_link_libraries(axis_test ${LIBRARIES})
add_test(axis_test axis_test)

add_executable(dynamic_storage_test
               ../test/dynamic_storage_test.cpp)
target_link_libraries(dynamic_storage_test ${LIBRARIES})
add_test(dynamic_storage_test dynamic_storage_test)

add_executable(utility_test
               ../test/utility_test.cpp)
target_link_libraries(utility_test ${LIBRARIES})
add_test(utility_test utility_test)

if(HAVE_PYTHON)
  configure_file(../test/python_suite_test.py.in python_suite_test.py)
  add_test(python_suite_test python_suite_test.py)
endif()
