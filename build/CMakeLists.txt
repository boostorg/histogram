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

# -fpermissive needed for cpp_int in gcc-6
add_definitions(-Wall -std=c++11 -fpermissive)

set(STD_LIBRARIES stdc++ m)

if(BUILD_PYTHON)
  find_package(Boost 1.55 REQUIRED
               COMPONENTS iostreams serialization python)
  if(NOT DEFINED PYTHON_VERSION)
    execute_process(COMMAND python -c "import sys; sys.stdout.write('%i.%i.%i' % (sys.version_info.major,sys.version_info.minor,sys.version_info.micro))"
                    OUTPUT_VARIABLE PYTHON_VERSION)
  endif()
  find_package(PythonLibs ${PYTHON_VERSION} EXACT REQUIRED)
  find_package(PythonInterp ${PYTHON_VERSION} EXACT REQUIRED) # used by python_suite_test and FindNumpy

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
    ../src/python/module.cpp
    ../src/python/axis.cpp
    ../src/python/histogram.cpp
  )

  set(LIBRARIES ${STD_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
  target_link_libraries(histogram ${LIBRARIES})
  set_target_properties(histogram PROPERTIES PREFIX "" SUFFIX ".so")
else()
  find_package(Boost 1.55 REQUIRED
               COMPONENTS iostreams serialization)
  set(LIBRARIES ${STD_LIBRARIES} ${Boost_LIBRARIES})
endif()

include_directories(../include ${Boost_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE STREQUAL "debug")
  add_definitions(-O0 -g -fno-sanitize-recover=undefined)
  set(CMAKE_LD_FLAGS ${CMAKE_LD_FLAGS} -fno-sanitize-recover=undefined)
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
  add_executable(speed_cpp ../test/speed_cpp.cpp)
endif()

# tests
enable_testing()
add_executable(dynamic_histogram_test
               ../test/dynamic_histogram_test.cpp)
target_link_libraries(dynamic_histogram_test ${LIBRARIES})
add_test(dynamic_histogram_test dynamic_histogram_test)

add_executable(static_histogram_test
               ../test/static_histogram_test.cpp)
target_link_libraries(static_histogram_test ${LIBRARIES})
add_test(static_histogram_test static_histogram_test)

add_executable(axis_test
               ../test/axis_test.cpp)
target_link_libraries(axis_test ${LIBRARIES})
add_test(axis_test axis_test)

add_executable(adaptive_storage_test
               ../test/adaptive_storage_test.cpp)
target_link_libraries(adaptive_storage_test ${LIBRARIES})
add_test(adaptive_storage_test adaptive_storage_test)

add_executable(container_storage_test
               ../test/container_storage_test.cpp)
target_link_libraries(container_storage_test ${LIBRARIES})
add_test(container_storage_test container_storage_test)

add_executable(detail_test
               ../test/detail_test.cpp)
target_link_libraries(detail_test ${LIBRARIES})
add_test(detail_test detail_test)

if(HAVE_PYTHON)
  configure_file(../test/python_suite_test.py.in python_suite_test.py)
  add_test(python_suite_test python_suite_test.py)
  configure_file(../test/speed_numpy.py.in speed_numpy.py)
endif()

# for clang format and clang tidy
file(GLOB_RECURSE
     ALL_SOURCE_FILES
     LIST_DIRECTORIES false
     ../test/*_test.cpp ../include/*.hpp
     )

add_custom_target(
  clang-format
  COMMAND /usr/bin/clang-format
  -i
  -style=file
  ${ALL_SOURCE_FILES}
  )

get_property(INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(TIDY_INCLUDE)
foreach(x ${INCLUDE_DIRS})
  LIST(APPEND TIDY_INCLUDE -I${x})
endforeach()
add_custom_target(
  clang-tidy
  COMMAND /usr/bin/clang-tidy
  ${ALL_SOURCE_FILES}
  -checks=*,-*alpha*
  --
  -std=c++11
  ${TIDY_INCLUDE}
  )
