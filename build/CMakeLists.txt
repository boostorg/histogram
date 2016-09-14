cmake_minimum_required (VERSION 3.4)
# cmake_policy(SET CMP0042 NEW)

project(histogram)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# setup build
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_CHECKS "Build auxillary checks" OFF)

if(${CMAKE_BUILD_TYPE})
  STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

add_definitions(-Wall -std=c++11 -pedantic)
add_definitions(-DBOOST_TEST_DYN_LINK) # for unit_test_framework

find_package(Boost 1.55 REQUIRED
             COMPONENTS iostreams serialization unit_test_framework)

set(LIBRARIES stdc++ m ${Boost_LIBRARIES})

if(BUILD_PYTHON)
  find_package(PythonInterp)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; sys.stdout.write(sys.prefix)"
                  OUTPUT_VARIABLE PYTHON_ROOT)
  set(_VERSION ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
  string(REPLACE "." "" _VERSION_NO_DOTS ${_VERSION})
  find_library(PYTHON_LIBRARY 
               PATHS ${PYTHON_ROOT}/lib
               NAMES
                 python${_VERSION_NO_DOTS}
                 python${_VERSION}mu
                 python${_VERSION}m
                 python${_VERSION}u
                 python${_VERSION}
  )
  find_path(PYTHON_INCLUDE_DIR
    NAMES Python.h
    PATHS ${PYTHON_ROOT}/include
    PATH_SUFFIXES
      python${_VERSION}mu
      python${_VERSION}m
      python${_VERSION}u
      python${_VERSION}
  )
  message(STATUS "Python lib: ${PYTHON_LIBRARY}")
  message(STATUS "Python include: ${PYTHON_INCLUDE_DIR}")
  find_library(Boost_PYTHON_LIBRARY
               NAMES
                 boost_python-py${_VERSION_NO_DOTS}
                 boost_python
               PATHS ${Boost_LIBRARY_DIRS}
               NO_DEFAULT_PATHS
  )
  message(STATUS "Found boost.python: ${Boost_PYTHON_LIBRARY}")
  if(Boost_PYTHON_LIBRARY AND PYTHON_LIBRARY)
    set(HAVE_PYTHON 1)
    include_directories(${PYTHON_INCLUDE_DIR})

    if(USE_NUMPY)
      find_package(Numpy 1.7)
    endif()

    if(NUMPY_FOUND)
      set(HAVE_NUMPY 1)
      include_directories(${NUMPY_INCLUDE_DIR})
      add_definitions(-DHAVE_NUMPY)
    endif()

    set(LIBRARIES ${LIBRARIES} ${Boost_PYTHON_LIBRARY} ${PYTHON_LIBRARY})
  endif()

  if(HAVE_PYTHON)
    add_library(pyhistogram SHARED
      ../src/module.cpp
      ../src/axis.cpp
      ../src/histogram.cpp
    )
    target_link_libraries(pyhistogram ${LIBRARIES})
    set_target_properties(pyhistogram PROPERTIES OUTPUT_NAME "histogram" PREFIX "" SUFFIX ".so")
  endif()
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
  target_compile_definitions(speed_cpp_static PUBLIC histogramYPE=1)

  add_executable(speed_cpp_dynamic1
                 ../test/check/speed_cpp.cpp)
  target_link_libraries(speed_cpp_dynamic1 ${LIBRARIES})
  target_compile_definitions(speed_cpp_dynamic1 PUBLIC histogramYPE=2)

  add_executable(speed_cpp_dynamic2
                 ../test/check/speed_cpp.cpp)
  target_link_libraries(speed_cpp_dynamic2 ${LIBRARIES})
  target_compile_definitions(speed_cpp_dynamic2 PUBLIC histogramYPE=3)
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
