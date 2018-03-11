cmake_minimum_required (VERSION 3.2)

project(histogram CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1024")
endif()

# setup build
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_NUMPY_SUPPORT "Build numpy support" ON)
option(BUILD_CHECKS "Build auxillary checks" OFF)
option(TRACE_ALLOCS "Debug: Trace allocations" OFF)
mark_as_advanced(TRACE_ALLOCS)

if(CMAKE_BUILD_TYPE)
  STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  add_definitions(/W3)
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_definitions(-Wall)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6)
    # -fpermissive needed for cpp_int in gcc-6
    # -Wno-noexcept-type to silence warnings related to Boost.Python
    add_definitions(-fpermissive -Wno-noexcept-type)
  endif()
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_definitions(-Wall)
endif()

if(TRACE_ALLOCS)
  add_definitions(-DBOOST_HISTOGRAM_TRACE_ALLOCS)
endif()

if(BUILD_PYTHON)
  if(DEFINED PYTHON_VERSION)
    find_package(PythonLibs ${PYTHON_VERSION} EXACT REQUIRED)
  else()
    find_package(PythonLibs REQUIRED)
  endif()

  message(STATUS "Python lib: ${PYTHON_LIBRARIES} (MUST match what Boost.Python was compiled against)")
  message(STATUS "Python include: ${PYTHON_INCLUDE_DIRS}")
  include_directories(${PYTHON_INCLUDE_DIRS})

  # used by python_suite_test, version is matched with PythonLibs
  # automatically in cmake-3.9.6, but this seems not to be the case in older
  # cmake versions
  find_package(PythonInterp ${PYTHONLIBS_VERSION_STRING} EXACT REQUIRED)

  if (PYTHON_VERSION_MAJOR EQUAL 3)
    set(BOOST_PYTHON_NAME python3)
  else()
    set(BOOST_PYTHON_NAME python)
  endif()

  if(BUILD_NUMPY_SUPPORT)
    find_package(Boost 1.63 REQUIRED
                 COMPONENTS iostreams serialization ${BOOST_PYTHON_NAME} numpy)
    add_definitions(-DHAVE_NUMPY)
  else()
    find_package(Boost 1.63
                 COMPONENTS iostreams serialization ${BOOST_PYTHON_NAME})
  endif()

  add_library(histogram SHARED
    ../src/python/module.cpp
    ../src/python/axis.cpp
    ../src/python/histogram.cpp
  )
  set(LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
  target_link_libraries(histogram ${LIBRARIES})
  set_target_properties(histogram PROPERTIES PREFIX "" SUFFIX ".so")
else()
  find_package(Boost 1.63 REQUIRED
               COMPONENTS serialization)
  set(LIBRARIES ${Boost_LIBRARIES})
endif()

include_directories(../include ${Boost_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE MATCHES debug)
  add_definitions(-O0 -g -fno-sanitize-recover=undefined)
  set(CMAKE_LD_FLAGS ${CMAKE_LD_FLAGS} -fno-sanitize-recover=undefined)
  message(STATUS "Build type: DEBUG [optimizations off]")
elseif(CMAKE_BUILD_TYPE MATCHES cov)
  add_definitions(-O0 -g)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} --coverage)
  message(STATUS "Build type: COVERAGE [optimizations off]")
elseif(CMAKE_BUILD_TYPE MATCHES perf)
  message(STATUS "Build type: PERF [optimizations on]")
  add_definitions(-O3)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DBOOST_DISABLE_ASSERTS)
  set(BUILD_CHECKS ON)
else()
  message(STATUS "Build type: RELEASE [optimizations on]")
  if(MSVC)
    add_definitions(/DBOOST_DISABLE_ASSERTS)
  else()
    add_definitions(-O3 -DBOOST_DISABLE_ASSERTS)
  endif()
endif()

# sync examples in documentation with example folder
add_custom_target(sync_examples ALL COMMAND python ${PROJECT_SOURCE_DIR}/../doc/sync_code.py)

# checks
if(BUILD_CHECKS)
  add_executable(speed_cpp ../test/speed_cpp.cpp)
  if (BUILD_PYTHON)
    configure_file(../test/speed_numpy.py speed_numpy.py)
  endif()
  add_executable(axis_size ../test/axis_size.cpp)
endif()

# tests
enable_testing()

if(BUILD_PYTHON)
  file(GLOB TEST_SOURCES ../test/*_test.py ../examples/*.py ${CMAKE_BINARY_DIR}/doc_test/*.py)
  foreach(SRC IN ITEMS ${TEST_SOURCES})
    if (SRC MATCHES "/([_a-zA-Z0-9]+)\\.py")
      set(TESTNAME ${CMAKE_MATCH_1})
      file(READ ${SRC} SRC_CONTENT)
      if (BUILD_NUMPY_SUPPORT OR
          (NOT BUILD_NUMPY_SUPPORT AND NOT SRC_CONTENT MATCHES "^import numpy|\nimport numpy"))
        configure_file(${SRC} ${TESTNAME}.py)
        add_test(${TESTNAME} ${PYTHON_EXECUTABLE} ${TESTNAME}.py)
      endif()
    endif()
  endforeach()
endif()

file(GLOB TEST_SOURCES ../test/*_test.cpp ../examples/*.cpp)
foreach(SRC IN ITEMS ${TEST_SOURCES})
  if(SRC MATCHES "/([_a-zA-Z0-9]+)\\.cpp")
    set(BASENAME ${CMAKE_MATCH_1})
    file(READ ${SRC} SRC_CONTENT)
    if(SRC_CONTENT MATCHES "BOOST_PYTHON_MODULE\\(([a-z_]+)\\)")
      if(BUILD_PYTHON)
        add_library(${CMAKE_MATCH_1} SHARED ${SRC})
        set_target_properties(${CMAKE_MATCH_1} PROPERTIES PREFIX "" SUFFIX ".so")
        target_link_libraries(${CMAKE_MATCH_1} ${LIBRARIES})
      endif()
    else()
      add_executable(${BASENAME} ${SRC})
      add_dependencies(${BASENAME} sync_examples)
      target_link_libraries(${BASENAME} ${LIBRARIES})
      add_test(${BASENAME} ${BASENAME})
    endif()
  endif()
endforeach()

# for clang format and clang tidy
file(GLOB_RECURSE
     ALL_SOURCE_FILES
     LIST_DIRECTORIES false
     ../test/*_test.cpp ../include/*.hpp
     )

add_custom_target(clf
  COMMAND clang-format
  -i
  ${ALL_SOURCE_FILES}
  )

get_property(INCLUDE_DIRS DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(TIDY_INCLUDE)
foreach(x ${INCLUDE_DIRS})
  LIST(APPEND TIDY_INCLUDE -I${x})
endforeach()
add_custom_target(
  clang-tidy
  COMMAND clang-tidy
  ${ALL_SOURCE_FILES}
  -checks=*,-*alpha*
  --
  -std=c++11
  ${TIDY_INCLUDE}
  )
