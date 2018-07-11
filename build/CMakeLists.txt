cmake_minimum_required (VERSION 3.2)

project(histogram CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# setup build
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_NUMPY "Build numpy support" ON)
option(BUILD_CHECKS "Build auxillary checks" OFF)
option(BUILD_SERIALIZATION "Build serialization code" ON)
option(TRACE_ALLOCS "Debug: Trace allocations" OFF)
option(SANITIZE "Debug: Compile with santizers" OFF)
mark_as_advanced(BUILD_CHECKS)
mark_as_advanced(BUILD_SERIALIZATION)
mark_as_advanced(TRACE_ALLOCS)
mark_as_advanced(SANITIZE)

# set build type if none is specified
set(default_build_type "RELEASE")
if (EXISTS "${PROJECT_SOURCE_DIR}/../.git")
  set(default_build_type "Debug")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# set special compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/W4)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wall -Wextra -pedantic)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6)
    # -fpermissive needed for cpp_int in gcc-6
    # -Wno-noexcept-type to silence warnings related to Boost.Python
    add_compile_options(-fpermissive -Wno-noexcept-type)
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wall -Wextra -pedantic -D__STRICT_ANSI__ -ftemplate-depth=1024)
endif()

if(TRACE_ALLOCS)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/DBOOST_HISTOGRAM_TRACE_ALLOCS)
  else()
    add_compile_options(-DBOOST_HISTOGRAM_TRACE_ALLOCS)
  endif()
endif()

if(BUILD_PYTHON)
  # We need both find_packge(PythonInterp) and find_package(PythonLibs),
  # but as of cmake-3.11.4, cmake is not able to find matching versions of
  # libs and interpreter. So we just discover the python executable (can be
  # overriden with -DPYTHON_EXECUTABLE=/path/to/python in the shell) and
  # use Python facilities to find the correct libs and header paths.
  if(NOT DEFINED PYTHON_EXECUTABLE)
    find_package(PythonInterp REQUIRED)
  endif()

  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/get_python_include.py
                  OUTPUT_VARIABLE PYTHON_INCLUDE)

  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/get_python_library.py
                  OUTPUT_VARIABLE PYTHON_LIBRARY)

  message(STATUS "Python include: ${PYTHON_INCLUDE}")
  message(STATUS "Python lib: ${PYTHON_LIBRARY}")
  message(STATUS "Beware: Python lib MUST match what Boost.Python was compiled against")
  include_directories(${PYTHON_INCLUDE})

  find_package(Boost 1.63 REQUIRED COMPONENTS iostreams serialization)
  set(LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARY})

  find_library(Boost_PYTHON
               NAMES boost_python boost_python${PYTHON_VERSION_MAJOR} boost_python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
               PATHS ${Boost_LIBRARY_DIRS}
               NO_DEFAULT_PATH)
  list(APPEND LIBRARIES ${Boost_PYTHON})
  message(STATUS "Found boost_python: ${Boost_PYTHON}")

  if(BUILD_NUMPY)
    find_library(Boost_NUMPY
                 NAMES boost_numpy boost_numpy${PYTHON_VERSION_MAJOR} boost_numpy${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
                 PATHS ${Boost_LIBRARY_DIRS}
                 NO_DEFAULT_PATH)
    message(STATUS "Found boost_numpy: ${Boost_NUMPY}")
    add_definitions(-DHAVE_NUMPY)
    list(APPEND LIBRARIES ${Boost_NUMPY})
  endif()

  add_library(histogram SHARED
    ../src/python/module.cpp
    ../src/python/axis.cpp
    ../src/python/histogram.cpp
  )
  target_link_libraries(histogram ${LIBRARIES})
  set_target_properties(histogram PROPERTIES PREFIX "" SUFFIX ".so")
else()
  # serialization only required for tests
  if (BUILD_SERIALIZATION)
    find_package(Boost 1.63 REQUIRED serialization)
    add_definitions(-DHAVE_SERIALIZATION)
  else ()
    find_package(Boost 1.63 REQUIRED)
  endif()
  set(LIBRARIES ${Boost_LIBRARIES})
endif()

include_directories(../include ${Boost_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE MATCHES coverage)
  add_definitions(-O0 -g)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} --coverage)
elseif(CMAKE_BUILD_TYPE MATCHES release)
  if(MSVC)
    add_definitions(/DBOOST_DISABLE_ASSERTS)
  else()
    add_definitions(-DBOOST_DISABLE_ASSERTS)
  endif()
endif()

# checks
if(BUILD_CHECKS)
  add_executable(axis_size ../test/axis_size.cpp)
  # find_package(benchmark)
  add_executable(speed_cpp ../test/speed_cpp.cpp)
  # target_link_libraries(speed_cpp -lbenchmark -lpthread)
  if (BUILD_PYTHON)
    configure_file(../test/speed_numpy.py speed_numpy.py)
  endif()
endif()

# tests
enable_testing()

if(BUILD_PYTHON)
  file(GLOB TEST_SOURCES ../test/*_test.py ../examples/*.py ${CMAKE_BINARY_DIR}/doc_test/*.py)
  foreach(SRC IN ITEMS ${TEST_SOURCES})
    if (SRC MATCHES "/([_a-zA-Z0-9]+)\\.py")
      set(TESTNAME ${CMAKE_MATCH_1})
      file(READ ${SRC} SRC_CONTENT)
      if (BUILD_NUMPY OR
          (NOT BUILD_NUMPY AND NOT SRC_CONTENT MATCHES "^import numpy|\nimport numpy"))
        add_test(${TESTNAME} ${PYTHON_EXECUTABLE} ${SRC})
      endif()
    endif()
  endforeach()
endif()

file(GLOB TEST_SOURCES ../test/*_test.cpp ../examples/*.cpp)
foreach(SRC IN ITEMS ${TEST_SOURCES})
  if(SRC MATCHES "/([_a-zA-Z0-9]+)\\.cpp")
    set(BASENAME ${CMAKE_MATCH_1})
    if (NOT BUILD_SERIALIZATION AND BASENAME MATCHES "serialization")
      continue()
    endif()
    file(READ ${SRC} SRC_CONTENT)
    if(SRC_CONTENT MATCHES "BOOST_PYTHON_MODULE\\(([a-z_]+)\\)")
      if(BUILD_PYTHON)
        add_library(${CMAKE_MATCH_1} SHARED ${SRC})
        set_target_properties(${CMAKE_MATCH_1} PROPERTIES PREFIX "" SUFFIX ".so")
        target_link_libraries(${CMAKE_MATCH_1} ${LIBRARIES})
      endif()
    else()
      add_executable(${BASENAME} ${SRC})
      if (SANITIZE)
        target_compile_options(${BASENAME} PRIVATE -fsanitize=address,undefined)
        target_link_libraries(${BASENAME} -lasan -lubsan ${LIBRARIES})
      else()
        target_link_libraries(${BASENAME} ${LIBRARIES})
      endif()
      if (BASENAME MATCHES "fail")
        if (DEFINED PYTHON_EXECUTABLE)
          add_test(NAME ${BASENAME} COMMAND ${PYTHON_EXECUTABLE}
                   ../test/pass_on_fail.py ${BASENAME})
        endif()
      else()
        add_test(${BASENAME} ${BASENAME})
      endif()
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
