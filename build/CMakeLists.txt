cmake_minimum_required (VERSION 3.2)

project(histogram CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# setup build
option(BUILD_PYTHON "Build python bindings" ON)
option(BUILD_NUMPY_SUPPORT "Build numpy support" ON)
option(BUILD_CHECKS "Build auxillary checks" OFF)

if(${CMAKE_BUILD_TYPE})
  STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

if(MSVC)
  add_definitions(/W3)
else()
  # -fpermissive needed for cpp_int in gcc-6
  add_definitions(-Wall -fpermissive)
endif()

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

  set(LIBRARIES ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
  target_link_libraries(histogram ${LIBRARIES})
  set_target_properties(histogram PROPERTIES PREFIX "" SUFFIX ".so")
else()
  find_package(Boost 1.55 REQUIRED
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

# checks
if(BUILD_CHECKS)
  add_executable(speed_cpp ../test/speed_cpp.cpp)
  if (HAVE_PYTHON)
    configure_file(../test/speed_numpy.py speed_numpy.py)
  endif()
endif()

# tests
enable_testing()

# run python ../doc/extract_code.py manually to make doc_test/ directory
if(HAVE_PYTHON)
  file(GLOB TEST_SOURCES ../test/*_test.py ../examples/*.py ${CMAKE_BINARY_DIR}/doc_test/*.py)
  foreach(SRC IN ITEMS ${TEST_SOURCES})
    if (SRC MATCHES "/([_a-zA-Z0-9]+)\\.py")
      configure_file(${SRC} ${CMAKE_MATCH_1}.py)
      add_test(${CMAKE_MATCH_1} ${PYTHON_EXECUTABLE} ${CMAKE_MATCH_1}.py)
    endif()
  endforeach()
endif()

file(GLOB TEST_SOURCES ../test/*_test.cpp ../examples/*.cpp ${CMAKE_BINARY_DIR}/doc_test/*.cpp)
foreach(SRC IN ITEMS ${TEST_SOURCES})
  if(SRC MATCHES "/module_([_a-zA-Z0-9]+)\\.cpp")
    add_library(${CMAKE_MATCH_1} SHARED ${SRC})
    set_target_properties(${CMAKE_MATCH_1} PROPERTIES PREFIX "" SUFFIX ".so")
    target_link_libraries(${CMAKE_MATCH_1} ${LIBRARIES})
  else()
    if(SRC MATCHES "/([_a-zA-Z0-9]+)\\.cpp")
      add_executable(${CMAKE_MATCH_1} ${SRC})
      target_link_libraries(${CMAKE_MATCH_1} ${LIBRARIES})
      add_test(${CMAKE_MATCH_1} ${CMAKE_MATCH_1})
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
