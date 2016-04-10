cmake_minimum_required (VERSION 2.8)

project(histogram)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

option(USE_PYTHON ON)
option(USE_NUMPY ON)
option(BUILD_CHECKS OFF)

if(${CMAKE_BUILD_TYPE})
  STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

find_package(Boost 1.55 REQUIRED
             COMPONENTS iostreams serialization unit_test_framework)
find_library(Boost_PYTHON_FOUND boost_python) # optional
find_package(PythonLibs) # optional
find_package(Numpy) # optional

add_definitions(-DBOOST_TEST_DYN_LINK) # for unit_test_framework
add_definitions(-Wall)
if(CMAKE_BUILD_TYPE STREQUAL "debug")
  message(STATUS "debug mode: optimizations off")
else()
  message(STATUS "release mode: optimizations on")
  add_definitions(-O3 -fomit-frame-pointer -mtune=generic)
endif()

include_directories(../include ${Boost_INCLUDE_DIRS})
set(LIBRARIES stdc++ m ${Boost_LIBRARIES})

# core library
add_library(boost_histogram SHARED
  ../src/axis.cpp
  ../src/histogram_base.cpp
  ../src/histogram.cpp
  ../src/nstore.cpp
  ../src/zero_suppression.cpp
)
target_link_libraries(boost_histogram ${LIBRARIES})

# python bindings
if(Boost_PYTHON_FOUND AND PYTHONLIBS_FOUND AND USE_PYTHON)

  include_directories(${PYTHON_INCLUDE_DIRS})
  add_definitions(-DUSE_PYTHON)

  if(NUMPY_FOUND AND USE_NUMPY)
    include_directories(${NUMPY_INCLUDE_DIR})
    add_definitions(-DUSE_NUMPY)
  endif()

  add_library(pyhistogram MODULE
    ../src/python/module.cpp
    ../src/python/histogram_base.cpp
    ../src/python/histogram.cpp
  )
  target_link_libraries(pyhistogram boost_histogram ${LIBRARIES} boost_python ${PYTHON_LIBRARIES})
  set_target_properties(pyhistogram PROPERTIES OUTPUT_NAME "histogram" PREFIX "" SUFFIX ".so")

endif()

# checks
if(BUILD_CHECKS)
  add_executable(sizeof
    ../test/check/sizeof.cpp
  )
  target_link_libraries(sizeof ${LIBRARIES})

  find_package(ROOT) # only used in speed comparison
  if(ROOT_FOUND)
    add_executable(nhistogram_speed
                   ../test/check/speed_vs_root.cpp)
    target_include_directories(nhistogram_speed PUBLIC ${ROOT_INCLUDE_DIR})
    target_link_libraries(nhistogram_speed boost_histogram ${ROOT_LIBRARIES} ${LIBRARIES})
  endif()
endif()

# tests
enable_testing()
add_executable(zero_suppression_test
               ../test/zero_suppression_test.cpp)
target_link_libraries(zero_suppression_test boost_histogram ${LIBRARIES})
add_test(zero_suppression_test zero_suppression_test)

add_executable(histogram_test
               ../test/histogram_test.cpp)
target_link_libraries(histogram_test boost_histogram ${LIBRARIES})
add_test(histogram_test histogram_test)

if(USE_PYTHON)
  add_test(python_suite_test ${PROJECT_SOURCE_DIR}/../test/python_suite_test.py)
  set_tests_properties(python_suite_test
    PROPERTIES ENVIRONMENT "PYTHONPATH=.:${PYTHONPATH}")
endif()

# install
install(DIRECTORY ../include/boost DESTINATION include)
install(TARGETS boost_histogram DESTINATION lib)
if (USE_PYTHON)
  execute_process(COMMAND python -c "from distutils.sysconfig import get_python_lib; import sys; sys.stdout.write(get_python_lib())"
                  OUTPUT_VARIABLE PYTHON_MODULE_DIRS) 
  set(PYTHON_MODULE_DIRS "${PYTHON_MODULE_DIRS}:$ENV{PYTHONPATH}")
  string(REPLACE ":" "\n  " PYTHON_MODULE_DIRS ${PYTHON_MODULE_DIRS})
  get_target_property(PYTHON_MODULE pyhistogram LOCATION)

  install(CODE "message(\"= How-to install Python module =\\nCopy\\n  ${PYTHON_MODULE}\\ninto one of these paths:\\n  ${PYTHON_MODULE_DIRS}\")")
endif()
