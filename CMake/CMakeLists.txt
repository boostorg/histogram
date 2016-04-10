cmake_minimum_required (VERSION 2.8)

project(histogram)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

if(${CMAKE_BUILD_TYPE})
  STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

find_package(Boost 1.55 REQUIRED
             COMPONENTS python iostreams serialization unit_test_framework)
find_package(PythonLibs) # optional
find_package(Numpy) # optional

add_definitions(-DBOOST_TEST_DYN_LINK) # for unit_test_framework
add_definitions(-Wall)
include_directories(../include ${Boost_INCLUDE_DIRS})
set(LIBRARIES stdc++ m ${Boost_LIBRARIES})

if(Boost_PYTHON_FOUND AND PYTHONLIBS_FOUND)
  set(USE_PYTHON True)

  if(USE_PYTHON)
    include_directories(${PYTHON_INCLUDE_DIRS})
    LIST(APPEND LIBRARIES ${PYTHON_LIBRARIES})
    add_definitions(-DUSE_PYTHON)
  endif()

  if(NUMPY_FOUND)
    set(USE_NUMPY True)
  endif()

  if(USE_NUMPY)
    include_directories(${NUMPY_INCLUDE_DIR})
    add_definitions(-DUSE_NUMPY)
  endif()
endif()

# core library
add_library(boost_histogram SHARED
  ../src/axis.cpp
  ../src/histogram_base.cpp
  ../src/histogram.cpp
  ../src/nstore.cpp
  ../src/zero_suppression.cpp
)
target_link_libraries(boost_histogram ${LIBRARIES})

if(CMAKE_BUILD_TYPE STREQUAL "debug")
  message(STATUS "debug mode: optimizations off")
else()
  message(STATUS "release mode: optimizations on")
  target_compile_options(boost_histogram PUBLIC "-O3 -fomit-frame-pointer -mtune=generic")
endif()

# python bindings
if(USE_PYTHON)
  add_library(pyhistogram MODULE
    ../src/python/module.cpp
    ../src/python/histogram_base.cpp
    ../src/python/histogram.cpp
  )
  target_link_libraries(pyhistogram boost_histogram ${LIBRARIES})
  set_target_properties(pyhistogram PROPERTIES OUTPUT_NAME "histogram" PREFIX "" SUFFIX ".so")
endif()

# examples
add_executable(sizeof
  ../examples/sizeof.cpp
)
target_link_libraries(sizeof ${LIBRARIES})

find_package(ROOT) # only used in speed comparison
if(ROOT_FOUND)
  add_executable(nhistogram_speed
                 ../examples/speed_vs_root.cpp)
  target_include_directories(nhistogram_speed PUBLIC ${ROOT_INCLUDE_DIR})
  target_link_libraries(nhistogram_speed boost_histogram ${ROOT_LIBRARIES} ${LIBRARIES})
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
  add_custom_target(python_suite_test ALL
                    ln -sf ${PROJECT_SOURCE_DIR}/../test/python_suite_test.py .)
  add_test(python_suite_test python_suite_test.py)
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
