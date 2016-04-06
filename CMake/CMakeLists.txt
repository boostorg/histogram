project(histogram)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

if(${CMAKE_BUILD_TYPE})
  STRING(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
endif()

find_package(Boost 1.55 REQUIRED COMPONENTS python iostreams serialization unit_test_framework)
find_package(PythonLibs)
find_package(Numpy) # optional
find_package(ROOT) # only used in one of the tests

include_directories(include ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
add_definitions(-DBOOST_TEST_DYN_LINK) # for unit_test_framework
set(LIBRARIES stdc++ ${Boost_LIBRARIES})

if(Boost_PYTHON_FOUND)
  LIST(APPEND LIBRARIES ${PYTHON_LIBRARIES})
endif()

if(NUMPY_FOUND)
  include_directories(${NUMPY_INCLUDE_DIR})
  add_definitions(-DUSE_NUMPY)
endif()

add_library(histogram SHARED
  src/axis.cpp
  src/histogram_base.cpp
  src/nhistogram.cpp
  src/nstore.cpp
  src/zero_suppression.cpp
  # src/whistogram.cpp
)
target_link_libraries(histogram ${LIBRARIES})

if(CMAKE_BUILD_TYPE STREQUAL "debug")
  message(STATUS "debug mode: optimizations off")
else()
  message(STATUS "release mode: optimizations on")
  target_compile_options(histogram PUBLIC "-O3 -fomit-frame-pointer -mtune=generic")
endif()

if(Boost_PYTHON_FOUND)
  add_library(pyhistogram MODULE
    src/python/module.cpp
    src/python/histogram_base.cpp
    src/python/nhistogram.cpp
  )
  target_link_libraries(pyhistogram histogram ${LIBRARIES})
  set_target_properties(pyhistogram PROPERTIES OUTPUT_NAME "histogram" PREFIX "" SUFFIX ".so")
endif()

add_executable(sizeof_axis
  examples/sizeof_axis.cpp
)
target_link_libraries(sizeof_axis ${LIBRARIES})

if(ROOT_FOUND)
  add_executable(nhistogram_speed
                 examples/speed_vs_root.cpp)
  target_include_directories(nhistogram_speed PUBLIC ${ROOT_INCLUDE_DIR})
  target_link_libraries(nhistogram_speed histogram ${ROOT_LIBRARIES} ${LIBRARIES})
endif()

# tests
enable_testing()
add_executable(zero_suppression_test
               test/zero_suppression_test.cpp)
target_link_libraries(zero_suppression_test histogram ${LIBRARIES})
add_test(zero_suppression_test zero_suppression_test)

if(Boost_PYTHON_FOUND)
  add_custom_target(python_suite_test ALL
                    cp ${CMAKE_SOURCE_DIR}/test/python_suite_test.py .)
  add_test(python_suite_test python_suite_test.py)
endif()
