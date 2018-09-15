cmake_minimum_required (VERSION 3.5)

project(histogram CXX)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MIN_BOOST_VERSION 1.66)

# setup build
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TEST_SERIALIZATION "Test serialization code" OFF)
mark_as_advanced(BUILD_BENCHMARKS)
mark_as_advanced(TEST_SERIALIZATION)

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
  add_compile_options(-D_SCL_SECURE_NO_WARNINGS)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wall -Wextra)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6)
    # -fpermissive needed for cpp_int in gcc-6
    # -Wno-noexcept-type to silence warnings related to Boost.Python
    add_compile_options(-fpermissive -Wno-noexcept-type)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wno-unused-local-typedef -D__STRICT_ANSI__)
endif()

# serialization only required for tests
if (TEST_SERIALIZATION)
  find_package(Boost ${MIN_BOOST_VERSION} REQUIRED COMPONENTS serialization)
else ()
  find_package(Boost ${MIN_BOOST_VERSION} REQUIRED)
endif()

include_directories(../include ${Boost_INCLUDE_DIRS})

if(CMAKE_BUILD_TYPE MATCHES coverage)
  add_compile_options(-O0 -g)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} --coverage) # enable linker flags as well
elseif(CMAKE_BUILD_TYPE MATCHES release)
  add_definitions(-DBOOST_DISABLE_ASSERTS)
endif()

# tests
enable_testing()

function(compiled_test SRC)
  get_filename_component(BASENAME ${SRC} NAME_WE)
  add_executable(${BASENAME} ${SRC})
  if (${BASENAME} MATCHES "serialization")
    target_link_libraries(${BASENAME} PUBLIC Boost::Serialization)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${BASENAME} PRIVATE
                           -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_libraries(${BASENAME} PRIVATE -fsanitize=address,undefined)
  endif()
  if (${BASENAME} MATCHES "_fail")
    add_test(${BASENAME} python ${PROJECT_SOURCE_DIR}/pass_on_fail.py ${BASENAME})
  else()
    add_test(${BASENAME} ${BASENAME})
  endif()
endfunction()

compiled_test(../test/adaptive_storage_test.cpp)
compiled_test(../test/array_storage_test.cpp)
compiled_test(../test/axis_test.cpp)
compiled_test(../test/detail_test.cpp)
compiled_test(../test/histogram_dynamic_add_fail.cpp)
compiled_test(../test/histogram_dynamic_at_tuple_wrong_dimension_fail.cpp)
compiled_test(../test/histogram_dynamic_at_vector_wrong_dimension_fail.cpp)
compiled_test(../test/histogram_dynamic_at_wrong_dimension_fail.cpp)
compiled_test(../test/histogram_dynamic_reduce_wrong_order_fail.cpp)
compiled_test(../test/histogram_static_add_fail.cpp)
compiled_test(../test/histogram_static_at_vector_wrong_dimension_fail.cpp)
compiled_test(../test/histogram_dynamic_test.cpp)
compiled_test(../test/histogram_mixed_test.cpp)
compiled_test(../test/histogram_test.cpp)
compiled_test(../test/index_mapper_test.cpp)
compiled_test(../test/meta_test.cpp)
compiled_test(../test/utility_test.cpp)
compiled_test(../test/weight_counter_test.cpp)

compiled_test(../examples/getting_started_listing_01.cpp)
compiled_test(../examples/getting_started_listing_02.cpp)
compiled_test(../examples/guide_access_bin_counts.cpp)
compiled_test(../examples/guide_axis_with_labels.cpp)
compiled_test(../examples/guide_axis_with_uoflow_off.cpp)
compiled_test(../examples/guide_custom_axis.cpp)
compiled_test(../examples/guide_custom_storage.cpp)
compiled_test(../examples/guide_fill_histogram.cpp)
compiled_test(../examples/guide_histogram_operators.cpp)
compiled_test(../examples/guide_histogram_reduction.cpp)
compiled_test(../examples/guide_histogram_streaming.cpp)
compiled_test(../examples/guide_make_dynamic_histogram.cpp)
compiled_test(../examples/guide_make_static_histogram.cpp)

if (TEST_SERIALIZATION)
  compiled_test(../examples/guide_histogram_serialization.cpp)
  compiled_test(../test/adaptive_storage_serialization_test.cpp)
endif()

if(BUILD_BENCHMARKS)
  add_executable(speed_cpp ../test/speed_cpp.cpp)
  target_compile_options(speed_cpp PRIVATE -O3)
endif()
