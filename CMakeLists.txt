# Copyright 2018-2019 Peter Dimov, Hans Dembinski
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

# Beware: Boost-CMake support is experimental; testing works, install target doesn't

cmake_minimum_required(VERSION 3.6)

project(BoostHistogram VERSION 1.70.0 LANGUAGES CXX)

add_library(boost_histogram INTERFACE)
set_property(TARGET boost_histogram PROPERTY EXPORT_NAME histogram)

add_library(Boost::histogram ALIAS boost_histogram)

target_compile_features(boost_histogram INTERFACE
  cxx_alias_templates cxx_variadic_templates cxx_decltype_auto
  cxx_defaulted_functions cxx_generic_lambdas cxx_range_for
  cxx_relaxed_constexpr cxx_return_type_deduction)
target_include_directories(boost_histogram
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(boost_histogram
    INTERFACE
        Boost::assert
        Boost::config
        Boost::core
        Boost::mp11
        Boost::callable_traits
        Boost::variant
        Boost::throw_exception
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

  ###########################################################
  # Setup `check_histogram` target to build and run all tests.
  ###########################################################

  # `function` confuses FetchContent, sees empty CMAKE_CURRENT_LIST_DIR
  macro(fetch_and_include name)
      message(STATUS "Fetching ${name}")

      set(fetch_and_include_local_path "${CMAKE_BINARY_DIR}/fetch_and_include/${name}")
      if(NOT EXISTS ${fetch_and_include_local_path})
        file(DOWNLOAD
            "https://raw.githubusercontent.com/boostorg/mincmake/develop/${name}"
            "${CMAKE_BINARY_DIR}/fetch_and_include/${name}"
        )
      endif()
      include("${CMAKE_BINARY_DIR}/fetch_and_include/${name}")

  endmacro()

  fetch_and_include(cmake/boost_test.cmake)
  fetch_and_include(cmake/boost_fetch.cmake)

  boost_fetch(boostorg/assert TAG develop)
  boost_fetch(boostorg/bind TAG develop)
  boost_fetch(hdembinski/callable_traits TAG origin/fix_cmake)
  boost_fetch(boostorg/config TAG develop)
  boost_fetch(boostorg/container_hash TAG develop)
  boost_fetch(boostorg/core TAG develop)
  boost_fetch(boostorg/detail TAG develop)
  boost_fetch(boostorg/integer TAG develop)
  boost_fetch(boostorg/move TAG develop)
  boost_fetch(boostorg/mp11 TAG develop)
  boost_fetch(boostorg/mpl TAG develop)
  boost_fetch(boostorg/predef TAG develop)
  boost_fetch(boostorg/preprocessor TAG develop)
  boost_fetch(boostorg/smart_ptr TAG develop)
  boost_fetch(boostorg/static_assert TAG develop)
  boost_fetch(boostorg/throw_exception TAG develop)
  boost_fetch(boostorg/type_index TAG develop)
  boost_fetch(boostorg/type_traits TAG develop)
  boost_fetch(boostorg/utility TAG develop)
  boost_fetch(boostorg/variant TAG develop)

  ## No cmake support yet
  # boost_fetch(boostorg/accumulators TAG develop)
  # boost_fetch(boostorg/range TAG develop)
  # boost_fetch(boostorg/serialization TAG develop)
  # boost_fetch(boostorg/units TAG develop)

  enable_testing()
  add_custom_target(histogram_check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Build and then run all the tests and examples.")

  ###########################################################
  # Setup 'install' target and package config file.
  ###########################################################

  ## installings targets doesn't work currently, fails with errors like
  # Target "boost_variant" INTERFACE_INCLUDE_DIRECTORIES property contains
  # path:
  #
  # "/path/to/histogram/build/_deps/boostorg_variant-src/include"
  #
  # which is prefixed in the build directory.Target "boost_variant"
  # INTERFACE_INCLUDE_DIRECTORIES property contains path:
  #
  # "/path/to/histogram/build/_deps/boostorg_variant-src/include"
  #
  # which is prefixed in the source directory.

  # set(CONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME}-${PROJECT_VERSION})
  #
  # install(TARGETS boost_histogram
  #   boost_assert
  #   boost_bind
  #   boost_callable_traits
  #   boost_concept_check
  #   boost_config
  #   boost_container_hash
  #   boost_conversion
  #   boost_core
  #   boost_detail
  #   boost_function_types
  #   boost_fusion
  #   boost_integer
  #   boost_iterator
  #   boost_move
  #   boost_mp11
  #   boost_mpl
  #   boost_optional
  #   boost_predef
  #   boost_preprocessor
  #   boost_smart_ptr
  #   boost_static_assert
  #   boost_throw_exception
  #   boost_tuple
  #   boost_type_index
  #   boost_type_traits
  #   boost_typeof
  #   boost_utility
  #   boost_variant
  #   EXPORT ${PROJECT_NAME}Targets)
  # install(EXPORT ${PROJECT_NAME}Targets
  #   DESTINATION ${CONFIG_INSTALL_DIR}
  #   NAMESPACE Boost::
  #   FILE ${PROJECT_NAME}Config.cmake)
  #
  # install(DIRECTORY include/ DESTINATION include)

endif()

if (COMMAND boost_test)
  add_subdirectory(test)
endif()
