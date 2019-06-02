// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <memory>
#include "../test/throw_exception.hpp"
#include "../test/utility_histogram.hpp"
#include "generator.hpp"

#include <boost/assert.hpp>
struct assert_check {
  assert_check() {
    BOOST_ASSERT(false); // don't run with asserts enabled
  }
} _;

using SStore = std::vector<int>;

// make benchmark compatible with older versions of the library
#if __has_include(<boost/histogram/unlimited_storage.hpp>)
#include <boost/histogram/unlimited_storage.hpp>
using DStore = boost::histogram::unlimited_storage<>;
#else
#include <boost/histogram/adaptive_storage.hpp>
using DStore = boost::histogram::adaptive_storage<>;
#endif

using namespace boost::histogram;
using reg = axis::regular<>;

template <class Tag, class Storage, class Distribution>
static void fill_1d(benchmark::State& state) {
  auto h = make_s(Tag(), Storage(), reg(100, 0, 1));
  auto gen = generator<Distribution>();
  for (auto _ : state) benchmark::DoNotOptimize(h(gen()));
}

template <class Tag, class Storage, class Distribution>
static void fill_2d(benchmark::State& state) {
  auto h = make_s(Tag(), Storage(), reg(100, 0, 1), reg(100, 0, 1));
  auto gen = generator<Distribution>();
  for (auto _ : state) benchmark::DoNotOptimize(h(gen(), gen()));
}

template <class Tag, class Storage, class Distribution>
static void fill_3d(benchmark::State& state) {
  auto h = make_s(Tag(), Storage(), reg(100, 0, 1), reg(100, 0, 1), reg(100, 0, 1));
  auto gen = generator<Distribution>();
  for (auto _ : state) benchmark::DoNotOptimize(h(gen(), gen(), gen()));
}

template <class Tag, class Storage, class Distribution>
static void fill_6d(benchmark::State& state) {
  auto h = make_s(Tag(), Storage(), reg(10, 0, 1), reg(10, 0, 1), reg(10, 0, 1),
                  reg(10, 0, 1), reg(10, 0, 1), reg(10, 0, 1));
  auto gen = generator<Distribution>();
  for (auto _ : state)
    benchmark::DoNotOptimize(h(gen(), gen(), gen(), gen(), gen(), gen()));
}

BENCHMARK_TEMPLATE(fill_1d, static_tag, SStore, uniform);
BENCHMARK_TEMPLATE(fill_1d, static_tag, DStore, uniform);
BENCHMARK_TEMPLATE(fill_1d, dynamic_tag, SStore, uniform);
BENCHMARK_TEMPLATE(fill_1d, dynamic_tag, DStore, uniform);
BENCHMARK_TEMPLATE(fill_2d, static_tag, SStore, uniform);
BENCHMARK_TEMPLATE(fill_2d, static_tag, DStore, uniform);
BENCHMARK_TEMPLATE(fill_2d, dynamic_tag, SStore, uniform);
BENCHMARK_TEMPLATE(fill_2d, dynamic_tag, DStore, uniform);
BENCHMARK_TEMPLATE(fill_3d, static_tag, SStore, uniform);
BENCHMARK_TEMPLATE(fill_3d, static_tag, DStore, uniform);
BENCHMARK_TEMPLATE(fill_3d, dynamic_tag, SStore, uniform);
BENCHMARK_TEMPLATE(fill_3d, dynamic_tag, DStore, uniform);
BENCHMARK_TEMPLATE(fill_6d, static_tag, SStore, uniform);
BENCHMARK_TEMPLATE(fill_6d, static_tag, DStore, uniform);
BENCHMARK_TEMPLATE(fill_6d, dynamic_tag, SStore, uniform);
BENCHMARK_TEMPLATE(fill_6d, dynamic_tag, DStore, uniform);

BENCHMARK_TEMPLATE(fill_1d, static_tag, SStore, normal);
BENCHMARK_TEMPLATE(fill_1d, static_tag, DStore, normal);
BENCHMARK_TEMPLATE(fill_1d, dynamic_tag, SStore, normal);
BENCHMARK_TEMPLATE(fill_1d, dynamic_tag, DStore, normal);
BENCHMARK_TEMPLATE(fill_2d, static_tag, SStore, normal);
BENCHMARK_TEMPLATE(fill_2d, static_tag, DStore, normal);
BENCHMARK_TEMPLATE(fill_2d, dynamic_tag, SStore, normal);
BENCHMARK_TEMPLATE(fill_2d, dynamic_tag, DStore, normal);
BENCHMARK_TEMPLATE(fill_3d, static_tag, SStore, normal);
BENCHMARK_TEMPLATE(fill_3d, static_tag, DStore, normal);
BENCHMARK_TEMPLATE(fill_3d, dynamic_tag, SStore, normal);
BENCHMARK_TEMPLATE(fill_3d, dynamic_tag, DStore, normal);
BENCHMARK_TEMPLATE(fill_6d, static_tag, SStore, normal);
BENCHMARK_TEMPLATE(fill_6d, static_tag, DStore, normal);
BENCHMARK_TEMPLATE(fill_6d, dynamic_tag, SStore, normal);
BENCHMARK_TEMPLATE(fill_6d, dynamic_tag, DStore, normal);
