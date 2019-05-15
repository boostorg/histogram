// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/detail/throw_exception.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unlimited_storage.hpp>

#include <random>
#include "../test/utility_histogram.hpp"

using namespace boost::histogram;
using reg = axis::regular<>;
using uniform = std::uniform_real_distribution<>;
using normal = std::normal_distribution<>;

template <class Distribution>
Distribution init();

template <>
uniform init<uniform>() {
  return uniform{0.0, 1.0};
}

template <>
normal init<normal>() {
  return normal{0.5, 0.3};
}

template <class Tag, class Storage, class Distribution>
static void fill_1d(benchmark::State& state) {
  auto h = make_s(Tag(), Storage(), reg(100, 0, 1));
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) h(dis(gen));
}

template <class Tag, class Storage, class Distribution>
static void fill_2d(benchmark::State& state) {
  auto h = make_s(Tag(), Storage(), reg(100, 0, 1), reg(100, 0, 1));
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) { h(dis(gen), dis(gen)); }
}

template <class Tag, class Storage, class Distribution>
static void fill_3d(benchmark::State& state) {
  auto h = make_s(Tag(), Storage(), reg(100, 0, 1), reg(100, 0, 1), reg(100, 0, 1));
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) { h(dis(gen), dis(gen), dis(gen)); }
}

template <class Tag, class Storage, class Distribution>
static void fill_6d(benchmark::State& state) {
  auto h = make_s(Tag(), Storage(), reg(10, 0, 1), reg(10, 0, 1), reg(10, 0, 1),
                  reg(10, 0, 1), reg(10, 0, 1), reg(10, 0, 1));
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) { h(dis(gen), dis(gen), dis(gen), dis(gen), dis(gen), dis(gen)); }
}

using SStore = std::vector<int>;
using DStore = unlimited_storage<>;

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
