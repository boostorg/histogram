// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/mp11/algorithm.hpp>
#include <tuple>
#include <type_traits>
#include <vector>
#include "../test/throw_exception.hpp"
#include "generator.hpp"

#include <boost/assert.hpp>
struct assert_check {
  assert_check() {
    BOOST_ASSERT(false); // don't run with asserts enabled
  }
} _;

using namespace boost::histogram;
using reg = axis::regular<>;
using integ = axis::integer<>;
using var = axis::variable<>;
using vector_of_variant = std::vector<axis::variant<reg, integ, var>>;

template <class T, class U>
auto make_storage(const U& axes) {
  return std::vector<T>(detail::bincount(axes), 0);
}

template <class Axes, class Storage, class Tuple>
void fill_a(const Axes& axes, Storage& storage, const Tuple& t) {
  using namespace boost::mp11;
  std::size_t stride = 1, index = 0;
  mp_for_each<mp_iota<mp_size<Tuple>>>([&](auto i) {
    const auto& a = boost::histogram::detail::axis_get<i>(axes);
    const auto& v = std::get<i>(t);
    index += (a.index(v) + 1) * stride;
    stride *= axis::traits::extent(a);
  });
  ++storage[index];
}

template <class T, class Distribution>
static void fill_1d_a(benchmark::State& state) {
  auto axes = std::make_tuple(reg(100, 0, 1));
  generator<Distribution> gen;
  auto storage = make_storage<T>(axes);
  for (auto _ : state) fill_a(axes, storage, std::forward_as_tuple(gen()));
  state.SetItemsProcessed(state.iterations());
}

template <class T, class Distribution>
static void fill_1d_a_dyn(benchmark::State& state) {
  auto axes = vector_of_variant({reg(100, 0, 1)});
  generator<Distribution> gen;
  auto storage = make_storage<T>(axes);
  for (auto _ : state) fill_a(axes, storage, std::forward_as_tuple(gen()));
  state.SetItemsProcessed(state.iterations());
}

template <class T, class Distribution>
static void fill_2d_a(benchmark::State& state) {
  auto axes = std::make_tuple(reg(100, 0, 1), reg(100, 0, 1));
  generator<Distribution> gen;
  auto storage = make_storage<T>(axes);
  for (auto _ : state) fill_a(axes, storage, std::forward_as_tuple(gen(), gen()));
  state.SetItemsProcessed(state.iterations() * 2);
}

template <class T, class Distribution>
static void fill_2d_a_dyn(benchmark::State& state) {
  auto axes = vector_of_variant({reg(100, 0, 1), reg(100, 0, 1)});
  generator<Distribution> gen;
  auto storage = make_storage<T>(axes);
  for (auto _ : state) fill_a(axes, storage, std::forward_as_tuple(gen(), gen()));
  state.SetItemsProcessed(state.iterations() * 2);
}

BENCHMARK_TEMPLATE(fill_1d_a, double, uniform);
BENCHMARK_TEMPLATE(fill_1d_a_dyn, double, uniform);
BENCHMARK_TEMPLATE(fill_2d_a, double, uniform);
BENCHMARK_TEMPLATE(fill_2d_a_dyn, double, uniform);

BENCHMARK_TEMPLATE(fill_1d_a, double, normal);
BENCHMARK_TEMPLATE(fill_2d_a, double, normal);
