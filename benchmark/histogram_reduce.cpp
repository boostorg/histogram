// Copyright 2026 Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/histogram/algorithm/reduce.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/make_histogram.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unlimited_storage.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11/integral.hpp>
#include <random>
#include <vector>
#include "../test/throw_exception.hpp"

#include <cassert>
struct assert_check {
  assert_check() {
    assert(false); // don't run with asserts enabled
  }
} _;

using namespace boost::histogram;

struct dense {};
struct unlimited {};

template <unsigned I>
using Dim_t = boost::mp11::mp_int<I>;

auto make_storage(dense) { return dense_storage<double>(); }
auto make_storage(unlimited) { return unlimited_storage<>(); }

template <class Tag, int Dim>
auto make_histogram(Tag, boost::mp11::mp_int<Dim>, int n) {
  std::vector<axis::regular<>> axes;
  for (int d = 0; d < Dim; ++d) axes.emplace_back(n, 0.0, 1.0);
  auto h = make_histogram_with(make_storage(Tag{}), std::move(axes));
  // cell values do not affect reduce speed, they only must be non-trivial
  std::default_random_engine gen(1);
  std::uniform_real_distribution<double> dis(0, 10);
  for (auto&& v : unsafe_access::storage(h)) v = dis(gen);
  return h;
}

template <class Histogram>
void run_reduce(benchmark::State& state, const Histogram& h,
                const std::vector<algorithm::reduce_command>& opts) {
  for (auto _ : state) {
    auto r = algorithm::reduce(h, opts);
    benchmark::DoNotOptimize(r);
  }
  // report throughput in input cells (including flow bins) per second
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(h.size()));
}

template <class Tag, int Dim>
static void Rebin(benchmark::State& state, Tag, boost::mp11::mp_int<Dim> d) {
  auto h = make_histogram(Tag{}, d, static_cast<int>(state.range(0)));
  std::vector<algorithm::reduce_command> opts;
  for (int i = 0; i < Dim; ++i) opts.push_back(algorithm::rebin(i, 2));
  run_reduce(state, h, opts);
}

template <class Tag, int Dim>
static void Shrink(benchmark::State& state, Tag, boost::mp11::mp_int<Dim> d) {
  auto h = make_histogram(Tag{}, d, static_cast<int>(state.range(0)));
  std::vector<algorithm::reduce_command> opts;
  for (int i = 0; i < Dim; ++i) opts.push_back(algorithm::shrink(i, 0.2, 0.8));
  run_reduce(state, h, opts);
}

template <class Tag, int Dim>
static void ShrinkAndRebin(benchmark::State& state, Tag, boost::mp11::mp_int<Dim> d) {
  auto h = make_histogram(Tag{}, d, static_cast<int>(state.range(0)));
  std::vector<algorithm::reduce_command> opts;
  for (int i = 0; i < Dim; ++i)
    opts.push_back(algorithm::shrink_and_rebin(i, 0.2, 0.8, 2));
  run_reduce(state, h, opts);
}

#define BENCH(Type, Tag, Dim)                              \
  BENCHMARK_CAPTURE(Type, (Tag, Dim), Tag{}, Dim_t<Dim>{}) \
      ->RangeMultiplier(4)                                 \
      ->Range(4, 256)

BENCH(Rebin, dense, 1);
BENCH(Shrink, dense, 1);
BENCH(ShrinkAndRebin, dense, 1);

BENCH(Rebin, dense, 2);
BENCH(Shrink, dense, 2);
BENCH(ShrinkAndRebin, dense, 2);

BENCH(Rebin, dense, 3);
BENCH(Shrink, dense, 3);
BENCH(ShrinkAndRebin, dense, 3);

BENCH(Rebin, unlimited, 1);
BENCH(Shrink, unlimited, 1);
BENCH(ShrinkAndRebin, unlimited, 1);

BENCH(Rebin, unlimited, 2);
BENCH(Shrink, unlimited, 2);
BENCH(ShrinkAndRebin, unlimited, 2);

BENCH(Rebin, unlimited, 3);
BENCH(Shrink, unlimited, 3);
BENCH(ShrinkAndRebin, unlimited, 3);
