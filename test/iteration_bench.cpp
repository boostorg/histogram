// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/histogram.hpp>
#include <vector>

struct static_tag {};
struct semi_dynamic_tag {};
struct full_dynamic_tag {};

using namespace boost::histogram;

auto make_histogram(static_tag, unsigned n) {
  return make_histogram_with(std::vector<unsigned>(), axis::integer<>(0, n),
                             axis::integer<>(0, n), axis::integer<>(0, n));
}

auto make_histogram(semi_dynamic_tag, unsigned n) {
  std::vector<axis::integer<>> axes = {axis::integer<>(0, n), axis::integer<>(0, n),
                                       axis::integer<>(0, n)};
  return make_histogram_with(std::vector<unsigned>(), axes);
}

auto make_histogram(full_dynamic_tag, unsigned n) {
  std::vector<axis::variant<axis::integer<>>> axes = {
      axis::integer<>(0, n), axis::integer<>(0, n), axis::integer<>(0, n)};
  return make_histogram_with(std::vector<unsigned>(), axes);
}

template <class Tag, coverage cov>
static void NaiveForLoop(benchmark::State& state) {
  auto h = make_histogram(Tag(), state.range(0));
  const int d = cov == coverage::all;
  for (auto _ : state) {
    for (int i = -d; i < h.axis(0).size() + d; ++i)
      for (int j = -d; j < h.axis(1).size() + d; ++j)
        for (int k = -d; k < h.axis(2).size() + d; ++k)
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag, coverage cov>
static void InsiderForLoop(benchmark::State& state) {
  using namespace boost::histogram::literals;
  auto h = make_histogram(Tag(), state.range(0));
  const int d = cov == coverage::all;
  for (auto _ : state) {
    for (int k = -d, nk = h.axis(2_c).size() + d; k < nk; ++k)
      for (int j = -d, nj = h.axis(1_c).size() + d; j < nj; ++j)
        for (int i = -d, ni = h.axis(0_c).size() + d; i < ni; ++i)
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag, coverage cov>
static void IndexedLoop(benchmark::State& state) {
  auto h = make_histogram(Tag(), state.range(0));
  for (auto _ : state) {
    for (auto x : boost::histogram::indexed(h, cov)) {
      benchmark::DoNotOptimize(*x);
      benchmark::DoNotOptimize(x.index(0));
      benchmark::DoNotOptimize(x.index(1));
      benchmark::DoNotOptimize(x.index(2));
    }
  }
}

BENCHMARK_TEMPLATE(NaiveForLoop, static_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, static_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, static_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, static_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, static_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, static_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, semi_dynamic_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, semi_dynamic_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, semi_dynamic_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, semi_dynamic_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, semi_dynamic_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, semi_dynamic_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, full_dynamic_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, full_dynamic_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, full_dynamic_tag, coverage::inner)
    ->RangeMultiplier(2)
    ->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, full_dynamic_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, full_dynamic_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, full_dynamic_tag, coverage::all)
    ->RangeMultiplier(2)
    ->Range(4, 128);
