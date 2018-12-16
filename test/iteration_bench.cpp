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

auto make_histogram(static_tag, unsigned n) {
  using namespace boost::histogram;
  return make_histogram_with(std::vector<unsigned>(), axis::integer<>(0, n),
                             axis::integer<>(0, n), axis::integer<>(0, n));
}

auto make_histogram(semi_dynamic_tag, unsigned n) {
  using namespace boost::histogram;
  std::vector<axis::integer<>> axes = {axis::integer<>(0, n), axis::integer<>(0, n),
                                       axis::integer<>(0, n)};
  return make_histogram_with(std::vector<unsigned>(), axes);
}

auto make_histogram(full_dynamic_tag, unsigned n) {
  using namespace boost::histogram;
  std::vector<axis::variant<axis::integer<>>> axes = {
      axis::integer<>(0, n), axis::integer<>(0, n), axis::integer<>(0, n)};
  return make_histogram_with(std::vector<unsigned>(), axes);
}

template <class Tag, bool include_extra_bins>
static void NaiveForLoop(benchmark::State& state) {
  auto h = make_histogram(Tag(), state.range(0));
  for (auto _ : state) {
    for (int i = -include_extra_bins; i < h.axis(0).size() + include_extra_bins; ++i)
      for (int j = -include_extra_bins; j < h.axis(1).size() + include_extra_bins; ++j)
        for (int k = -include_extra_bins; k < h.axis(2).size() + include_extra_bins; ++k)
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag, bool include_extra_bins>
static void InsiderForLoop(benchmark::State& state) {
  using namespace boost::histogram::literals;
  auto h = make_histogram(Tag(), state.range(0));
  for (auto _ : state) {
    for (int k = -include_extra_bins, nk = h.axis(2_c).size() + include_extra_bins;
         k < nk; ++k)
      for (int j = -include_extra_bins, nj = h.axis(1_c).size() + include_extra_bins;
           j < nj; ++j)
        for (int i = -include_extra_bins, ni = h.axis(0_c).size() + include_extra_bins;
             i < ni; ++i)
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag, bool include_extra_bins>
static void IndexedLoop(benchmark::State& state) {
  auto h = make_histogram(Tag(), state.range(0));
  for (auto _ : state) {
    for (auto x : boost::histogram::indexed(h, include_extra_bins)) {
      benchmark::DoNotOptimize(*x);
      benchmark::DoNotOptimize(x[0]);
      benchmark::DoNotOptimize(x[1]);
      benchmark::DoNotOptimize(x[2]);
    }
  }
}

BENCHMARK_TEMPLATE(NaiveForLoop, static_tag, false)->RangeMultiplier(2)->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, static_tag, false)->RangeMultiplier(2)->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, static_tag, false)->RangeMultiplier(2)->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, static_tag, true)->RangeMultiplier(2)->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, static_tag, true)->RangeMultiplier(2)->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, static_tag, true)->RangeMultiplier(2)->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, semi_dynamic_tag, false)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, semi_dynamic_tag, false)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, semi_dynamic_tag, false)
    ->RangeMultiplier(2)
    ->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, semi_dynamic_tag, true)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, semi_dynamic_tag, true)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, semi_dynamic_tag, true)
    ->RangeMultiplier(2)
    ->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, full_dynamic_tag, false)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, full_dynamic_tag, false)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, full_dynamic_tag, false)
    ->RangeMultiplier(2)
    ->Range(4, 128);

BENCHMARK_TEMPLATE(NaiveForLoop, full_dynamic_tag, true)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(InsiderForLoop, full_dynamic_tag, true)
    ->RangeMultiplier(2)
    ->Range(4, 128);
BENCHMARK_TEMPLATE(IndexedLoop, full_dynamic_tag, true)
    ->RangeMultiplier(2)
    ->Range(4, 128);
