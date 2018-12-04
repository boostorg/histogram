// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>
#include <benchmark/benchmark.h>
#include <vector>

struct static_tag {};
struct semi_dynamic_tag {};
struct full_dynamic_tag {};

auto make_histogram(static_tag) {
  using namespace boost::histogram;
  return make_histogram_with(
    std::vector<unsigned>(),
    axis::integer<>(0, 10),
    axis::integer<>(0, 10),
    axis::integer<>(0, 10)
  );
}

auto make_histogram(semi_dynamic_tag) {
  using namespace boost::histogram;
  std::vector<axis::integer<>> axes = {
    axis::integer<>(0, 10),
    axis::integer<>(0, 10),
    axis::integer<>(0, 10)
  };
  return make_histogram_with(
    std::vector<unsigned>(),
    axes
  );
}

auto make_histogram(full_dynamic_tag) {
  using namespace boost::histogram;
  std::vector<axis::variant<axis::integer<>>> axes = {
    axis::integer<>(0, 10),
    axis::integer<>(0, 10),
    axis::integer<>(0, 10)
  };
  return make_histogram_with(
    std::vector<unsigned>(),
    axes
  );
}

template <class Tag>
static void NaiveForLoop(benchmark::State& state) {
  auto h = make_histogram(Tag());
  for (auto _ : state) {
    for (int i = 0; i < h.axis(0).size(); ++i)
      for (int j = 0; j < h.axis(1).size(); ++j)
        for (int k = 0; k < h.axis(2).size(); ++k)
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag>
static void LessNaiveForLoop(benchmark::State& state) {
  using namespace boost::histogram::literals;
  auto h = make_histogram(Tag());
  for (auto _ : state) {
    for (auto i : h.axis(0_c))
      for (auto j : h.axis(1_c))
        for (auto k : h.axis(2_c))
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag>
static void InsiderForLoop(benchmark::State& state) {
  using namespace boost::histogram::literals;
  auto h = make_histogram(Tag());
  for (auto _ : state) {
    for (int k = 0, nk = h.axis(2_c).size(); k < nk; ++k)
      for (int j = 0, nj = h.axis(1_c).size(); j < nj; ++j)
        for (int i = 0, ni = h.axis(0_c).size(); i < ni; ++i)
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag, bool include_all>
static void IndexedLoop(benchmark::State& state) {
  auto h = make_histogram(Tag());
  for (auto _ : state) {
    for (auto x : boost::histogram::indexed(h, include_all)) {
      benchmark::DoNotOptimize(*x);
      benchmark::DoNotOptimize(x[0]);
      benchmark::DoNotOptimize(x[1]);
      benchmark::DoNotOptimize(x[2]);
    }
  }
}

BENCHMARK_TEMPLATE(NaiveForLoop, static_tag);
BENCHMARK_TEMPLATE(NaiveForLoop, semi_dynamic_tag);
BENCHMARK_TEMPLATE(NaiveForLoop, full_dynamic_tag);
BENCHMARK_TEMPLATE(LessNaiveForLoop, static_tag);
BENCHMARK_TEMPLATE(LessNaiveForLoop, semi_dynamic_tag);
BENCHMARK_TEMPLATE(LessNaiveForLoop, full_dynamic_tag);
BENCHMARK_TEMPLATE(InsiderForLoop, static_tag);
BENCHMARK_TEMPLATE(InsiderForLoop, semi_dynamic_tag);
BENCHMARK_TEMPLATE(InsiderForLoop, full_dynamic_tag);
BENCHMARK_TEMPLATE(IndexedLoop, static_tag, false);
BENCHMARK_TEMPLATE(IndexedLoop, semi_dynamic_tag, false);
BENCHMARK_TEMPLATE(IndexedLoop, full_dynamic_tag, false);
BENCHMARK_TEMPLATE(IndexedLoop, static_tag, true);
BENCHMARK_TEMPLATE(IndexedLoop, semi_dynamic_tag, true);
BENCHMARK_TEMPLATE(IndexedLoop, full_dynamic_tag, true);
