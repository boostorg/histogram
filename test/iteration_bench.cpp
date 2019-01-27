// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/histogram.hpp>
#include <boost/mp11.hpp>
#include <vector>

using namespace boost::histogram;

struct tuple {};
struct vector {};
struct vector_of_variant {};

using d1 = boost::mp11::mp_int<1>;
using d2 = boost::mp11::mp_int<2>;
using d3 = boost::mp11::mp_int<3>;

auto make_histogram(tuple, d1, unsigned n) {
  return make_histogram_with(std::vector<unsigned>(), axis::integer<>(0, n));
}
auto make_histogram(tuple, d2, unsigned n) {
  return make_histogram_with(std::vector<unsigned>(), axis::integer<>(0, n),
                             axis::integer<>(0, n));
}

auto make_histogram(tuple, d3, unsigned n) {
  return make_histogram_with(std::vector<unsigned>(), axis::integer<>(0, n),
                             axis::integer<>(0, n), axis::integer<>(0, n));
}

template <int Dim>
auto make_histogram(vector, boost::mp11::mp_int<Dim>, unsigned n) {
  std::vector<axis::integer<>> axes;
  for (unsigned d = 0; d < Dim; ++d) axes.emplace_back(axis::integer<>(0, n));
  return make_histogram_with(std::vector<unsigned>(), std::move(axes));
}

template <int Dim>
auto make_histogram(vector_of_variant, boost::mp11::mp_int<Dim>, unsigned n) {
  std::vector<axis::variant<axis::integer<>>> axes;
  for (unsigned d = 0; d < Dim; ++d) axes.emplace_back(axis::integer<>(0, n));
  return make_histogram_with(std::vector<unsigned>(), std::move(axes));
}

template <class Tag>
static void Naive(benchmark::State& state, Tag, d1, coverage cov) {
  auto h = make_histogram(Tag(), d1(), state.range(0));
  const int d = cov == coverage::all;
  for (auto _ : state) {
    for (int i = -d; i < h.axis(0).size() + d; ++i) benchmark::DoNotOptimize(h.at(i));
  }
}

template <class Tag>
static void Naive(benchmark::State& state, Tag, d2, coverage cov) {
  auto h = make_histogram(Tag(), d2(), state.range(0));
  const int d = cov == coverage::all;
  for (auto _ : state) {
    for (int i = -d; i < h.axis(0).size() + d; ++i)
      for (int j = -d; j < h.axis(1).size() + d; ++j)
        benchmark::DoNotOptimize(h.at(i, j));
  }
}

template <class Tag>
static void Naive(benchmark::State& state, Tag, d3, coverage cov) {
  auto h = make_histogram(Tag(), d3(), state.range(0));
  const int d = cov == coverage::all;
  for (auto _ : state) {
    for (int i = -d; i < h.axis(0).size() + d; ++i)
      for (int j = -d; j < h.axis(1).size() + d; ++j)
        for (int k = -d; k < h.axis(2).size() + d; ++k)
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag>
static void Insider(benchmark::State& state, Tag, d2, coverage cov) {
  using namespace literals;
  auto h = make_histogram(Tag(), d2(), state.range(0));
  const int d = cov == coverage::all;
  for (auto _ : state) {
    for (int j = -d, nj = h.axis(1_c).size() + d; j < nj; ++j)
      for (int i = -d, ni = h.axis(0_c).size() + d; i < ni; ++i)
        benchmark::DoNotOptimize(h.at(i, j));
  }
}

template <class Tag>
static void Insider(benchmark::State& state, Tag, d3, coverage cov) {
  using namespace literals;
  auto h = make_histogram(Tag(), d3(), state.range(0));
  const int d = cov == coverage::all;
  for (auto _ : state) {
    for (int k = -d, nk = h.axis(2_c).size() + d; k < nk; ++k)
      for (int j = -d, nj = h.axis(1_c).size() + d; j < nj; ++j)
        for (int i = -d, ni = h.axis(0_c).size() + d; i < ni; ++i)
          benchmark::DoNotOptimize(h.at(i, j, k));
  }
}

template <class Tag>
static void Indexed(benchmark::State& state, Tag, d1, coverage cov) {
  auto h = make_histogram(Tag(), d1(), state.range(0));
  for (auto _ : state) {
    for (auto x : indexed(h, cov)) {
      benchmark::DoNotOptimize(*x);
      benchmark::DoNotOptimize(x.index());
    }
  }
}

template <class Tag>
static void Indexed(benchmark::State& state, Tag, d2, coverage cov) {
  auto h = make_histogram(Tag(), d2(), state.range(0));
  for (auto _ : state) {
    for (auto x : indexed(h, cov)) {
      benchmark::DoNotOptimize(*x);
      benchmark::DoNotOptimize(x.index(0));
      benchmark::DoNotOptimize(x.index(1));
    }
  }
}

template <class Tag>
static void Indexed(benchmark::State& state, Tag, d3, coverage cov) {
  auto h = make_histogram(Tag(), d3(), state.range(0));
  for (auto _ : state) {
    for (auto x : indexed(h, cov)) {
      benchmark::DoNotOptimize(*x);
      benchmark::DoNotOptimize(x.index(0));
      benchmark::DoNotOptimize(x.index(1));
      benchmark::DoNotOptimize(x.index(2));
    }
  }
}

#define BENCH(Type, Tag, Dim, Cov)                                      \
  BENCHMARK_CAPTURE(Type, (Tag, Dim, Cov), Tag{}, Dim{}, coverage::Cov) \
      ->RangeMultiplier(2)                                              \
      ->Range(4, 128)

BENCH(Naive, tuple, d3, inner);
BENCH(Insider, tuple, d3, inner);
BENCH(Indexed, tuple, d3, inner);

BENCH(Naive, vector, d3, inner);
// BENCH(Insider, vector, d3, inner);
BENCH(Indexed, vector, d3, inner);

BENCH(Naive, vector_of_variant, d3, inner);
// BENCH(Insider, vector_of_variant, d3, inner);
BENCH(Indexed, vector_of_variant, d3, inner);

BENCH(Naive, tuple, d2, inner);
// BENCH(Insider, tuple, d2, inner);
BENCH(Indexed, tuple, d2, inner);

BENCH(Naive, vector, d2, inner);
// BENCH(Insider, vector, d2, inner);
BENCH(Indexed, vector, d2, inner);

BENCH(Naive, vector_of_variant, d2, inner);
// BENCH(Insider, vector_of_variant, d2, inner);
BENCH(Indexed, vector_of_variant, d2, inner);

BENCH(Naive, tuple, d1, inner);
BENCH(Indexed, tuple, d1, inner);

BENCH(Naive, vector, d1, inner);
BENCH(Indexed, vector, d1, inner);

BENCH(Naive, vector_of_variant, d1, inner);
BENCH(Indexed, vector_of_variant, d1, inner);
