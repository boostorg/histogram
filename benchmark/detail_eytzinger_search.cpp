// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <algorithm>
#include <boost/core/bit.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <boost/histogram/detail/prefetch.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#include "../test/throw_exception.hpp"
#include "generator.hpp"
#ifdef HAVE_BOOST_ALIGN
#include <boost/align/aligned_allocator.hpp>
#endif
#include <boost/core/lightweight_test.hpp>

#include <cassert>
struct assert_check {
  assert_check() {
    assert(false); // don't run with asserts enabled
  }
} _;

using namespace boost::histogram;

template <typename T = double>
struct eytzinger_search {
  int ffs(size_t v) const noexcept {
    if (v == 0) return 0;
#if HAVE_BOOST_MULTIPRECISION
    return boost::multiprecision::lsb(v) + 1;
#else
    // we prefer boost::core since it is a dependency already
    return boost::core::countr_zero(v) + 1;
#endif
  }

  eytzinger_search(const std::vector<T>& a) : b_(a.size() + 1), idx_(a.size() + 1) {
    init(a);
    idx_[0] = a.size() - 1;
  }

  int index(T const& x) const {
    size_t k = 1;
    while (k < b_.size()) {
      constexpr int block_size = detail::cacheline_length / sizeof(T);
      detail::prefetch(b_.data() + k * block_size);
      k = 2 * k + !(x < b_[k]); // double negation to handle nan correctly
    }
    k >>= ffs(~k);
    return idx_[k];
  }

  size_t init(const std::vector<T>& a, size_t i = 0, size_t k = 1) {
    if (k <= a.size()) {
      i = init(a, i, 2 * k);
      idx_[k] = i - 1;
      b_[k] = a[i++];
      i = init(a, i, 2 * k + 1);
    }
    return i;
  }

#ifdef HAVE_BOOST_ALIGN
  std::vector<T, boost::alignment::aligned_allocator<T, detail::cache_alignment>> b_;
  std::vector<int, boost::alignment::aligned_allocator<int, detail::cache_alignment>>
      idx_;
#else
  std::vector<T> b_;
  std::vector<int> idx_;
#endif
};

template <class Distribution>
static void eytzinger(benchmark::State& state) {
  std::vector<double> v;
  for (double x = 0; x <= state.range(0); ++x) { v.push_back(x / state.range(0)); }
  auto a = eytzinger_search<double>(v);
  generator<Distribution> gen;

  auto aref = axis::variable<>(v);
  std::vector<double> inputs(1000);
  std::generate(inputs.begin(), inputs.end(), gen);
  inputs[0] = std::numeric_limits<double>::quiet_NaN();
  inputs[1] = -std::numeric_limits<double>::infinity();
  inputs[2] = std::numeric_limits<double>::infinity();
  inputs[3] = v[0];
  inputs[4] = v.back();
  for (auto&& x : inputs) {
    if (a.index(x) != aref.index(x)) {
      std::cout << "x = " << x << ": new = " << a.index(x) << " ref = " << aref.index(x)
                << std::endl;
      std::abort();
    }
  }

  for (auto _ : state) benchmark::DoNotOptimize(a.index(gen()));
}

template <class Distribution>
static void variable(benchmark::State& state) {
  std::vector<double> v;
  for (double x = 0; x <= state.range(0); ++x) { v.push_back(x / state.range(0)); }
  auto a = axis::variable<>(v);
  generator<Distribution> gen;
  for (auto _ : state) benchmark::DoNotOptimize(a.index(gen()));
}

BENCHMARK_TEMPLATE(eytzinger, uniform)->RangeMultiplier(10)->Range(10, 100000);
BENCHMARK_TEMPLATE(eytzinger, normal)->RangeMultiplier(10)->Range(10, 100000);
BENCHMARK_TEMPLATE(eytzinger, chi2)->RangeMultiplier(10)->Range(10, 100000);
BENCHMARK_TEMPLATE(eytzinger, expon)->RangeMultiplier(10)->Range(10, 100000);

BENCHMARK_TEMPLATE(variable, uniform)->RangeMultiplier(10)->Range(10, 100000);
BENCHMARK_TEMPLATE(variable, normal)->RangeMultiplier(10)->Range(10, 100000);
BENCHMARK_TEMPLATE(variable, chi2)->RangeMultiplier(10)->Range(10, 100000);
BENCHMARK_TEMPLATE(variable, expon)->RangeMultiplier(10)->Range(10, 100000);
