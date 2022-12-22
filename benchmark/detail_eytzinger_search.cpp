// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <boost/histogram/axis/variable.hpp>
#include <numeric>
#include "../test/throw_exception.hpp"
#include "generator.hpp"
#include <algorithm>
#include <vector>
#include <iostream>
#include <boost/core/bit.hpp>
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

template<typename T=double>
struct eytzinger_search {

  int ffs(size_t v) const noexcept
  {
      if(v==0) return 0;
  #if HAVE_BOOST_MULTIPRECISION
      return boost::multiprecision::lsb(v)+1;
  #else
      // we prefer boost::core since it is a dependency already
      return boost::core::countr_zero(v)+1;
  #endif
  }

  eytzinger_search(const std::vector<T>& a) :
      b_(a.size() + 1), idx_(a.size() + 1)
  {
      init(a);
      idx_[0] = a.size() - 1;
  }

  int index(T const& x) const {
      size_t k = 1;
      while (k < b_.size())
          k = 2 * k + (b_[k] < x);
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
  std::vector<T, boost::alignment::aligned_allocator<T, 64>> b_;
  std::vector<int, boost::alignment::aligned_allocator<int, 64>> idx_;
#else
  std::vector<T> b_;
  std::vector<int> idx_;
#endif
};

using namespace boost::histogram;

template <class Distribution>
static void variable(benchmark::State& state) {
  std::vector<double> v;
  for (double x = 0; x <= state.range(0); ++x) { v.push_back(x / state.range(0)); }
  auto a = axis::variable<>(v);
  generator<Distribution> gen;  
  for (auto _ : state) benchmark::DoNotOptimize(a.index(gen()));
}

template <class Distribution>
static void eytzinger(benchmark::State& state) {
  std::vector<double> v;
  for (double x = 0; x <= state.range(0); ++x) { v.push_back(x / state.range(0)); }
  auto a = eytzinger_search<double>(v);
  generator<Distribution> gen;    

  auto aref = axis::variable<>(v);
  for (int i = 0; i < 10000; ++i) {
    const double x = gen();
    if (a.index(x) != aref.index(x)) {
      std::cout << "x = " << x << " " 
                << a.index(x) << " " 
                << aref.index(x) << std::endl;
      std::abort();
    }
  }

  for (auto _ : state) benchmark::DoNotOptimize(a.index(gen()));
}

BENCHMARK_TEMPLATE(variable, uniform)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK_TEMPLATE(variable, normal)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK_TEMPLATE(variable, chi2)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK_TEMPLATE(variable, expon)->RangeMultiplier(10)->Range(10, 10000);

BENCHMARK_TEMPLATE(eytzinger, uniform)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK_TEMPLATE(eytzinger, normal)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK_TEMPLATE(eytzinger, chi2)->RangeMultiplier(10)->Range(10, 10000);
BENCHMARK_TEMPLATE(eytzinger, expon)->RangeMultiplier(10)->Range(10, 10000);
