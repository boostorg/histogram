// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <random>

#include "throw_exception.hpp"

#include <gsl/gsl_histogram.h>
#include <gsl/gsl_histogram2d.h>

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

template <class Distribution>
static void fill_1d(benchmark::State& state) {
  gsl_histogram* h = gsl_histogram_alloc(100);
  gsl_histogram_set_ranges_uniform(h, 0, 1);
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) benchmark::DoNotOptimize(gsl_histogram_increment(h, dis(gen)));
  gsl_histogram_free(h);
}

template <class Distribution>
static void fill_2d(benchmark::State& state) {
  gsl_histogram2d* h = gsl_histogram2d_alloc(100, 100);
  gsl_histogram2d_set_ranges_uniform(h, 0, 1, 0, 1);
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state)
    benchmark::DoNotOptimize(gsl_histogram2d_increment(h, dis(gen), dis(gen)));
  gsl_histogram2d_free(h);
}

BENCHMARK_TEMPLATE(fill_1d, uniform);
BENCHMARK_TEMPLATE(fill_2d, uniform);

BENCHMARK_TEMPLATE(fill_1d, normal);
BENCHMARK_TEMPLATE(fill_2d, normal);
