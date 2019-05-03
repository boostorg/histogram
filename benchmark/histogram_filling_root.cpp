// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <benchmark/benchmark.h>
#include <random>

#include <TH1I.h>
#include <TH2I.h>
#include <TH3I.h>
#include <THn.h>

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
  TH1I h("", "", 100, 0, 1);
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) h.Fill(dis(gen));
}

template <class Distribution>
static void fill_2d(benchmark::State& state) {
  TH2I h("", "", 100, 0, 1, 100, 0, 1);
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) h.Fill(dis(gen), dis(gen));
}

template <class Distribution>
static void fill_3d(benchmark::State& state) {
  TH3I h("", "", 100, 0, 1, 100, 0, 1, 100, 0, 1);
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) { h.Fill(dis(gen), dis(gen), dis(gen)); }
}

template <class Distribution>
static void fill_6d(benchmark::State& state) {
  int bin[] = {10, 10, 10, 10, 10, 10};
  double min[] = {0, 0, 0, 0, 0, 0};
  double max[] = {1, 1, 1, 1, 1, 1};
  THnI h("", "", 6, bin, min, max);
  std::default_random_engine gen(1);
  Distribution dis = init<Distribution>();
  for (auto _ : state) {
    const double buf[6] = {dis(gen), dis(gen), dis(gen), dis(gen), dis(gen), dis(gen)};
    h.Fill(buf);
  }
}

BENCHMARK_TEMPLATE(fill_1d, uniform);
BENCHMARK_TEMPLATE(fill_2d, uniform);
BENCHMARK_TEMPLATE(fill_3d, uniform);
BENCHMARK_TEMPLATE(fill_6d, uniform);

BENCHMARK_TEMPLATE(fill_1d, normal);
BENCHMARK_TEMPLATE(fill_2d, normal);
BENCHMARK_TEMPLATE(fill_3d, normal);
BENCHMARK_TEMPLATE(fill_6d, normal);
