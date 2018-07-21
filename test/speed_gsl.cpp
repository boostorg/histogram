// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <gsl/gsl_histogram.h>
#include <gsl/gsl_histogram2d.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <random>

std::unique_ptr<double[]> random_array(unsigned n, int type) {
  std::unique_ptr<double[]> r(new double[n]);
  std::default_random_engine gen(1);
  if (type) { // type == 1
    std::normal_distribution<> d(0.5, 0.3);
    for (unsigned i = 0; i < n; ++i) r[i] = d(gen);
  } else { // type == 0
    std::uniform_real_distribution<> d(0.0, 1.0);
    for (unsigned i = 0; i < n; ++i) r[i] = d(gen);
  }
  return r;
}

void compare_1d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  double best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 20; ++k) {
    gsl_histogram* h = gsl_histogram_alloc(100);
    gsl_histogram_set_ranges_uniform(h, 0, 1);
    auto t = clock();
    for (unsigned i = 0; i < n; ++i) gsl_histogram_increment(h, r[i]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
    gsl_histogram_free(h);
  }
  printf("gsl %.3f\n", best);
}

void compare_2d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  double best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 20; ++k) {
    gsl_histogram2d* h = gsl_histogram2d_alloc(100, 100);
    gsl_histogram2d_set_ranges_uniform(h, 0, 1, 0, 1);
    auto t = clock();
    for (unsigned i = 0; i < n / 2; ++i)
      gsl_histogram2d_increment(h, r[2 * i], r[2 * i + 1]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
    gsl_histogram2d_free(h);
  }
  printf("gsl %.3f\n", best);
}

int main(int argc, char** argv) {
  constexpr unsigned nfill = 6000000;
  printf("1D\n");
  printf("uniform distribution\n");
  compare_1d(nfill, 0);
  printf("normal distribution\n");
  compare_1d(nfill, 1);

  printf("2D\n");
  printf("uniform distribution\n");
  compare_2d(nfill, 0);
  printf("normal distribution\n");
  compare_2d(nfill, 1);
}
