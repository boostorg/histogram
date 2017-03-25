// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <gsl/gsl_histogram.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <limits>
#include <random>
#include <vector>

std::vector<double> random_array(unsigned n, int type) {
  std::vector<double> result(n);
  std::default_random_engine gen(1);
  if (type) { // type == 1
    std::normal_distribution<> d(0.5, 0.3);
    for (auto &x : result)
      x = d(gen);
  } else { // type == 0
    std::uniform_real_distribution<> d(0.0, 1.0);
    for (auto &x : result)
      x = d(gen);
  }
  return result;
}

void compare_1d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  double best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    gsl_histogram *h = gsl_histogram_alloc(100);
    gsl_histogram_set_ranges_uniform(h, 0, 1);
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      gsl_histogram_increment(h, r[i]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
    gsl_histogram_free(h);
  }
  printf("gsl %.3f\n", best);
}

int main(int argc, char **argv) {
  printf("1D\n");
  printf("uniform distribution\n");
  compare_1d(12000000, 0);
  printf("normal distribution\n");
  compare_1d(12000000, 1);
}
