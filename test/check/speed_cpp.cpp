// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/histogram.hpp>
#include <boost/histogram/axis.hpp>

#include <random>
#include <algorithm>
#include <limits>
#include <vector>
#include <ctime>
#include <cstdio>

using namespace boost::histogram;

std::vector<double> random_array(unsigned n, int type) {
  std::vector<double> result(n);
  std::default_random_engine gen(1);
  if (type) { // type == 1
    std::normal_distribution<> d(0.5, 0.3);
    for (auto& x : result)
      x = d(gen);
  }
  else { // type == 0
    std::uniform_real_distribution<> d(0.0, 1.0);
    for (auto& x: result)
      x = d(gen);
  }
  return result;
}

void compare_1d(unsigned n, int distrib)
{
  auto r = random_array(n, distrib);

  auto best_boost = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    auto h = histogram(regular_axis(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      h.fill(r[i]);
    t = clock() - t;
    best_boost = std::min(best_boost, double(t) / CLOCKS_PER_SEC);
  }

  printf("1D\n");
  printf("t[boost] = %.3f\n", best_boost);
}

void compare_3d(unsigned n, int distrib)
{
  auto r = random_array(3 * n, distrib);

  auto best_boost = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    auto h = histogram(regular_axis(100, 0, 1),
                       regular_axis(100, 0, 1),
                       regular_axis(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      h.fill(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    t = clock() - t;
    best_boost = std::min(best_boost, double(t) / CLOCKS_PER_SEC);
  } 

  printf("3D\n");
  printf("t[boost] = %.3f\n", best_boost);
}

void compare_6d(unsigned n, int distrib)
{
  auto r = random_array(6 * n, distrib);

  auto best_boost = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    double x[6];

    auto h = histogram(regular_axis(10, 0, 1),
                       regular_axis(10, 0, 1),
                       regular_axis(10, 0, 1),
                       regular_axis(10, 0, 1),
                       regular_axis(10, 0, 1),
                       regular_axis(10, 0, 1));

    auto t = clock();
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned k = 0; k < 6; ++k)
        x[k] = r[6 * i + k];      
      h.fill(x[0], x[1], x[2], x[3], x[4], x[5]);
    }
    t = clock() - t;
    best_boost = std::min(best_boost, double(t) / CLOCKS_PER_SEC);
  } 

  printf("6D\n");
  printf("t[boost] = %.3f\n", best_boost);
}

int main() {
  printf("uniform distribution\n");
  compare_1d(12000000, 0);
  compare_3d(4000000, 0);
  compare_6d(2000000, 0);
  printf("normal distribution\n");
  compare_1d(12000000, 1);
  compare_3d(4000000, 1);
  compare_6d(2000000, 1);
}
