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

template <typename Histogram>
double compare_1d(unsigned n, int distrib)
{
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    auto h = Histogram(regular_axis(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      h.fill(r[i]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }

  return best;
}

template <typename Histogram>
double compare_3d(unsigned n, int distrib)
{
  auto r = random_array(3 * n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    auto h = Histogram(regular_axis(100, 0, 1),
                       regular_axis(100, 0, 1),
                       regular_axis(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      h.fill(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  } 

  return best;
}

template <typename Histogram>
double compare_6d(unsigned n, int distrib)
{
  auto r = random_array(6 * n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 10; ++k) {
    double x[6];

    auto h = Histogram(regular_axis(10, 0, 1),
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
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  } 

  return best;
}

int main() {
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");

    printf("1D\n");
    printf("t[boost]   = %.3f\n",
#if histogramYPE == 1
           compare_1d<histogram<1, static_storage<int>>>(12000000, itype)
#elif histogramYPE == 2
           compare_1d<histogram<1, dynamic_storage>>(12000000, itype)
#elif histogramYPE == 3
           compare_1d<histogram<Dynamic, dynamic_storage>>(12000000, itype)
#endif
    );

    printf("3D\n");
    printf("t[boost]   = %.3f\n",
#if histogramYPE == 1
           compare_3d<histogram<3, static_storage<int>>>(4000000, itype)
#elif histogramYPE == 2
           compare_3d<histogram<3, dynamic_storage>>(4000000, itype)
#elif histogramYPE == 3
           compare_3d<histogram<Dynamic, dynamic_storage>>(4000000, itype)
#endif
    );

    printf("6D\n");
    printf("t[boost]   = %.3f\n",
#if histogramYPE == 1
           compare_6d<histogram<6, static_storage<int>>>(2000000, itype)
#elif histogramYPE == 2
           compare_6d<histogram<6, dynamic_storage>>(2000000, itype)
#elif histogramYPE == 3
           compare_6d<histogram<Dynamic, dynamic_storage>>(2000000, itype)
#endif
    );
  }
}
