// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>
#include <random>
#include <algorithm>
#include <limits>
#include <vector>
#include <ctime>
#include <cstdio>

using namespace boost::histogram;
namespace mpl = boost::mpl;

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
  for (unsigned k = 0; k < 50; ++k) {
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
  for (unsigned k = 0; k < 50; ++k) {
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
  for (unsigned k = 0; k < 50; ++k) {
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
  printf("1D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("t[hs_ss] %.3f\n",
      compare_1d<
        static_histogram<
          mpl::vector<regular_axis>,
          container_storage<std::vector<int>>
        >
      >(12000000, itype)
    );
    printf("t[hs_sd] %.3f\n",
      compare_1d<
        static_histogram<
          mpl::vector<regular_axis>,
          adaptive_storage<>
        >
      >(12000000, itype)
    );
    printf("t[hd_ss] %.3f\n",
      compare_1d<
        dynamic_histogram<
          default_axes,
          container_storage<std::vector<int>>
        >
      >(12000000, itype)
    );
    printf("t[hd_sd] %.3f\n",
      compare_1d<
        dynamic_histogram<
          default_axes,
          adaptive_storage<>
        >
      >(12000000, itype)
    );
  }

  printf("3D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("t[hs_ss] %.3f\n",
      compare_3d<
        static_histogram<
          mpl::vector<regular_axis, regular_axis, regular_axis>,
          container_storage<std::vector<int>>
        >
      >(4000000, itype)
    );
    printf("t[hs_sd] %.3f\n",
      compare_3d<
        static_histogram<
          mpl::vector<regular_axis, regular_axis, regular_axis>,
          adaptive_storage<>
        >
      >(4000000, itype)
    );
    printf("t[hd_ss] %.3f\n",
      compare_3d<
        dynamic_histogram<
          default_axes,
          container_storage<std::vector<int>>
        >
      >(4000000, itype)
    );
    printf("t[hd_sd] %.3f\n",
      compare_3d<
        dynamic_histogram<
          default_axes,
          adaptive_storage<>
        >
      >(4000000, itype)
    );
  }

  printf("6D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("t[hs_ss] %.3f\n",
      compare_6d<
        static_histogram<
          mpl::vector<regular_axis, regular_axis, regular_axis,
                      regular_axis, regular_axis, regular_axis>,
          container_storage<std::vector<int>>
        >
      >(2000000, itype)
    );
    printf("t[hs_sd] %.3f\n",
      compare_6d<
        static_histogram<
          mpl::vector<regular_axis, regular_axis, regular_axis,
                      regular_axis, regular_axis, regular_axis>,
          adaptive_storage<>
        >
      >(2000000, itype)
    );
    printf("t[hd_ss] %.3f\n",
      compare_6d<
        dynamic_histogram<
          default_axes,
          container_storage<std::vector<int>>
        >
      >(2000000, itype)
    );
    printf("t[hd_sd] %.3f\n",
      compare_6d<
        dynamic_histogram<
          default_axes,
          adaptive_storage<>
        >
      >(2000000, itype)
    );
  }
}
