// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <boost/histogram.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <cstdio>
#include <ctime>
#include <limits>
#include <random>
#include <vector>

using namespace boost::histogram;
namespace mpl = boost::mpl;

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

template <typename Histogram> double compare_1d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    auto h = Histogram(axis::regular<>(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      h.fill(r[i]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }

  return best;
}

template <typename Histogram> double compare_2d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    auto h = Histogram(axis::regular<>(100, 0, 1), axis::regular<>(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n/2; ++i)
      h.fill(r[2 * i], r[2 * i + 1]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }

  return best;
}

template <typename Histogram> double compare_3d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    auto h = Histogram(axis::regular<>(100, 0, 1), axis::regular<>(100, 0, 1),
                       axis::regular<>(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n/3; ++i)
      h.fill(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }

  return best;
}

template <typename Histogram> double compare_6d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    auto h = Histogram(axis::regular<>(10, 0, 1), axis::regular<>(10, 0, 1),
                       axis::regular<>(10, 0, 1), axis::regular<>(10, 0, 1),
                       axis::regular<>(10, 0, 1), axis::regular<>(10, 0, 1));

    auto t = clock();
    for (unsigned i = 0; i < n/6; ++i) {
      h.fill(r[6 * i], r[6 * i + 1], r[6 * i + 2],
             r[6 * i + 3], r[6 * i + 4], r[6 * i + 5]);
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
    printf("hs_ss %.3f\n",
           compare_1d<static_histogram<mpl::vector<axis::regular<>>,
                                array_storage<int>>>(
               6000000, itype));
    printf("hs_sd %.3f\n",
           compare_1d<static_histogram<mpl::vector<axis::regular<>>,
                                adaptive_storage>>(6000000, itype));
    printf("hd_ss %.3f\n",
           compare_1d<dynamic_histogram<axis::builtins,
                                array_storage<int>>>(
               6000000, itype));
    printf("hd_sd %.3f\n",
           compare_1d<dynamic_histogram<axis::builtins, adaptive_storage>>(
               6000000, itype));
  }

  printf("2D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("hs_ss %.3f\n",
           compare_2d<static_histogram<
               mpl::vector<axis::regular<>, axis::regular<>>,
               array_storage<int>>>(6000000, itype));
    printf("hs_sd %.3f\n",
           compare_2d<static_histogram<
               mpl::vector<axis::regular<>, axis::regular<>>,
               adaptive_storage>>(6000000, itype));
    printf("hd_ss %.3f\n",
           compare_2d<dynamic_histogram<axis::builtins,
                      array_storage<int>>>(
               6000000, itype));
    printf("hd_sd %.3f\n",
           compare_2d<dynamic_histogram<axis::builtins, adaptive_storage>>(
               6000000, itype));
  }

  printf("3D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("hs_ss %.3f\n",
           compare_3d<static_histogram<
               mpl::vector<axis::regular<>, axis::regular<>, axis::regular<>>,
               array_storage<int>>>(6000000, itype));
    printf("hs_sd %.3f\n",
           compare_3d<static_histogram<
               mpl::vector<axis::regular<>, axis::regular<>, axis::regular<>>,
               adaptive_storage>>(6000000, itype));
    printf("hd_ss %.3f\n",
           compare_3d<dynamic_histogram<axis::builtins,
                                array_storage<int>>>(
               6000000, itype));
    printf("hd_sd %.3f\n",
           compare_3d<dynamic_histogram<axis::builtins, adaptive_storage>>(
               6000000, itype));
  }

  printf("6D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("hs_ss %.3f\n",
           compare_6d<static_histogram<
               mpl::vector<axis::regular<>, axis::regular<>, axis::regular<>,
                           axis::regular<>, axis::regular<>, axis::regular<>>,
               array_storage<int>>>(6000000, itype));
    printf("hs_sd %.3f\n",
           compare_6d<static_histogram<
               mpl::vector<axis::regular<>, axis::regular<>, axis::regular<>,
                           axis::regular<>, axis::regular<>, axis::regular<>>,
               adaptive_storage>>(6000000, itype));
    printf("hd_ss %.3f\n",
           compare_6d<dynamic_histogram<axis::builtins,
                                array_storage<int>>>(
               6000000, itype));
    printf("hd_sd %.3f\n",
           compare_6d<dynamic_histogram<axis::builtins, adaptive_storage>>(
               6000000, itype));
  }
}
