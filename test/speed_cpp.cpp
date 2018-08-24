// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <boost/histogram.hpp>
#include <boost/mp11.hpp>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <random>
#include "utility.hpp"

using namespace boost::histogram;
using boost::mp11::mp_list;

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

template <class T>
void ignore(const T&) {}

double baseline(unsigned n) {
  auto r = random_array(n, 0);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 20; ++k) {
    auto t = clock();
    for (unsigned i = 0; i < n; ++i) {
      volatile auto x = r[i];
      ignore(x);
    }
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }
  return best;
}

template <typename Tag, typename Storage>
double compare_1d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 20; ++k) {
    auto h = make_s(Tag(), Storage(), axis::regular<>(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n; ++i) h(r[i]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }

  return best;
}

template <typename Tag, typename Storage>
double compare_2d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 20; ++k) {
    auto h =
        make_s(Tag(), Storage(), axis::regular<>(100, 0, 1), axis::regular<>(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n / 2; ++i) h(r[2 * i], r[2 * i + 1]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }

  return best;
}

template <typename Tag, typename Storage>
double compare_3d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 20; ++k) {
    auto h = make_s(Tag(), Storage(), axis::regular<>(100, 0, 1),
                    axis::regular<>(100, 0, 1), axis::regular<>(100, 0, 1));
    auto t = clock();
    for (unsigned i = 0; i < n / 3; ++i) h(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }

  return best;
}

template <typename Tag, typename Storage>
double compare_6d(unsigned n, int distrib) {
  auto r = random_array(n, distrib);

  auto best = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 20; ++k) {
    auto h =
        make_s(Tag(), Storage(), axis::regular<>(10, 0, 1), axis::regular<>(10, 0, 1),
               axis::regular<>(10, 0, 1), axis::regular<>(10, 0, 1),
               axis::regular<>(10, 0, 1), axis::regular<>(10, 0, 1));

    auto t = clock();
    for (unsigned i = 0; i < n / 6; ++i) {
      h(r[6 * i], r[6 * i + 1], r[6 * i + 2], r[6 * i + 3], r[6 * i + 4], r[6 * i + 5]);
    }
    t = clock() - t;
    best = std::min(best, double(t) / CLOCKS_PER_SEC);
  }

  return best;
}

int main() {
  const unsigned nfill = 6000000;

  printf("baseline %.3f\n", baseline(nfill));

  printf("1D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("hs_ss %.3f\n", compare_1d<static_tag, array_storage<int>>(nfill, itype));
    printf("hs_sd %.3f\n", compare_1d<static_tag, adaptive_storage<>>(nfill, itype));
    printf("hd_ss %.3f\n", compare_1d<dynamic_tag, array_storage<int>>(nfill, itype));
    printf("hd_sd %.3f\n", compare_1d<dynamic_tag, adaptive_storage<>>(nfill, itype));
  }

  printf("2D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("hs_ss %.3f\n", compare_2d<static_tag, array_storage<int>>(nfill, itype));
    printf("hs_sd %.3f\n", compare_2d<static_tag, adaptive_storage<>>(nfill, itype));
    printf("hd_ss %.3f\n", compare_2d<dynamic_tag, array_storage<int>>(nfill, itype));
    printf("hd_sd %.3f\n", compare_2d<dynamic_tag, adaptive_storage<>>(nfill, itype));
  }

  printf("3D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("hs_ss %.3f\n", compare_3d<static_tag, array_storage<int>>(nfill, itype));
    printf("hs_sd %.3f\n", compare_3d<static_tag, adaptive_storage<>>(nfill, itype));
    printf("hd_ss %.3f\n", compare_3d<dynamic_tag, array_storage<int>>(nfill, itype));
    printf("hd_sd %.3f\n", compare_3d<dynamic_tag, adaptive_storage<>>(nfill, itype));
  }

  printf("6D\n");
  for (int itype = 0; itype < 2; ++itype) {
    if (itype == 0)
      printf("uniform distribution\n");
    else
      printf("normal distribution\n");
    printf("hs_ss %.3f\n", compare_6d<static_tag, array_storage<int>>(nfill, itype));
    printf("hs_sd %.3f\n", compare_6d<static_tag, adaptive_storage<>>(nfill, itype));
    printf("hd_ss %.3f\n", compare_6d<dynamic_tag, array_storage<int>>(nfill, itype));
    printf("hd_sd %.3f\n", compare_6d<dynamic_tag, adaptive_storage<>>(nfill, itype));
  }
}
