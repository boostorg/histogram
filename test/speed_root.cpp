// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <TH1I.h>
#include <TH2I.h>
#include <TH3I.h>
#include <THn.h>

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

  double best_root = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    TH1I hroot("", "", 100, 0, 1);
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      hroot.Fill(r[i]);
    t = clock() - t;
    best_root = std::min(best_root, double(t) / CLOCKS_PER_SEC);
  }
  printf("root %.3f\n", best_root);
}

void compare_2d(unsigned n, int distrib) {
  auto r = random_array(2 * n, distrib);

  double best_root = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    TH2I hroot("", "", 100, 0, 1, 100, 0, 1);
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      hroot.Fill(r[2 * i], r[2 * i + 1]);
    t = clock() - t;
    best_root = std::min(best_root, double(t) / CLOCKS_PER_SEC);
  }
  printf("root %.3f\n", best_root);
}

void compare_3d(unsigned n, int distrib) {
  auto r = random_array(3 * n, distrib);

  double best_root = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    TH3I hroot("", "", 100, 0, 1, 100, 0, 1, 100, 0, 1);
    auto t = clock();
    for (unsigned i = 0; i < n; ++i)
      hroot.Fill(r[3 * i], r[3 * i + 1], r[3 * i + 2]);
    t = clock() - t;
    best_root = std::min(best_root, double(t) / CLOCKS_PER_SEC);
  }
  printf("root %.3f\n", best_root);
}

void compare_6d(unsigned n, int distrib) {
  auto r = random_array(6 * n, distrib);

  double best_root = std::numeric_limits<double>::max();
  for (unsigned k = 0; k < 50; ++k) {
    double x[6];
    std::vector<int> bin(6, 10);
    std::vector<double> min(6, 0);
    std::vector<double> max(6, 1);
    THnI hroot("", "", 6, &bin.front(), &min.front(), &max.front());

    auto t = clock();
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned k = 0; k < 6; ++k)
        x[k] = r[6 * i + k];
      hroot.Fill(x);
    }
    t = clock() - t;
    best_root = std::min(best_root, double(t) / CLOCKS_PER_SEC);
  }
  printf("root %.3f\n", best_root);
}

int main(int argc, char **argv) {
  printf("1D\n");
  printf("uniform distribution\n");
  compare_1d(12000000, 0);
  printf("normal distribution\n");
  compare_1d(12000000, 1);

  printf("2D\n");
  printf("uniform distribution\n");
  compare_2d(6000000, 0);
  printf("normal distribution\n");
  compare_2d(6000000, 1);

  printf("3D\n");
  printf("uniform distribution\n");
  compare_3d(4000000, 0);
  printf("normal distribution\n");
  compare_3d(4000000, 1);

  printf("6D\n");
  printf("uniform distribution\n");
  compare_6d(2000000, 0);
  printf("normal distribution\n");
  compare_6d(2000000, 1);
}
