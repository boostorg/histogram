// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// clang-format off
//[ guide_stl_algorithms

#include <boost/histogram.hpp>
#include <cassert>
#include <numeric> // partial_sum, inner_product
#include <cmath>   // sqrt
#include <iostream>

namespace bh = boost::histogram;

int main() {
  // make histogram and set all cells to 0.25, simulating a perfectly uniform PDF,
  // this includes underflow and overflow cells
  auto a = bh::make_histogram(bh::axis::regular<>(4, 1.0, 3.0));
  a.at(0) = 0.25;
  a.at(1) = 0.25;
  a.at(2) = 0.25;
  a.at(3) = 0.25;

  // compute CDF, overriding histogram entries
  std::partial_sum(a.begin(), a.end(), a.begin());

  assert(a.at(0) == 0.25);
  assert(a.at(1) == 0.50);
  assert(a.at(2) == 0.75);
  assert(a.at(3) == 1.00);

  // make second PDF
  auto b = bh::make_histogram(bh::axis::regular<>(4, 1.0, 4.0));
  b.at(0) = 0.1;
  b.at(1) = 0.3;
  b.at(2) = 0.2;
  b.at(3) = 0.4;

  // computing cosine similiarity: cos(theta) = A dot B / sqrt((A dot A) * (B dot B))
  const auto aa = std::inner_product(a.begin(), a.end(), a.begin(), 0.0);
  const auto bb = std::inner_product(b.begin(), b.end(), b.begin(), 0.0);
  const auto ab = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  const auto cost = ab / std::sqrt(aa * bb);

  assert(std::abs(cost - 0.78) < 1e-2);
}

//]
