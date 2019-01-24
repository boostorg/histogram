// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_stl_algorithms

#include <boost/histogram.hpp>
#include <cassert>

#include <algorithm> // fill, any_of, min_element, max_element
#include <cmath>     // sqrt
#include <numeric>   // partial_sum, inner_product

namespace bh = boost::histogram;

int main() {
  // make histogram and set all cells to 0.25, simulating a perfectly uniform PDF,
  // this includes underflow and overflow cells
  auto a = bh::make_histogram(bh::axis::regular<>(4, 1.0, 3.0));
  std::fill(a.begin(), a.end(), 0.25); // set all counters to 0.25, including *flow cells
  // reset *flow cells to zero
  a.at(-1) = a.at(4) = 0;

  // compute CDF, overriding cell values
  std::partial_sum(a.begin(), a.end(), a.begin());

  assert(a.at(-1) == 0.0);
  assert(a.at(0) == 0.25);
  assert(a.at(1) == 0.50);
  assert(a.at(2) == 0.75);
  assert(a.at(3) == 1.00);
  assert(a.at(4) == 1.00);

  // use any_of to check if any cell values are smaller than 0.1,
  // and use indexed() to skip underflow and overflow cells
  auto a_ind = indexed(a);
  const auto any_small =
      std::any_of(a_ind.begin(), a_ind.end(), [](const auto& x) { return *x < 0.1; });
  assert(any_small == false); // underflow and overflow are zero, but skipped

  // find maximum element
  const auto max_it = std::max_element(a.begin(), a.end());
  assert(max_it == a.end() - 2);

  // find minimum element
  const auto min_it = std::min_element(a.begin(), a.end());
  assert(min_it == a.begin());

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
  const auto cos_sim = ab / std::sqrt(aa * bb);

  assert(std::abs(cos_sim - 0.78) < 1e-2);
}

//]
