// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_make_dynamic_histogram

#include <boost/histogram.hpp>
#include <cassert>
#include <vector>

namespace bh = boost::histogram;

int main() {
  // create vector of three regular axes
  auto v1 = std::vector<bh::axis::regular<>>();
  v1.emplace_back(10, 0.0, 1.0);
  v1.emplace_back(20, 0.0, 2.0);
  v1.emplace_back(30, 0.0, 3.0);

  // create dynamic histogram from iterator range
  auto h1 = bh::make_histogram(v1.begin(), v1.end());
  assert(h1.rank() == 3);

  // create second vector of axis::any, a polymorphic axis type
  auto v2 = std::vector<bh::axis::variant<bh::axis::regular<>, bh::axis::integer<>>>();
  v2.emplace_back(bh::axis::regular<>(100, -1.0, 1.0));
  v2.emplace_back(bh::axis::integer<>(1, 7));

  // create dynamic histogram by moving the vector (avoids copies)
  auto h2 = bh::make_histogram(std::move(v2));
  assert(h2.rank() == 2);
}

//]
