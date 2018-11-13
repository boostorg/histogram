// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// clang-format off
//[ guide_fill_histogram

#include <boost/histogram.hpp>
#include <cassert>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

namespace bh = boost::histogram;

int main() {
  auto h = bh::make_histogram(bh::axis::regular<>(8, 0, 4),
                              bh::axis::regular<>(10, 0, 5));

  // fill histogram, number of arguments must be equal to number of axes
  h(0, 1.1);                // increases bin counter by one
  h(bh::weight(2), 3, 3.4); // increase bin counter by 2 instead of 1

  // fills from a tuple are also supported; passing a tuple of wrong size
  // causes an error at compile-time or an assertion at runtime in debug mode
  auto xy = std::make_tuple(4, 3.1);
  h(xy);

  // functional-style processing is also supported
  std::vector<std::tuple<int, double>> input_data;
  input_data.emplace_back(0, 1.2);
  input_data.emplace_back(2, 3.4);
  input_data.emplace_back(4, 5.6);

  // std::for_each takes the functor by value, we use a reference wrapper
  // here to avoid costly copies
  auto h2 = bh::make_histogram(bh::axis::regular<>(8, 0, 4),
                               bh::axis::regular<>(10, 0, 5));

  std::for_each(input_data.begin(), input_data.end(), std::ref(h2));

  // h2 is filled
  const double sum = std::accumulate(h2.begin(), h2.end(), 0.0);
  assert(sum == 3);
}

//]
