// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// clang-format off

//[ guide_axis_growing

#include <boost/histogram.hpp>
#include <cassert>

#include <iostream>

int main() {
  using namespace boost::histogram;

  // make a growing regular axis
  // - it grows new bins with its constant bin width until the value is covered
  auto h1 = make_histogram(axis::regular<double,
                                         axis::transform::id,
                                         axis::default_metadata,
                                         axis::option::growth>{2, 0., 1.});
  // nothing special happens here
  h1(0.1);
  h1(0.9);
  // state: [0, 0.5): 1, [0.5, 1.0): 1
  assert(h1.axis().size() == 2);
  assert(h1.axis().bin(0).lower() == 0.0);
  assert(h1.axis().bin(1).upper() == 1.0);

  // value below range: axis grows new bins until value is in range, all indices shift
  h1(-0.3);
  // state: [-0.5, 0.0): 1, [0, 0.5): 1, [0.5, 1.0): 1
  std::cout << h1.axis().size() << std::endl;
  std::cout << h1.axis().bin(0).lower() << std::endl;
  std::cout << h1.axis().bin(2).upper() << std::endl;
  assert(h1.axis().size() == 3);
  assert(h1.axis().bin(0).lower() == -0.5);
  assert(h1.axis().bin(2).upper() == 1.0);

  h1(1.9);
  // state: [-0.5, 0.0): 1, [0, 0.5): 1, [0.5, 1.0): 1, [1.0, 1.5): 0 [1.5, 2.0): 1
  assert(h1.axis().size() == 5);
  assert(h1.axis().bin(0).lower() == -0.5);
  assert(h1.axis().bin(4).upper() == 2.0);
}

//]
