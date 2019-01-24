// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_custom_2d_axis

#include <boost/histogram.hpp>
#include <cassert>

namespace bh = boost::histogram;

int main() {
  // axis which returns 1 if the input falls inside the unit circle and zero otherwise
  struct circle_axis {
    bh::axis::index_type operator()(std::tuple<double, double> point) const {
      const auto x = std::get<0>(point);
      const auto y = std::get<1>(point);
      return x * x + y * y <= 1.0;
    }

    bh::axis::index_type size() const { return 2; }
  };

  auto h1 = bh::make_histogram(circle_axis());

  // call looks like for a histogram with N 1d axes if histogram has only one Nd axis
  h1(0, 0);   // in
  h1(0, -1);  // in
  h1(0, 1);   // in
  h1(-1, 0);  // in
  h1(1, 0);   // in
  h1(1, 1);   // out
  h1(-1, -1); // out

  assert(h1.at(0) == 2); // out
  assert(h1.at(1) == 5); // in

  auto h2 =
      bh::make_histogram(circle_axis(), bh::axis::category<std::string>({"red", "blue"}));

  // pass arguments for first axis as std::tuple
  h2(std::make_tuple(0, 0), "red");
  h2(std::make_tuple(1, 1), "blue");

  assert(h2.at(0, 0) == 0); // out, red
  assert(h2.at(0, 1) == 1); // out, blue
  assert(h2.at(1, 0) == 1); // in, red
  assert(h2.at(1, 1) == 0); // in, blue
}

//]
