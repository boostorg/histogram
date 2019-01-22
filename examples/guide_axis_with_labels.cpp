// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_axis_with_labels

#include <boost/histogram.hpp>

namespace bh = boost::histogram;

int main() {
  // create a 2d-histogram with an "age" and an "income" axis
  auto h =
      bh::make_histogram(bh::axis::regular<>(20, 0.0, 100.0, "age in years"),
                         bh::axis::regular<>(20, 0.0, 100.0, "yearly income in $1000"));

  // do something with h
}

//]
