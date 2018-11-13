// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_axis_with_uoflow_off

#include <boost/histogram.hpp>

namespace bh = boost::histogram;

int main() {
  // create a 1d-histogram for dice throws with integer values from 1 to 6
  auto h = bh::make_histogram(
    bh::axis::integer<>(1, 7, "eyes", bh::axis::option_type::none)
  );

  // do something with h
}

//]
