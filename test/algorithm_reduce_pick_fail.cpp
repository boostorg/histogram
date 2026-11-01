// Copyright 2026 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/algorithm/reduce.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/make_histogram.hpp>
#include <vector>

int main() {
  using namespace boost::histogram;
  using int_axis = axis::integer<>;
  struct ordered_with_pick_ctor : int_axis {
    using int_axis::int_axis;
    // ordered axis is not pickable, even if it has the pick constructor
    ordered_with_pick_ctor(const ordered_with_pick_ctor&, const axis::index_type*,
                           const axis::index_type*) {}
  };

  auto h = make_histogram(ordered_with_pick_ctor(0, 5));
  h(2);
  (void)algorithm::reduce(h, algorithm::pick({0, 2}));
}
