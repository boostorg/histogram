// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_axis_circular

#include <boost/histogram/axis.hpp>
#include <limits>

int main() {
  using namespace boost::histogram;

  // make a circular regular axis ... [1, 2), [2, 3), [3, 4), [1, 2) ....
  auto r = axis::regular<double, axis::transform::id, axis::default_metadata,
                         axis::join(axis::option::use_default, axis::option::circular)>{
      3, 1., 4.};
  assert(r.index(0.01) == 2);
  assert(r.index(1.01) == 0);
  assert(r.index(2.01) == 1);
  assert(r.index(3.01) == 2);
  assert(r.index(4.01) == 0);
  assert(r.index(5.01) == 1);
  assert(r.index(6.01) == 2);
  // special values are mapped to the overflow bin index
  assert(r.index(std::numeric_limits<double>::infinity()) == 3);
  assert(r.index(-std::numeric_limits<double>::infinity()) == 3);
  assert(r.index(std::numeric_limits<double>::quiet_NaN()) == 3);

  // since the regular axis is the most common circular axis, there exists an alias
  auto c = axis::circular<>{3, 1., 4.};
  assert(r == c);

  // make a circular integer axis
  auto i = axis::integer<int, axis::default_metadata, axis::option::circular>{1, 4};
  assert(i.index(0) == 2);
  assert(i.index(1) == 0);
  assert(i.index(2) == 1);
  assert(i.index(3) == 2);
  assert(i.index(4) == 0);
  assert(i.index(5) == 1);
}

//]
