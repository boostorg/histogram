// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_axis_basic_demo

#include <boost/histogram/axis.hpp>
#include <limits>

int main() {
  using namespace boost::histogram;

  // make a regular axis with 10 bins over interval from 1.5 to 2.5
  axis::regular<> r{10, 1.5, 2.5};
  // `<>` is needed because the axis is templated, you can drop this on a C++17 compiler
  assert(r.size() == 10); // ok

  // histogram uses the `index` method to convert values to indices
  // note: intervals of builtin axis types are always semi-open [a, b)
  assert(r.index(1.5) == 0);
  assert(r.index(1.6) == 1);
  assert(r.index(2.4) == 9);
  // index for a value below the start of the axis is always -1
  assert(r.index(1.0) == -1);
  assert(r.index(-std::numeric_limits<double>::infinity()) == -1);
  // index for a value below the above the end of the axis is always `size()`
  assert(r.index(3.0) == 10);
  assert(r.index(std::numeric_limits<double>::infinity()) == 10);
  // index for not-a-number is also `size()` by convention
  assert(r.index(std::numeric_limits<double>::quiet_NaN()) == 10);

  // make a variable axis with 3 bins [-1.5, 0.1), [0.1, 0.3), [0.3, 10)
  axis::variable<> v{-1.5, 0.1, 0.3, 10.};
  assert(v.index(-2.0) == -1);
  assert(v.index(-1.5) == 0);
  assert(v.index(0.1) == 1);
  assert(v.index(0.3) == 2);
  assert(v.index(10) == 3);
  assert(v.index(20) == 3);

  // make an integer axis with 3 bins at -1, 0, 1
  axis::integer<> i{-1, 2};
  assert(i.index(-2) == -1);
  assert(i.index(-1) == 0);
  assert(i.index(0) == 1);
  assert(i.index(1) == 2);
  assert(i.index(2) == 3);

  // make an integer axis called "foo"
  axis::integer<> i_with_label{-1, 2, "foo"};
  // all builtin axis types allow you to pass some optional metadata as the last
  // argument in the constructor; a string by default, but can be any copyable type

  // two axis do not compare equal if they differ in their metadata
  assert(i != i_with_label);

  // integer axis also work well with unscoped enums
  enum { red, blue };
  axis::integer<> i_for_enum{red, blue + 1};
  assert(i_for_enum.index(red) == 0);
  assert(i_for_enum.index(blue) == 1);

  // make a category axis from a scoped enum and/or if the identifiers are not consecutive
  enum class Bar { red = 12, blue = 6 };
  axis::category<Bar> c{Bar::red, Bar::blue};
  assert(c.index(Bar::red) == 0);
  assert(c.index(Bar::blue) == 1);
  // c.index(12) is a compile-time error, since the argument must be of type `Bar`

  // a category axis can be created for any copyable and equal-comparable type
  axis::category<std::string> c_string{"red", "blue"};
  assert(c_string.index("red") == 0);
  assert(c_string.index("blue") == 1);
}

//]
