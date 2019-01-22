// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ getting_started_listing_02

#include <boost/format.hpp>
#include <boost/histogram.hpp>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

namespace bh = boost::histogram;

int main() {
  /*
    Create a histogram which can be configured dynamically at run-time. The axis
    configuration is first collected in a vector of the boost::histogram::axis::variant
    type, which can hold different axis types (those in its template argument list).
    Here, we use a variant that can store a regular and a category axis.
  */
  using reg = bh::axis::regular<>;
  using cat = bh::axis::category<std::string>;
  using variant = bh::axis::variant<bh::axis::regular<>, bh::axis::category<std::string>>;
  std::vector<variant> axes;
  axes.emplace_back(cat({"red", "blue"}));
  axes.emplace_back(reg(3, 0.0, 1.0, "x"));
  axes.emplace_back(reg(3, 0.0, 1.0, "y"));
  auto h = bh::make_histogram(std::move(axes)); // passing an iterator range also works

  // fill histogram with data, usually this happens in a loop
  h("red", 0.1, 0.2);
  h("blue", 0.7, 0.3);
  h("red", 0.3, 0.7);
  h("red", 0.7, 0.7);

  /*
    Print histogram by iterating over bins.
    Since the [bin type] of the category axis cannot be converted into a double, it cannot
    be handled by the polymorphic interface of boost::histogram::axis::variant. We use
    boost::histogram::axis::get to "cast" the variant type to the actual category type.
  */

  const auto& cat_axis = bh::axis::get<cat>(h.axis(0)); // get reference to category axis
  std::ostringstream os;
  for (auto x : bh::indexed(h)) {
    os << boost::format("(%i, %i, %i) %4s [%3.1f, %3.1f) [%3.1f, %3.1f) %3.0f\n") %
              x.index(0) % x.index(1) % x.index(2) % cat_axis[x.index(0)] %
              x.bin(1).lower() % x.bin(1).upper() % x.bin(2).lower() % x.bin(2).upper() %
              *x;
  }

  std::cout << os.str() << std::flush;
  assert(os.str() == "(0, 0, 0)  red [0.0, 0.3) [0.0, 0.3)   1\n"
                     "(1, 0, 0) blue [0.0, 0.3) [0.0, 0.3)   0\n"
                     "(0, 1, 0)  red [0.3, 0.7) [0.0, 0.3)   0\n"
                     "(1, 1, 0) blue [0.3, 0.7) [0.0, 0.3)   0\n"
                     "(0, 2, 0)  red [0.7, 1.0) [0.0, 0.3)   0\n"
                     "(1, 2, 0) blue [0.7, 1.0) [0.0, 0.3)   1\n"
                     "(0, 0, 1)  red [0.0, 0.3) [0.3, 0.7)   0\n"
                     "(1, 0, 1) blue [0.0, 0.3) [0.3, 0.7)   0\n"
                     "(0, 1, 1)  red [0.3, 0.7) [0.3, 0.7)   0\n"
                     "(1, 1, 1) blue [0.3, 0.7) [0.3, 0.7)   0\n"
                     "(0, 2, 1)  red [0.7, 1.0) [0.3, 0.7)   0\n"
                     "(1, 2, 1) blue [0.7, 1.0) [0.3, 0.7)   0\n"
                     "(0, 0, 2)  red [0.0, 0.3) [0.7, 1.0)   1\n"
                     "(1, 0, 2) blue [0.0, 0.3) [0.7, 1.0)   0\n"
                     "(0, 1, 2)  red [0.3, 0.7) [0.7, 1.0)   0\n"
                     "(1, 1, 2) blue [0.3, 0.7) [0.7, 1.0)   0\n"
                     "(0, 2, 2)  red [0.7, 1.0) [0.7, 1.0)   1\n"
                     "(1, 2, 2) blue [0.7, 1.0) [0.7, 1.0)   0\n");
}

//]
