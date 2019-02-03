// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_histogram_streaming

#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/ostream.hpp>
#include <cassert>
#include <memory>
#include <sstream>
#include <string>

namespace bh = boost::histogram;

int main() {
  namespace axis = bh::axis;

  auto h = bh::make_histogram(
      axis::regular<>(2, -1.0, 1.0),
      axis::regular<double, axis::transform::log>(2, 1.0, 10.0, "axis 1"),
      axis::circular<double, axis::null_type>(4, 0.0, 360.0), // axis without metadata
      axis::variable<double, std::string, axis::option::none, std::allocator<double>>(
          {-1.0, 0.0, 1.0}, "axis 3"),
      axis::category<>({2, 1, 3}, "axis 4"), axis::integer<>(-1, 1, "axis 5"));

  std::ostringstream os;
  os << h;

  std::cout << os.str() << std::endl;

  assert(os.str() ==
         "histogram(\n"
         "  regular(2, -1, 1, options=underflow | overflow),\n"
         "  regular_log(2, 1, 10, metadata=\"axis 1\", options=underflow | overflow),\n"
         "  regular(4, 0, 360, options=overflow | circular),\n"
         "  variable(-1, 0, 1, metadata=\"axis 3\", options=none),\n"
         "  category(2, 1, 3, metadata=\"axis 4\", options=overflow),\n"
         "  integer(-1, 1, metadata=\"axis 5\", options=underflow | overflow)\n"
         ")");
}

//]
