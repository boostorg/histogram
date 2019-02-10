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

int main() {
  using namespace boost::histogram;
  namespace tr = axis::transform;

  auto h = make_histogram(
      axis::regular<>(2, -1.0, 1.0),
      axis::regular<double, tr::log>(2, 1.0, 10.0, "axis 1"),
      axis::regular<double, tr::pow, axis::default_metadata, axis::option::growth>(
          tr::pow{1.5}, 2, 1.0, 10.0, "axis 2"),
      axis::circular<double, axis::null_type>(4, 0.0, 360.0), // axis without metadata
      axis::variable<double, std::string, axis::option::none, std::allocator<double>>(
          {-1.0, 0.0, 1.0}, "axis 4"),
      axis::category<>({2, 1, 3}, "axis 5"),
      axis::category<std::string>({"red", "blue"}, "axis 6"),
      axis::integer<>(-1, 1, "axis 7"));

  std::ostringstream os;
  os << h;

  std::cout << os.str() << std::endl;

  assert(os.str() ==
         "histogram(\n"
         "  regular(2, -1, 1, options=underflow | overflow),\n"
         "  regular_log(2, 1, 10, metadata=\"axis 1\", options=underflow | overflow),\n"
         "  regular_pow(2, 1, 10, metadata=\"axis 2\", options=growth, power=1.5),\n"
         "  regular(4, 0, 360, options=overflow | circular),\n"
         "  variable(-1, 0, 1, metadata=\"axis 4\", options=none),\n"
         "  category(2, 1, 3, metadata=\"axis 5\", options=overflow),\n"
         "  category(\"red\", \"blue\", metadata=\"axis 6\", options=overflow),\n"
         "  integer(-1, 1, metadata=\"axis 7\", options=underflow | overflow)\n"
         ")");
}

//]
