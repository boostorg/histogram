// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//[ guide_histogram_streaming

#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

int main() {
  using namespace boost::histogram;
  namespace tr = axis::transform;

  auto h = make_histogram(axis::regular<>(2, -1.0, 1.0, "axis 1"),
                          axis::category<std::string>({"red", "blue"}, "axis 2"));

  std::ostringstream os;
  os << h;

  std::cout << os.str() << std::endl;

  assert(os.str() ==
         "histogram(\n"
         "  regular(2, -1, 1, metadata=\"axis 1\", options=underflow | overflow)\n"
         "  category(\"red\", \"blue\", metadata=\"axis 2\", options=overflow)\n"
         "  (-1 0): 0 ( 0 0): 0 ( 1 0): 0 ( 2 0): 0 (-1 1): 0 ( 0 1): 0\n"
         "  ( 1 1): 0 ( 2 1): 0 (-1 2): 0 ( 0 2): 0 ( 1 2): 0 ( 2 2): 0\n"
         ")");
}

//]
