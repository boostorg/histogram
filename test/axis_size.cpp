// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/axis.hpp>
#include <iostream>

int main() {

#define SIZEOF(axis)                                                           \
  std::cout << #axis << " " << sizeof(boost::histogram::axis) << std::endl
  SIZEOF(axis::regular<>);
  SIZEOF(axis::circular<>);
  SIZEOF(axis::variable<>);
  SIZEOF(axis::integer);
  SIZEOF(axis::category);
}
