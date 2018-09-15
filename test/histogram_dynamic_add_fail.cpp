// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>

using namespace boost::histogram;
int main() {
  auto a = make_dynamic_histogram(axis::integer<>(0, 2));
  auto b = make_dynamic_histogram(axis::integer<>(0, 3));
  a += b;
}
