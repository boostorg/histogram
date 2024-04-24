// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>
#include "throw_exception.hpp"

int main() {
  using namespace boost::histogram;

  auto h = make_histogram(axis::integer<>(0, 5));

  h(0);

  auto values = {1, 2};
  h.fill(values);
}
