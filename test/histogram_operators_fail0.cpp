// Copyright 2020 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/make_histogram.hpp>
#include "dummy_storage.hpp"

int main() {
  using namespace boost::histogram;

  auto h =
      make_histogram_with(dummy_storage<unscaleable, false>{}, axis::integer<>(0, 1));

  h *= 2;
}
