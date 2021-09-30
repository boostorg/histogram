// Copyright 2021 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// The header windows.h and possibly others do the following
#define small char
// which violates the C++ standard. We make sure here that including our headers work
// nevertheless. We avoid the name `small`.

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include "throw_exception.hpp"

namespace bh = boost::histogram;

int main() {
  bh::accumulators::sum<double> sum;

  sum += 1;

  BOOST_TEST_EQ(sum.large_part(), 1);
  BOOST_TEST_EQ(sum.small_part(), 0);

  sum += 1e20;

  BOOST_TEST_EQ(sum.large_part(), 1e20);
  BOOST_TEST_EQ(sum.small_part(), 1);

  return boost::report_errors();
}
