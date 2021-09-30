// Copyright 2021 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include "throw_exception.hpp"

namespace bh = boost::histogram;

// test deprecated API

int main() {
  bh::accumulators::sum<double> sum;

  sum += 1;

  BOOST_TEST_EQ(sum.large(), 1);
  BOOST_TEST_EQ(sum.small(), 0);

  sum += 1e20;

  BOOST_TEST_EQ(sum.large(), 1e20);
  BOOST_TEST_EQ(sum.small(), 1);

  return boost::report_errors();
}
