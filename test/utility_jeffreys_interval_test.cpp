// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/utility/jeffreys_interval.hpp>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::utility;

int main() {

  const double deps = std::numeric_limits<double>::epsilon();

  jeffreys_interval<> iv(deviation{1});

  {
    const auto x = iv(0, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.015608332029966311, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.537560439331611, deps);
  }

  {
    const auto x = iv(1, 0);
    BOOST_TEST_IS_CLOSE(x.first, 0.4624395606683889, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.9843916679700336, deps);
  }

  {
    const auto x = iv(5, 5);
    BOOST_TEST_IS_CLOSE(x.first, 0.34940656031996686, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.6505934396800331, deps);
  }

  {
    const auto x = iv(1, 9);
    BOOST_TEST_IS_CLOSE(x.first, 0.041888372640752326, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.23375944339475085, deps);
  }

  {
    const auto x = iv(9, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.7662405566052491, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.9581116273592477, deps);
  }

  return boost::report_errors();
}
