// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/utility/wald_interval.hpp>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::utility;

int main() {

  const double deps = std::numeric_limits<double>::epsilon();

  wald_interval<> iv(deviation{1});

  {
    const auto x = iv(0, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.0, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.0, deps);
  }

  {
    const auto x = iv(1, 0);
    BOOST_TEST_IS_CLOSE(x.first, 1.0, deps);
    BOOST_TEST_IS_CLOSE(x.second, 1.0, deps);
  }

  {
    const auto x = iv(5, 5);
    BOOST_TEST_IS_CLOSE(x.first, 0.341886116991581, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.658113883008419, deps);
  }

  {
    const auto x = iv(1, 9);
    BOOST_TEST_IS_CLOSE(x.first, 0.005131670194948618, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.1948683298050514, deps);
  }

  {
    const auto x = iv(9, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.8051316701949487, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.9948683298050514, deps);
  }

  return boost::report_errors();
}
