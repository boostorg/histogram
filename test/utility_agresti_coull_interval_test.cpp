// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/utility/agresti_coull_interval.hpp>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::utility;

int main() {

  const double deps = std::numeric_limits<double>::epsilon();

  agresti_coull_interval<> iv(deviation{1});

  {
    const auto x = iv(0, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.0, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.5561862178478972, deps);
  }

  {
    const auto x = iv(1, 0);
    BOOST_TEST_IS_CLOSE(x.first, 0.44381378215210276, deps);
    BOOST_TEST_IS_CLOSE(x.second, 1.0, deps);
  }

  {
    const auto x = iv(5, 5);
    BOOST_TEST_IS_CLOSE(x.first, 0.3492443277111182, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.6507556722888819, deps);
  }

  {
    const auto x = iv(1, 9);
    BOOST_TEST_IS_CLOSE(x.first, 0.032892694003727976, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.23983457872354474, deps);
  }

  {
    const auto x = iv(9, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.7601654212764553, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.967107305996272, deps);
  }

  return boost::report_errors();
}
