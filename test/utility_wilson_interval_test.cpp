// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/utility/wilson_interval.hpp>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::utility;

int main() {

  const double deps = std::numeric_limits<double>::epsilon();

  wilson_interval<> iv(deviation{1});

  {
    const auto x = iv(0, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.0, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.5, deps);
  }

  {
    const auto x = iv(1, 0);
    BOOST_TEST_IS_CLOSE(x.first, 0.5, deps);
    BOOST_TEST_IS_CLOSE(x.second, 1.0, deps);
  }

  {
    const auto x = iv(5, 5);
    BOOST_TEST_IS_CLOSE(x.first, 0.3492443277111182, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.6507556722888819, deps);
  }

  {
    const auto x = iv(1, 9);
    BOOST_TEST_IS_CLOSE(x.first, 0.03887449732033081, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.23385277540694188, deps);
  }

  {
    const auto x = iv(9, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.7661472245930581, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.9611255026796692, deps);
  }

  return boost::report_errors();
}
