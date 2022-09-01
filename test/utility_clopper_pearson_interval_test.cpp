// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/utility/clopper_pearson_interval.hpp>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::utility;

int main() {

  // const double deps = std::numeric_limits<double>::epsilon();
  const double deps = 0.1;

  clopper_pearson_interval<> iv(deviation{1});

  {
    const auto x = iv(0, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.0, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.8413447460685429, deps);
  }

  {
    const auto x = iv(1, 0);
    BOOST_TEST_IS_CLOSE(x.first, 0.1586552539314571, deps);
    BOOST_TEST_IS_CLOSE(x.second, 1.0, deps);
  }

  {
    const auto x = iv(5, 5);
    BOOST_TEST_IS_CLOSE(x.first, 0.3048178830085313, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.6951821169914687, deps);
  }

  {
    const auto x = iv(1, 9);
    BOOST_TEST_IS_CLOSE(x.first, 0.017127014136728243, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.29413538963846697, deps);
  }

  {
    const auto x = iv(9, 1);
    BOOST_TEST_IS_CLOSE(x.first, 0.705864610361533, deps);
    BOOST_TEST_IS_CLOSE(x.second, 0.9828729858632718, deps);
  }

  return boost::report_errors();
}
