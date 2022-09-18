// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/utility/jeffreys_interval.hpp>
#include <limits>
#include "boost/histogram/utility/binomial_proportion_interval.hpp"
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::utility;

int main() {

  // reference: table A.1 in
  // L.D. Brown, T.T. Cai, A. DasGupta, Statistical Science 16 (2001) 101–133,
  // doi:10.1214/ss/1009213286

  const double atol = 0.001;

  jeffreys_interval<> iv(confidence_level{0.95});

  struct data_t {
    int n, s;
    double a, b;
  };

  {
    auto p = iv(0, 1);
    BOOST_TEST_IS_CLOSE(p.first, 0, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.975, atol);
  }

  {
    auto p = iv(1, 0);
    BOOST_TEST_IS_CLOSE(p.first, 0.025, atol);
    BOOST_TEST_IS_CLOSE(p.second, 1, atol);
  }

  {
    auto p = iv(0, 7 - 0);
    BOOST_TEST_IS_CLOSE(p.first, 0, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.41, atol);
  }

  {
    auto p = iv(1, 7 - 1);
    BOOST_TEST_IS_CLOSE(p.first, 0, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.501, atol);
  }

  {
    auto p = iv(2, 7 - 2);
    BOOST_TEST_IS_CLOSE(p.first, 0.065, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.648, atol);
  }

  {
    auto p = iv(3, 7 - 3);
    BOOST_TEST_IS_CLOSE(p.first, 0.139, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.766, atol);
  }

  {
    auto p = iv(4, 7 - 4);
    BOOST_TEST_IS_CLOSE(p.first, 0.234, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.861, atol);
  }

  return boost::report_errors();
}
