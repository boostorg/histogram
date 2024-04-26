// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/fraction.hpp>
#include <boost/histogram/utility/jeffreys_interval.hpp>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::utility;
using namespace boost::histogram::accumulators;

template <class T>
void test() {
  // reference: table A.1 in
  // L.D. Brown, T.T. Cai, A. DasGupta, Statistical Science 16 (2001) 101–133,
  // doi:10.1214/ss/1009213286

  const T atol = 0.001;

  jeffreys_interval<T> iv(confidence_level{0.95});

  {
    auto p = iv(0.f, 7.f);
    BOOST_TEST_IS_CLOSE(p.first, 0.f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.41f, atol);
  }

  {
    auto p = iv(1.f, 6.f);
    BOOST_TEST_IS_CLOSE(p.first, 0.f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.501f, atol);
  }

  {
    auto p = iv(2.f, 5.f);
    BOOST_TEST_IS_CLOSE(p.first, 0.065f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.648f, atol);
  }

  {
    auto p = iv(3.f, 4.f);
    BOOST_TEST_IS_CLOSE(p.first, 0.139f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.766f, atol);
  }

  {
    auto p = iv(4.f, 7.f - 4.f);
    BOOST_TEST_IS_CLOSE(p.first, 0.234f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.861f, atol);
  }

  // extrapolated from table
  {
    auto p = iv(5.f, 2.f);
    BOOST_TEST_IS_CLOSE(p.first, 1.f - 0.648f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 1.f - 0.065f, atol);
  }

  // extrapolated from table
  {
    auto p = iv(6.f, 1.f);
    BOOST_TEST_IS_CLOSE(p.first, 1.f - 0.501f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 1.f, atol);
  }

  // extrapolated from table
  {
    auto p = iv(7.f, 0.f);
    BOOST_TEST_IS_CLOSE(p.first, 1.f - 0.41f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 1.f, atol);
  }

  // not in table
  {
    auto p = iv(0.f, 1.f);
    BOOST_TEST_IS_CLOSE(p.first, 0.f, atol);
    BOOST_TEST_IS_CLOSE(p.second, 0.975f, atol);

    fraction<T> f(0.f, 1.f);
    const auto y = iv(f);
    BOOST_TEST_IS_CLOSE(y.first, 0.f, atol);
    BOOST_TEST_IS_CLOSE(y.second, 0.975f, atol);
  }

  // not in table
  {
    auto p = iv(1.f, 0.f);
    BOOST_TEST_IS_CLOSE(p.first, 0.025, atol);
    BOOST_TEST_IS_CLOSE(p.second, 1, atol);

    fraction<T> f(1.f, 0.f);
    const auto y = iv(f);
    BOOST_TEST_IS_CLOSE(y.first, 0.025f, atol);
    BOOST_TEST_IS_CLOSE(y.second, 1.f, atol);
  }
}

int main() {

  test<float>();
  test<double>();

  return boost::report_errors();
}
