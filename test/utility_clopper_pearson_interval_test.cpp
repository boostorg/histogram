// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/fraction.hpp>
#include <boost/histogram/utility/clopper_pearson_interval.hpp>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::utility;
using namespace boost::histogram::accumulators;

template <class T>
void test() {
  const T atol = 0.001;

  clopper_pearson_interval<T> iv(deviation{1});

  {
    const auto x = iv(0.f, 1.f);
    BOOST_TEST_IS_CLOSE(x.first, 0.f, atol);
    BOOST_TEST_IS_CLOSE(x.second, 0.841f, atol);

    fraction<T> f(0.f, 1.f);
    const auto y = iv(f);
    BOOST_TEST_IS_CLOSE(y.first, 0.f, atol);
    BOOST_TEST_IS_CLOSE(y.second, 0.841f, atol);
  }

  {
    const auto x = iv(1.f, 0.f);
    BOOST_TEST_IS_CLOSE(x.first, 0.158f, atol);
    BOOST_TEST_IS_CLOSE(x.second, 1.f, atol);

    fraction<T> f(1.f, 0.f);
    const auto y = iv(f);
    BOOST_TEST_IS_CLOSE(y.first, 0.158f, atol);
    BOOST_TEST_IS_CLOSE(y.second, 1.f, atol);
  }

  {
    const auto x = iv(5.f, 5.f);
    BOOST_TEST_IS_CLOSE(x.first, 0.304f, atol);
    BOOST_TEST_IS_CLOSE(x.second, 0.695f, atol);
  }

  {
    const auto x = iv(1.f, 9.f);
    BOOST_TEST_IS_CLOSE(x.first, 0.017f, atol);
    BOOST_TEST_IS_CLOSE(x.second, 0.294f, atol);
  }

  {
    const auto x = iv(9.f, 1.f);
    BOOST_TEST_IS_CLOSE(x.first, 0.705f, atol);
    BOOST_TEST_IS_CLOSE(x.second, 0.982f, atol);
  }
}

int main() {
  test<float>();
  test<double>();
  return boost::report_errors();
}
