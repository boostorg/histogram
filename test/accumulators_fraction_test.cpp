// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/fraction.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/utility/wilson_interval.hpp>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"
#include "utility_str.hpp"

using namespace boost::histogram;
using namespace boost::histogram::accumulators;

template <class T>
void run_tests() {
  using f_t = fraction<T>;

  const double eps = std::numeric_limits<typename f_t::real_type>::epsilon();

  {
    f_t f;
    BOOST_TEST_EQ(f.successes(), 0);
    BOOST_TEST_EQ(f.failures(), 0);
  }

  {
    f_t f;
    f(true); f(true); f(true);
    f(false); f(false);
    BOOST_TEST_EQ(f.successes(), 3);
    BOOST_TEST_EQ(f.failures(), 2);
  }

  {
    using f_t1 = fraction<double>;
    using f_t2 = fraction<int>;
    f_t1 f1(5, 3);
    f_t2 f2(f1);
    BOOST_TEST_EQ(f2.successes(), 5);
    BOOST_TEST_EQ(f2.failures(), 3);
  }

  {
    f_t f(3, 1);
    BOOST_TEST_EQ(f.successes(), 3);
    BOOST_TEST_EQ(f.failures(), 1);
    BOOST_TEST_EQ(f.value(), 0.75);
    BOOST_TEST_IS_CLOSE(f.variance(), 0.75 * (1 - 0.75) / 4, eps);

    const auto ci = f.confidence_interval();
    const auto expected = utility::wilson_interval<double>()(3, 1);
    BOOST_TEST_IS_CLOSE(ci.first, expected.first, eps);
    BOOST_TEST_IS_CLOSE(ci.second, expected.second, eps);
  }

  {
    f_t f(3, 5);
    BOOST_TEST_EQ(f.successes(), 3);
    BOOST_TEST_EQ(f.failures(), 5);
    BOOST_TEST_EQ(f.value(), 0.375);
    BOOST_TEST_IS_CLOSE(f.variance(), 0.375 * (1 - 0.375) / 8, eps);

    const auto ci = f.confidence_interval();
    const auto expected = utility::wilson_interval<double>()(3, 5);
    BOOST_TEST_IS_CLOSE(ci.first, expected.first, eps);
    BOOST_TEST_IS_CLOSE(ci.second, expected.second, eps);
  }
}

int main() {

  run_tests<int>();
  run_tests<double>();
  run_tests<float>();

  return boost::report_errors();
}
