// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/fraction.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include "is_close.hpp"
#include "throw_exception.hpp"
#include "utility_str.hpp"

using namespace boost::histogram::accumulators;

template <class T>
void run_tests() {
  using f_t = fraction<T>;

  // ctor
  {
    f_t f1(3, 1);
    f_t f2(3.0, 5.0);
    f_t f3(1.0, 3.0);
    BOOST_TEST_EQ(f1.successes(), 3); BOOST_TEST_EQ(f1.failures(), 1);
    BOOST_TEST_EQ(f2.successes(), 3); BOOST_TEST_EQ(f2.failures(), 5);
    BOOST_TEST_EQ(f3.successes(), 1); BOOST_TEST_EQ(f3.failures(), 3);
    BOOST_TEST_EQ(f1.value(), 0.75);
    BOOST_TEST_EQ(f2.value(), 0.375);
    BOOST_TEST_EQ(f3.value(), 0.25);
    BOOST_TEST_IS_CLOSE(f1.variance(), 12);
    BOOST_TEST_IS_CLOSE(f2.variance(), 120);
    BOOST_TEST_IS_CLOSE(f3.variance(), 12);
    // BOOST_TEST_EQ(f1.confidence_interval().first, 0.5); BOOST_TEST_EQ(f1.confidence_interval().second, 0.9);
  }
}

int main() {

  // run_tests<int>(); // confidence_interval() throws error due to static_assert in binomial_proportion_interval()
  run_tests<double>();
  run_tests<float>();

  return boost::report_errors();
}
