// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/efficiency.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include "throw_exception.hpp"
#include "utility_str.hpp"

using namespace boost::histogram;

template <class T>
void run_tests() {
  using e_t = accumulators::efficiency<T>;

  // ctor
  {
    e_t e(1, 2);
    BOOST_TEST_EQ(e.successes(), 1);
    BOOST_TEST_EQ(e.failures(), 2);
  }
}

int main() {
  run_tests<int>();
  run_tests<double>();
  run_tests<float>();

  return boost::report_errors();
}
