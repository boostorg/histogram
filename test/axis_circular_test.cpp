// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/circular.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <limits>
#include "utility.hpp"

using namespace boost::histogram;

int main() {
  // bad_ctor
  {
    BOOST_TEST_THROWS(axis::circular<>(0), std::invalid_argument);
    BOOST_TEST_THROWS(axis::circular<>(2, 0, -1), std::invalid_argument);
  }

  // axis::circular
  {
    axis::circular<> a{4, 0, 1};
    BOOST_TEST_EQ(a[-1].lower(), a[a.size() - 1].lower() - 1);
    axis::circular<> b;
    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::circular<> c = std::move(b);
    BOOST_TEST_EQ(c, a);
    axis::circular<> d;
    BOOST_TEST_NE(c, d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a(-1.0 * 3), 0);
    BOOST_TEST_EQ(a(0.0), 0);
    BOOST_TEST_EQ(a(0.25), 1);
    BOOST_TEST_EQ(a(0.5), 2);
    BOOST_TEST_EQ(a(0.75), 3);
    BOOST_TEST_EQ(a(1.0), 0);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::infinity()), 4);
    BOOST_TEST_EQ(a(-std::numeric_limits<double>::infinity()), 4);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::quiet_NaN()), 4);
  }

  // iterators
  {
    test_axis_iterator(axis::circular<>(5, 0, 1, ""), 0, 5);
  }

  return boost::report_errors();
}
