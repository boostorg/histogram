// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/axis/circular.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include "utility.hpp"

using namespace boost::histogram;

int main() {
  // bad_ctor
  {
    BOOST_TEST_THROWS(axis::integer<>(1, 1), std::invalid_argument);
    BOOST_TEST_THROWS(axis::integer<>(1, -1), std::invalid_argument);
  }

  // axis::integer
  {
    axis::integer<> a{-1, 2};
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<double>::infinity());
    axis::integer<> b;
    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::integer<> c = std::move(b);
    BOOST_TEST_EQ(c, a);
    axis::integer<> d;
    BOOST_TEST_NE(c, d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a(-10), -1);
    BOOST_TEST_EQ(a(-2), -1);
    BOOST_TEST_EQ(a(-1), 0);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(1), 2);
    BOOST_TEST_EQ(a(2), 3);
    BOOST_TEST_EQ(a(10), 3);
  }

  // axis::integer with int type
  {
    axis::integer<int> a{-1, 2};
    BOOST_TEST_EQ(a[-2].value(), std::numeric_limits<int>::min());
    BOOST_TEST_EQ(a[4].value(), std::numeric_limits<int>::max());
    BOOST_TEST_EQ(a(-10), -1);
    BOOST_TEST_EQ(a(-2), -1);
    BOOST_TEST_EQ(a(-1), 0);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(1), 2);
    BOOST_TEST_EQ(a(2), 3);
    BOOST_TEST_EQ(a(10), 3);
  }

  // iterators
  {
    test_axis_iterator(axis::integer<>(0, 4, ""), 0, 4);
  }

  return boost::report_errors();
}
