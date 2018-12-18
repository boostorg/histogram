// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <limits>
#include <sstream>
#include <type_traits>
#include "utility_axis.hpp"

using namespace boost::histogram;

int main() {
  // bad_ctor
  {
    BOOST_TEST_THROWS(axis::integer<>(1, 1), std::invalid_argument);
    BOOST_TEST_THROWS(axis::integer<>(1, -1), std::invalid_argument);
  }

  // axis::integer with double type
  {
    axis::integer<double> a{-1, 2};
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<double>::infinity());
    axis::integer<double> b;
    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::integer<double> c = std::move(b);
    BOOST_TEST_EQ(c, a);
    axis::integer<double> d;
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
    BOOST_TEST_EQ(a(std::numeric_limits<double>::quiet_NaN()), 3);
  }

  // axis::integer with int type
  {
    axis::integer<int> a{-1, 2};
    BOOST_TEST_EQ(a[-2], std::numeric_limits<int>::min());
    BOOST_TEST_EQ(a[4], std::numeric_limits<int>::max());
    BOOST_TEST_EQ(a(-10), -1);
    BOOST_TEST_EQ(a(-2), -1);
    BOOST_TEST_EQ(a(-1), 0);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(1), 2);
    BOOST_TEST_EQ(a(2), 3);
    BOOST_TEST_EQ(a(10), 3);
  }

  // axis::integer int,circular
  {
    axis::integer<int, axis::null_type, axis::option_type::circular> a(-1, 1);
    BOOST_TEST_EQ(a.value(-1), -2);
    BOOST_TEST_EQ(a.value(0), -1);
    BOOST_TEST_EQ(a.value(1), 0);
    BOOST_TEST_EQ(a.value(2), 1);
    BOOST_TEST_EQ(a.value(3), 2);
    BOOST_TEST_EQ(a(-2), 1);
    BOOST_TEST_EQ(a(-1), 0);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(1), 0);
    BOOST_TEST_EQ(a(2), 1);
  }

  // axis::integer double,circular
  {
    axis::integer<double, axis::null_type, axis::option_type::circular> a(-1, 1);
    BOOST_TEST_EQ(a.value(-1), -2);
    BOOST_TEST_EQ(a.value(0), -1);
    BOOST_TEST_EQ(a.value(1), 0);
    BOOST_TEST_EQ(a.value(2), 1);
    BOOST_TEST_EQ(a.value(3), 2);
    BOOST_TEST_EQ(a(-2), 1);
    BOOST_TEST_EQ(a(-1), 0);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(1), 0);
    BOOST_TEST_EQ(a(2), 1);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::quiet_NaN()), 2);
  }

  // iterators
  {
    test_axis_iterator(axis::integer<int>(0, 4), 0, 4);
    test_axis_iterator(axis::integer<double>(0, 4), 0, 4);
    test_axis_iterator(
        axis::integer<int, axis::null_type, axis::option_type::circular>(0, 4), 0, 4);
  }

  // shrink and rebin
  {
    using A = axis::integer<>;
    auto a = A(0, 5);
    auto b = A(a, 1, 4, 1);
    BOOST_TEST_EQ(b.size(), 3);
    BOOST_TEST_EQ(b.value(0), 1);
    BOOST_TEST_EQ(b.value(3), 4);
    auto c = A(a, 0, 4, 1);
    BOOST_TEST_EQ(c.size(), 4);
    BOOST_TEST_EQ(c.value(0), 0);
    BOOST_TEST_EQ(c.value(4), 4);
    auto e = A(a, 1, 4, 1);
    BOOST_TEST_EQ(e.size(), 3);
    BOOST_TEST_EQ(e.value(0), 1);
    BOOST_TEST_EQ(e.value(3), 4);
  }

  // shrink and rebin with circular option
  {
    using A = axis::integer<int, axis::null_type, axis::option_type::circular>;
    auto a = A(1, 5);
    BOOST_TEST_THROWS(A(a, 1, 4, 1), std::invalid_argument);
    BOOST_TEST_THROWS(A(a, 0, 3, 1), std::invalid_argument);
    BOOST_TEST_THROWS(A(a, 0, 4, 2), std::invalid_argument);
  }

  return boost::report_errors();
}
