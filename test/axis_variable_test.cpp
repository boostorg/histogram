// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <limits>
#include "utility_axis.hpp"

using namespace boost::histogram;

int main() {
  // bad_ctors
  {
    auto empty = std::vector<double>(0);
    BOOST_TEST_THROWS((axis::variable<>(empty)), std::invalid_argument);
    BOOST_TEST_THROWS(axis::variable<>({1.0}), std::invalid_argument);
  }

  // axis::variable
  {
    axis::variable<> a{-1, 0, 1};
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<double>::infinity());
    axis::variable<> b;
    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::variable<> c = std::move(b);
    BOOST_TEST_EQ(c, a);
    axis::variable<> d;
    BOOST_TEST_NE(c, d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    axis::variable<> e{-2, 0, 2};
    BOOST_TEST_NE(a, e);
    BOOST_TEST_EQ(a(-10), -1);
    BOOST_TEST_EQ(a(-1), 0);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(1), 2);
    BOOST_TEST_EQ(a(10), 2);
    BOOST_TEST_EQ(a(-std::numeric_limits<double>::infinity()), -1);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::infinity()), 2);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::quiet_NaN()), 2);
  }

  // axis::variable circular
  {
    axis::variable<double, axis::null_type, axis::option_type::circular> a{-1, 1, 2};
    BOOST_TEST_EQ(a.value(-2), -4);
    BOOST_TEST_EQ(a.value(-1), -2);
    BOOST_TEST_EQ(a.value(0), -1);
    BOOST_TEST_EQ(a.value(1), 1);
    BOOST_TEST_EQ(a.value(2), 2);
    BOOST_TEST_EQ(a.value(3), 4);
    BOOST_TEST_EQ(a.value(4), 5);
    BOOST_TEST_EQ(a(-3), 0); // -3 + 3 = 0
    BOOST_TEST_EQ(a(-2), 1); // -2 + 3 = 1
    BOOST_TEST_EQ(a(-1), 0);
    BOOST_TEST_EQ(a(0), 0);
    BOOST_TEST_EQ(a(1), 1);
    BOOST_TEST_EQ(a(2), 0);
    BOOST_TEST_EQ(a(3), 0); // 3 - 3 = 0
    BOOST_TEST_EQ(a(4), 1); // 4 - 3 = 1
  }

  // iterators
  {
    test_axis_iterator(axis::variable<>{1, 2, 3}, 0, 2);
    test_axis_iterator(
        axis::variable<double, axis::null_type, axis::option_type::circular>{1, 2, 3}, 0,
        2);
  }

  // shrink and rebin
  {
    using A = axis::variable<>;
    auto a = A({0, 1, 2, 3, 4, 5});
    auto b = A(a, 1, 4, 1);
    BOOST_TEST_EQ(b.size(), 3);
    BOOST_TEST_EQ(b.value(0), 1);
    BOOST_TEST_EQ(b.value(3), 4);
    auto c = A(a, 0, 4, 2);
    BOOST_TEST_EQ(c.size(), 2);
    BOOST_TEST_EQ(c.value(0), 0);
    BOOST_TEST_EQ(c.value(2), 4);
    auto e = A(a, 1, 5, 2);
    BOOST_TEST_EQ(e.size(), 2);
    BOOST_TEST_EQ(e.value(0), 1);
    BOOST_TEST_EQ(e.value(2), 5);
  }

  // shrink and rebin with circular option
  {
    using A = axis::variable<double, axis::null_type, axis::option_type::circular>;
    auto a = A({1, 2, 3, 4, 5});
    BOOST_TEST_THROWS(A(a, 1, 4, 1), std::invalid_argument);
    BOOST_TEST_THROWS(A(a, 0, 3, 1), std::invalid_argument);
    auto b = A(a, 0, 4, 2);
    BOOST_TEST_EQ(b.size(), 2);
    BOOST_TEST_EQ(b.value(0), 1);
    BOOST_TEST_EQ(b.value(2), 5);
  }

  return boost::report_errors();
}
