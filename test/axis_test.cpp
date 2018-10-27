// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/axis/category.hpp>
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
  // bad_ctors
  {
    auto empty = std::vector<double>(0);
    BOOST_TEST_THROWS(axis::circular<>(0), std::invalid_argument);
    BOOST_TEST_THROWS((axis::variable<>(empty)), std::invalid_argument);
    BOOST_TEST_THROWS(axis::variable<>({1.0}), std::invalid_argument);
    BOOST_TEST_THROWS(axis::integer<>(1, -1), std::invalid_argument);
    BOOST_TEST_THROWS((axis::category<>(empty)), std::invalid_argument);
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
    BOOST_TEST_NE(b, a);
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

  // axis::integer
  {
    axis::integer<> a{-1, 2};
    BOOST_TEST_EQ(a[-1].lower(), std::numeric_limits<int>::min());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<int>::max());
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

  // axis::category
  {
    std::string A("A"), B("B"), C("C"), other;
    axis::category<std::string> a({A, B, C});
    axis::category<std::string> b;
    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = axis::category<std::string>{{B, A, C}};
    BOOST_TEST_NE(a, b);
    b = a;
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::category<std::string> c = std::move(b);
    BOOST_TEST_EQ(c, a);
    BOOST_TEST_NE(b, a);
    axis::category<std::string> d;
    BOOST_TEST_NE(c, d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(a(A), 0);
    BOOST_TEST_EQ(a(B), 1);
    BOOST_TEST_EQ(a(C), 2);
    BOOST_TEST_EQ(a(other), 3);
    BOOST_TEST_EQ(a.value(0), A);
    BOOST_TEST_EQ(a.value(1), B);
    BOOST_TEST_EQ(a.value(2), C);
    BOOST_TEST_THROWS(a.value(3), std::out_of_range);
  }

  // iterators
  {
    enum { A, B, C };
    test_axis_iterator(axis::circular<>(5, 0, 1, ""), 0, 5);
    test_axis_iterator(axis::variable<>({1, 2, 3}, ""), 0, 2);
    test_axis_iterator(axis::integer<>(0, 4, ""), 0, 4);
    test_axis_iterator(axis::category<>({A, B, C}, ""), 0, 3);
    test_axis_iterator(axis::variant<axis::regular<>>(axis::regular<>(5, 0, 1)), 0, 5);
    // BOOST_TEST_THROWS(axis::variant<axis::category<>>(axis::category<>({A, B,
    // C}))[0].lower(),
    //                   std::runtime_error);
  }

  return boost::report_errors();
}
