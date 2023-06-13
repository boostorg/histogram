// CopyR 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/int_resolver.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/axis/piece.hpp>
#include <boost/histogram/axis/piecewise.hpp>
#include <limits>
#include <sstream>
#include <type_traits>
#include "axis.hpp"
#include "is_close.hpp"
#include "std_ostream.hpp"
#include "str.hpp"
#include "throw_exception.hpp"

#include <cmath>
#include <iostream>
#include <numeric>

using namespace boost::histogram;

int main() {
  // int_resolver_circular
  {
    using type_multiply = axis::piece_multiply<double>;

    // Has x0 = 4, x1 = 5, x2 = 7, x3 = 11, x4 = 17
    const int k_N = 3;
    const double k_x0 = 3.0;
    const double k_b0 = 1.0;
    const double k_r = 2.0;

    const auto p = type_multiply::create(k_N, k_x0, k_b0, k_r);
    const auto a = axis::int_resolver_circular<double, type_multiply>(p, 2, 22);

    BOOST_TEST_EQ(a.index(3), 0);
    BOOST_TEST_EQ(a.index(4), 1);
    BOOST_TEST_EQ(a.index(6), 2);
    BOOST_TEST_EQ(a.index(10), 3);
    BOOST_TEST_EQ(a.index(18), 4);
    // BOOST_TEST_EQ(a.index(5), k_x0 + k_b0);
  }

  // int_resolver_linear
  {
    using type_uniform = axis::piece_uniform<double>;
    const type_uniform p_uniform = type_uniform::solve_b0(4, -2, +2);
    const auto a = axis::int_resolver_linear<double, type_uniform>(p_uniform);

    // Copied from axis_regular_test.cpp
    BOOST_TEST_EQ(a.value(0), -2);
    BOOST_TEST_EQ(a.value(1), -1);
    BOOST_TEST_EQ(a.value(2), 0);
    BOOST_TEST_EQ(a.value(3), 1);
    BOOST_TEST_EQ(a.value(4), 2);
    // BOOST_TEST_EQ(a.bin(-1).lower(), -std::numeric_limits<double>::infinity());
    // BOOST_TEST_EQ(a.bin(-1).upper(), -2);
    // BOOST_TEST_EQ(a.bin(a.size()).lower(), 2);
    // BOOST_TEST_EQ(a.bin(a.size()).upper(), std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a.index(-10.), -1);
    BOOST_TEST_EQ(a.index(-2.1), -1);
    BOOST_TEST_EQ(a.index(-2.0), 0);
    BOOST_TEST_EQ(a.index(-1.1), 0);
    BOOST_TEST_EQ(a.index(0.0), 2);
    BOOST_TEST_EQ(a.index(0.9), 2);
    BOOST_TEST_EQ(a.index(1.0), 3);
    BOOST_TEST_EQ(a.index(10.), 4);
    BOOST_TEST_EQ(a.index(-std::numeric_limits<double>::infinity()), -1);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 4);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::quiet_NaN()), 4);
  }

  // with growth
  {
    using pii_t = std::pair<axis::index_type, axis::index_type>;
    using type_uniform = axis::piece_uniform<double>;
    using type_shift = axis::bin_shift_integrator<double, type_uniform>;
    using type_resolver = axis::int_resolver_linear<double, type_shift>;

    const auto p_uniform = type_uniform::solve_b0(1, 0, 1);
    const auto p_shift = type_shift(p_uniform);
    auto a = type_resolver(p_shift);

    // Copied from axis_regular_test.cpp
    BOOST_TEST_EQ(a.size(), 1);
    BOOST_TEST_EQ(a.update(0), pii_t(0, 0));
    BOOST_TEST_EQ(a.size(), 1);
    BOOST_TEST_EQ(a.update(1), pii_t(1, -1));
    BOOST_TEST_EQ(a.size(), 2);
    BOOST_TEST_EQ(a.value(0), 0);
    BOOST_TEST_EQ(a.value(2), 2);
    BOOST_TEST_EQ(a.update(-1), pii_t(0, 1));
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(a.value(0), -1);
    BOOST_TEST_EQ(a.value(3), 2);
    BOOST_TEST_EQ(a.update(-10), pii_t(0, 9));
    BOOST_TEST_EQ(a.size(), 12);
    BOOST_TEST_EQ(a.value(0), -10);
    BOOST_TEST_EQ(a.value(12), 2);
    BOOST_TEST_EQ(a.update(std::numeric_limits<double>::infinity()), pii_t(a.size(), 0));
    BOOST_TEST_EQ(a.update(std::numeric_limits<double>::quiet_NaN()), pii_t(a.size(), 0));
    BOOST_TEST_EQ(a.update(-std::numeric_limits<double>::infinity()), pii_t(-1, 0));
  }

  return boost::report_errors();
}
