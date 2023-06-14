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
#include <numeric>

using namespace boost::histogram;

int main() {
  // int_resolver_circular
  {
    using type_multiply = axis::piece_multiply<double>;

    // Has x0 = 3, x1 = 4, x2 = 6, x3 = 10, x4 = 18
    const int k_N = 3;
    const double k_x0 = 3.0;
    const double k_b0 = 1.0;
    const double k_r = 2.0;

    const auto p = type_multiply::create(k_N, k_x0, k_b0, k_r);
    const auto a = axis::int_resolver_circular<double, type_multiply>(p, 2, 22);

    // Has a period of 20 based on the given bounds 2 and 22
    for (int i = -200; i < 200; i += 20) {
      BOOST_TEST_EQ(a.index(i + 3), 0);
      BOOST_TEST_EQ(a.index(i + 4), 1);
      BOOST_TEST_EQ(a.index(i + 6), 2);
      BOOST_TEST_EQ(a.index(i + 10), 3);
      BOOST_TEST_EQ(a.index(i + 18), 4);
    }
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

  // Test based on the example https://godbolt.org/z/oaPo6n17h
  // from @vakokako in #336
  {
    auto fn_test_precision = [](int N, double x0, double xN, auto fn_axis) {
      const auto a = fn_axis(N, x0, xN);

      // // Calculate bin spacing b0
      const double b0 = (xN - x0) / N;

      // Check to see if the index and value calculations are exact
      for (int y = 0; y < a.size(); ++y) {
        const double x = x0 + y * b0;
        BOOST_TEST_EQ(y, a.index(x));
        BOOST_TEST_EQ((double)(x), a.value(y));
      }
    };

    auto fn_test_piece = [](int N, double x0, double xN) {
      const auto p_uniform = axis::piece_uniform<double>::solve_b0(N, x0, xN);
      return axis::int_resolver_linear<double, axis::piece_uniform<double>>(p_uniform);
    };

#if 0
    auto fn_test_regular = [](int N, double x0, double xN){
      return boost::histogram::axis::regular<double>(N, x0, xN);
    };
    fn_test_precision(27000, 0, 27000, fn_test_regular); // Fails
#endif
    fn_test_precision(27000, 0, 27000, fn_test_piece); // Passes

    // Bin spacings and starting points that take few floating point bits to represent
    const std::vector<double> v_spacing = {0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.625, 7.25};
    const std::vector<double> v_x0 = {-1000.25, -2.5, -0.5, 0, 0.5, 2.5, 1000.25};

    for (const int n : {1, 16, 27000, 350017, 1234567}) {
      for (const double spacing : v_spacing) {
        for (const double x0 : v_x0) {
          fn_test_precision(n, x0, x0 + n * spacing, fn_test_piece);
        }
      }
    }
  }

  return boost::report_errors();
}
