// CopyR 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
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

#include <boost/variant2/variant.hpp>

using namespace boost::histogram;

int main() {
  // Piecewise Connected
  {
    const auto p_uniform = axis::piece_uniform<double>::solve_b0(8, -1.5, +2.5);
    auto p = axis::piecewise<double, axis::piece_variant<double>>(p_uniform);

    // Add 4 bins to the R expanding by 1.1 each time
    p.extrapolate_R<axis::piece_multiply<double>>(4, 1.1);

    BOOST_TEST_EQ(p.size(), 12);
    BOOST_TEST_EQ(p.x0(), -1.5);

    // The bin spacing of the first piece is 0.5, so the first extrapolated piece is 0.5
    // * 1.1
    BOOST_TEST_EQ(p.xN(), 2.5 + 0.5 * (1.1 + std::pow(1.1, 2) + std::pow(1.1, 3) +
                                       std::pow(1.1, 4)));
  }

  // Piecewise Disconnected
  {
    const auto p_arbitrary = axis::piece_variable<double>::create(
        std::vector<double>{-1.5, -1.2, 0.4, 1.2, 2.5});
    auto p = axis::piecewise<double, axis::piece_variant<double>>(p_arbitrary);

    // New piece
    const auto p_uniform = axis::piece_uniform<double>::solve_b0(8, 3.5, +5.5);
    p.add_R_gap_okay(p_uniform);

    BOOST_TEST_EQ(p.size(), 12);
    BOOST_TEST_EQ(p.x0(), -1.5);
    BOOST_TEST_EQ(p.xN(), 5.5);

    // Check indices
    BOOST_TEST_EQ(p.forward(-1.5), 0);
    BOOST_TEST_EQ(p.forward(-1.2), 1);
    BOOST_TEST_EQ(p.forward(0.4), 2);
    BOOST_TEST_EQ(p.forward(1.2), 3);
    BOOST_TEST_EQ(p.forward(2.49999999999), 3);

    BOOST_TEST(std::isnan(p.forward(2.5)));
    BOOST_TEST(std::isnan(p.forward(3.0)));
    BOOST_TEST(std::isnan(p.forward(3.49999999999)));

    BOOST_TEST_EQ(p.forward(3.5), 4);
    BOOST_TEST_EQ(p.forward(4.5), 8);
    BOOST_TEST_EQ(p.forward(5.5), 12);
  }

  return boost::report_errors();
}
