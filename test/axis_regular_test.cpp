// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/units/quantity.hpp>
#include <boost/units/systems/si/length.hpp>
#include <limits>
#include <sstream>
#include <string>
#include "utility.hpp"

using namespace boost::histogram;

int main() {
  // bad_ctors
  {
    BOOST_TEST_THROWS(axis::regular<>(1, 0, 0), std::invalid_argument);
    BOOST_TEST_THROWS(axis::regular<>(0, 0, 1), std::invalid_argument);
  }

  // axis::regular
  {
    axis::regular<> a{4, -2, 2};
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<double>::infinity());
    axis::regular<> b;
    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::regular<> c = std::move(b);
    BOOST_TEST_EQ(c, a);
    axis::regular<> d;
    BOOST_TEST_NE(c, d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a(-10.), -1);
    BOOST_TEST_EQ(a(-2.1), -1);
    BOOST_TEST_EQ(a(-2.0), 0);
    BOOST_TEST_EQ(a(-1.1), 0);
    BOOST_TEST_EQ(a(0.0), 2);
    BOOST_TEST_EQ(a(0.9), 2);
    BOOST_TEST_EQ(a(1.0), 3);
    BOOST_TEST_EQ(a(10.), 4);
    BOOST_TEST_EQ(a(-std::numeric_limits<double>::infinity()), -1);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::infinity()), 4);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::quiet_NaN()), 4);
  }

  // regular axis with inverted range
  {
    axis::regular<> a{2, 1, -2};
    BOOST_TEST_EQ(a[-1].lower(), std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[0].lower(), 1);
    BOOST_TEST_EQ(a[1].lower(), -0.5);
    BOOST_TEST_EQ(a[2].lower(), -2);
    BOOST_TEST_EQ(a[2].upper(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a(2), -1);
    BOOST_TEST_EQ(a(1.001), -1);
    BOOST_TEST_EQ(a(1), 0);
    BOOST_TEST_EQ(a(0), 0);
    BOOST_TEST_EQ(a(-0.499), 0);
    BOOST_TEST_EQ(a(-0.5), 1);
    BOOST_TEST_EQ(a(-1), 1);
    BOOST_TEST_EQ(a(-2), 2);
    BOOST_TEST_EQ(a(-20), 2);
  }

  // axis::regular with log transform
  {
    axis::regular<axis::transform::log<>> b{2, 1e0, 1e2};
    BOOST_TEST_EQ(b[-1].lower(), 0.0);
    BOOST_TEST_IS_CLOSE(b[0].lower(), 1.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[1].lower(), 10.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[2].lower(), 100.0, 1e-9);
    BOOST_TEST_EQ(b[2].upper(), std::numeric_limits<double>::infinity());

    BOOST_TEST_EQ(b(-1), 2); // produces NaN in conversion
    BOOST_TEST_EQ(b(0), -1);
    BOOST_TEST_EQ(b(1), 0);
    BOOST_TEST_EQ(b(9), 0);
    BOOST_TEST_EQ(b(10), 1);
    BOOST_TEST_EQ(b(90), 1);
    BOOST_TEST_EQ(b(100), 2);
    BOOST_TEST_EQ(b(std::numeric_limits<double>::infinity()), 2);
  }

  // axis::regular with sqrt transform
  {
    axis::regular<axis::transform::sqrt<>> b{2, 0, 4};
    // this is weird: -inf * -inf = inf, thus the lower bound
    BOOST_TEST_EQ(b[-1].lower(), std::numeric_limits<double>::infinity());
    BOOST_TEST_IS_CLOSE(b[0].lower(), 0.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[1].lower(), 1.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[2].lower(), 4.0, 1e-9);
    BOOST_TEST_EQ(b[2].upper(), std::numeric_limits<double>::infinity());

    BOOST_TEST_EQ(b(-1), 2); // produces NaN in conversion
    BOOST_TEST_EQ(b(0), 0);
    BOOST_TEST_EQ(b(0.99), 0);
    BOOST_TEST_EQ(b(1), 1);
    BOOST_TEST_EQ(b(3.99), 1);
    BOOST_TEST_EQ(b(4), 2);
    BOOST_TEST_EQ(b(100), 2);
    BOOST_TEST_EQ(b(std::numeric_limits<double>::infinity()), 2);
  }

  // axis::regular with quantity
  {
    using namespace boost::units;
    using namespace boost::units::si;
    using Q = quantity<length>;

    axis::regular<axis::transform::unit<Q>> b(2, 0 * meter, 2 * meter);
    BOOST_TEST_EQ(b[-1].lower() / meter, -std::numeric_limits<double>::infinity());
    BOOST_TEST_IS_CLOSE(b[0].lower() / meter, 0.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[1].lower() / meter, 1.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[2].lower() / meter, 2.0, 1e-9);
    BOOST_TEST_EQ(b[2].upper() / meter, std::numeric_limits<double>::infinity());

    BOOST_TEST_EQ(b(-1 * meter), -1); // produces NaN in conversion
    BOOST_TEST_EQ(b(0 * meter), 0);
    BOOST_TEST_EQ(b(0.99 * meter), 0);
    BOOST_TEST_EQ(b(1 * meter), 1);
    BOOST_TEST_EQ(b(1.99 * meter), 1);
    BOOST_TEST_EQ(b(2 * meter), 2);
    BOOST_TEST_EQ(b(100 * meter), 2);
    BOOST_TEST_EQ(b(std::numeric_limits<double>::infinity() * meter), 2);
  }

  // iterators
  {
    test_axis_iterator(axis::regular<>(5, 0, 1, "", axis::option_type::none), 0, 5);
    test_axis_iterator(
        axis::regular<>(5, 0, 1, "", axis::option_type::underflow_and_overflow), 0, 5);
  }

  // bin_type streamable
  {
    auto test = [](const auto& x, const char* ref) {
      std::ostringstream os;
      os << x;
      BOOST_TEST_EQ(os.str(), std::string(ref));
    };

    auto a = axis::regular<>(2, 0, 1);
    test(a[0], "[0, 0.5)");
  }

  return boost::report_errors();
}
