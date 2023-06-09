// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/ostream.hpp>
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

int main() {
  using namespace boost::histogram;
  // using def = use_default;
  // namespace tr = axis::transform;
  // namespace op = axis::option;

  // BOOST_TEST(std::is_nothrow_move_assignable<axis::regular<>>::value);
  // BOOST_TEST(std::is_nothrow_move_constructible<axis::regular<>>::value);

  BOOST_TEST(std::is_nothrow_move_assignable<axis::unit_regular<>>::value);
  BOOST_TEST(std::is_nothrow_move_constructible<axis::unit_regular<>>::value);

  // Construct rollout right
  {
    const auto p = axis::piece<>::rollout(5, axis::x_start(2.0), axis::width_start(1.1),
                                          axis::bin_trans::multiply(1.2));
    BOOST_TEST_EQ(p.size(), 5);
    BOOST_TEST(p.bt().is_multiply());
    BOOST_TEST_EQ(p.bt().value(), 1.2);

    BOOST_TEST_EQ(p.b0(), 1.1);
    BOOST_TEST_EQ(p.x0(), 2.0);
    BOOST_TEST_EQ(p.xN(), p.x0() + p.b0() * (1 + 1.2 + std::pow(1.2, 2) +
                                             std::pow(1.2, 3) + std::pow(1.2, 4)));

    BOOST_TEST_EQ(p.size_ic(), 5);
    BOOST_TEST_EQ(p.size(), 5);
    BOOST_TEST_EQ(p.accumulated_left_shift(), 0);
    BOOST_TEST_EQ(p.accumulated_right_shift(), 0);

    const auto b0_check = axis::piece<>::calc_b0(p.x0(), p.xN(), p.size_ic(), p.bt());
    BOOST_TEST_IS_CLOSE(b0_check.value(), p.b0(), 1e-12);

    BOOST_TEST_EQ(p.forward(2.0), 0.0);
    BOOST_TEST_EQ(p.forward(2.0 + 1.1), 1.0);
    BOOST_TEST_EQ(p.forward(2.0 + 1.1 * (1 + 1.2)), 2.0);
    BOOST_TEST_EQ(p.forward(2.0 + 1.1 * (1 + 1.2 + std::pow(1.2, 2))), 3.0);

    for (int i = -100; i < 100; ++i) {
      const auto d = i * 0.021;
      BOOST_TEST_IS_CLOSE(p.reverse(p.forward(d)), d, 1e-12);
    }
  }

  // Construct rollout left
  {
    const auto p = axis::piece<>::rollout(-5, axis::x_start(2.0), axis::width_start(1.1),
                                          axis::bin_trans::multiply(1.2));
    BOOST_TEST_EQ(p.size(), 5);
    BOOST_TEST(p.bt().is_multiply());
    BOOST_TEST_EQ(p.bt().value(), 1 / 1.2);

    // const auto bt_rev = axis::bin_trans::multiply(1 / 1.2);
    BOOST_TEST_IS_CLOSE(p.b0(), 1.1 * std::pow(1.2, 4), 1e-12);
    BOOST_TEST_IS_CLOSE(p.xN(), 2.0, 1e-12);

    BOOST_TEST_IS_CLOSE(
        p.x0(),
        p.xN() - 1.1 * (1 + 1.2 + std::pow(1.2, 2) + std::pow(1.2, 3) + std::pow(1.2, 4)),
        1e-12);

    // const auto bt = p.bt();
    // BOOST_TEST(bt.is_multiply());
    // BOOST_TEST_EQ(bt.value(), 1.2);

    // BOOST_TEST_EQ(p.size_ic(), 5);
    // BOOST_TEST_EQ(p.size(), 5);
    // BOOST_TEST_EQ(p.accumulated_left_shift(), 0);
    // BOOST_TEST_EQ(p.accumulated_right_shift(), 0);

    // std::cout << "p.bt().value(): " << p.bt().value() << std::endl;
    // std::cout << "b0:             " << p.b0() << std::endl;
    // std::cout << "x0:             " << p.x0() << std::endl;
    // std::cout << "xN:             " << p.xN() << std::endl;

    // for (int i = 1; i < 6; ++i) {
    //   std::cout << "bin_width: " << p.calc_bin_width(i) << std::endl;
    // }

    // assert(false);
  }

  // Construct piecewise
  {
    const auto p = axis::piece<>::rollout(-5, axis::x_start(2.0), axis::width_start(1.1),
                                          axis::bin_trans::multiply(1.2));
    auto pw = axis::piecewise(p);
  }

  // // bad_ctors
  // {
  //   BOOST_TEST_THROWS(axis::unit_regular<>(1, 0, 0), std::invalid_argument);
  //   BOOST_TEST_THROWS(axis::unit_regular<>(0, 0, 1), std::invalid_argument);
  // }

  // // ctors and assignment
  // {
  //   axis::unit_regular<> a{4, -2, 2};
  //   axis::unit_regular<> b;
  //   BOOST_TEST_NE(a, b);
  //   b = a;
  //   BOOST_TEST_EQ(a, b);
  //   axis::unit_regular<> c = std::move(b);
  //   BOOST_TEST_EQ(c, a);
  //   axis::unit_regular<> d;
  //   BOOST_TEST_NE(c, d);
  //   d = std::move(c);
  //   BOOST_TEST_EQ(d, a);
  // }

  // // input, output
  // {
  //   axis::unit_regular<> a{4, -2, 2, "foo"};
  //   BOOST_TEST_EQ(a.metadata(), "foo");
  //   const auto& cref = a;
  //   BOOST_TEST_EQ(cref.metadata(), "foo");
  //   cref.metadata() = "bar"; // this is allowed
  //   BOOST_TEST_EQ(cref.metadata(), "bar");
  //   BOOST_TEST_EQ(a.value(0), -2);
  //   BOOST_TEST_EQ(a.value(1), -1);
  //   BOOST_TEST_EQ(a.value(2), 0);
  //   BOOST_TEST_EQ(a.value(3), 1);
  //   BOOST_TEST_EQ(a.value(4), 2);
  //   BOOST_TEST_EQ(a.bin(-1).lower(), -std::numeric_limits<double>::infinity());
  //   BOOST_TEST_EQ(a.bin(-1).upper(), -2);
  //   BOOST_TEST_EQ(a.bin(a.size()).lower(), 2);
  //   BOOST_TEST_EQ(a.bin(a.size()).upper(), std::numeric_limits<double>::infinity());
  //   BOOST_TEST_EQ(a.index(-10.), -1);
  //   BOOST_TEST_EQ(a.index(-2.1), -1);
  //   BOOST_TEST_EQ(a.index(-2.0), 0);
  //   BOOST_TEST_EQ(a.index(-1.1), 0);
  //   BOOST_TEST_EQ(a.index(0.0), 2);
  //   BOOST_TEST_EQ(a.index(0.9), 2);
  //   BOOST_TEST_EQ(a.index(1.0), 3);
  //   BOOST_TEST_EQ(a.index(10.), 4);
  //   BOOST_TEST_EQ(a.index(-std::numeric_limits<double>::infinity()), -1);
  //   BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 4);
  //   BOOST_TEST_EQ(a.index(std::numeric_limits<double>::quiet_NaN()), 4);

  //   BOOST_TEST_EQ(str(a),
  //                 "regular(4, -2, 2, metadata=\"bar\", options=underflow | overflow)");
  // }

  // // with inverted range
  // {
  //   axis::unit_regular<> a{2, 1, -2};
  //   BOOST_TEST_EQ(a.bin(-1).lower(), std::numeric_limits<double>::infinity());
  //   BOOST_TEST_EQ(a.bin(0).lower(), 1);
  //   BOOST_TEST_EQ(a.bin(1).lower(), -0.5);
  //   BOOST_TEST_EQ(a.bin(2).lower(), -2);
  //   BOOST_TEST_EQ(a.bin(2).upper(), -std::numeric_limits<double>::infinity());
  //   BOOST_TEST_EQ(a.index(2), -1);
  //   BOOST_TEST_EQ(a.index(1.001), -1);
  //   BOOST_TEST_EQ(a.index(1), 0);
  //   BOOST_TEST_EQ(a.index(0), 0);
  //   BOOST_TEST_EQ(a.index(-0.499), 0);
  //   BOOST_TEST_EQ(a.index(-0.5), 1);
  //   BOOST_TEST_EQ(a.index(-1), 1);
  //   BOOST_TEST_EQ(a.index(-2), 2);
  //   BOOST_TEST_EQ(a.index(-20), 2);
  // }

  // // with log transform
  // {
  //   auto a = axis::unit_regular<double, tr::log>{2, 1e0, 1e2};
  //   BOOST_TEST_EQ(a.bin(-1).lower(), 0.0);
  //   BOOST_TEST_IS_CLOSE(a.bin(0).lower(), 1.0, 1e-9);
  //   BOOST_TEST_IS_CLOSE(a.bin(1).lower(), 10.0, 1e-9);
  //   BOOST_TEST_IS_CLOSE(a.bin(2).lower(), 100.0, 1e-9);
  //   BOOST_TEST_EQ(a.bin(2).upper(), std::numeric_limits<double>::infinity());

  //   BOOST_TEST_EQ(a.index(-1), 2); // produces NaN in conversion
  //   BOOST_TEST_EQ(a.index(0), -1);
  //   BOOST_TEST_EQ(a.index(1), 0);
  //   BOOST_TEST_EQ(a.index(9), 0);
  //   BOOST_TEST_EQ(a.index(10), 1);
  //   BOOST_TEST_EQ(a.index(90), 1);
  //   BOOST_TEST_EQ(a.index(100), 2);
  //   BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 2);

  //   BOOST_TEST_THROWS((axis::unit_regular<double, tr::log>{2, -1, 0}),
  //   std::invalid_argument);

  //   BOOST_TEST_CSTR_EQ(
  //       str(a).c_str(),
  //       "regular(transform::log{}, 2, 1, 100, options=underflow | overflow)");
  // }

  // // with sqrt transform
  // {
  //   axis::unit_regular<double, tr::sqrt> a(2, 0, 4);
  //   // this is weird, but -inf * -inf = inf, thus the lower bound
  //   BOOST_TEST_EQ(a.bin(-1).lower(), std::numeric_limits<double>::infinity());
  //   BOOST_TEST_IS_CLOSE(a.bin(0).lower(), 0.0, 1e-9);
  //   BOOST_TEST_IS_CLOSE(a.bin(1).lower(), 1.0, 1e-9);
  //   BOOST_TEST_IS_CLOSE(a.bin(2).lower(), 4.0, 1e-9);
  //   BOOST_TEST_EQ(a.bin(2).upper(), std::numeric_limits<double>::infinity());

  //   BOOST_TEST_EQ(a.index(-1), 2); // produces NaN in conversion
  //   BOOST_TEST_EQ(a.index(0), 0);
  //   BOOST_TEST_EQ(a.index(0.99), 0);
  //   BOOST_TEST_EQ(a.index(1), 1);
  //   BOOST_TEST_EQ(a.index(3.99), 1);
  //   BOOST_TEST_EQ(a.index(4), 2);
  //   BOOST_TEST_EQ(a.index(100), 2);
  //   BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 2);

  //   BOOST_TEST_EQ(str(a),
  //                 "regular(transform::sqrt{}, 2, 0, 4, options=underflow | overflow)");
  // }

  // // with pow transform
  // {
  //   axis::unit_regular<double, tr::pow> a(tr::pow{0.5}, 2, 0, 4);
  //   // this is weird, but -inf * -inf = inf, thus the lower bound
  //   BOOST_TEST_EQ(a.bin(-1).lower(), std::numeric_limits<double>::infinity());
  //   BOOST_TEST_IS_CLOSE(a.bin(0).lower(), 0.0, 1e-9);
  //   BOOST_TEST_IS_CLOSE(a.bin(1).lower(), 1.0, 1e-9);
  //   BOOST_TEST_IS_CLOSE(a.bin(2).lower(), 4.0, 1e-9);
  //   BOOST_TEST_EQ(a.bin(2).upper(), std::numeric_limits<double>::infinity());

  //   BOOST_TEST_EQ(a.index(-1), 2); // produces NaN in conversion
  //   BOOST_TEST_EQ(a.index(0), 0);
  //   BOOST_TEST_EQ(a.index(0.99), 0);
  //   BOOST_TEST_EQ(a.index(1), 1);
  //   BOOST_TEST_EQ(a.index(3.99), 1);
  //   BOOST_TEST_EQ(a.index(4), 2);
  //   BOOST_TEST_EQ(a.index(100), 2);
  //   BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 2);

  //   BOOST_TEST_EQ(str(a),
  //                 "regular(transform::pow{0.5}, 2, 0, 4, options=underflow |
  //                 overflow)");
  // }

  // // with step
  // {
  //   axis::unit_regular<> a(axis::step(0.5), 1, 3);
  //   BOOST_TEST_EQ(a.size(), 4);
  //   BOOST_TEST_EQ(a.bin(-1).lower(), -std::numeric_limits<double>::infinity());
  //   BOOST_TEST_EQ(a.value(0), 1);
  //   BOOST_TEST_EQ(a.value(1), 1.5);
  //   BOOST_TEST_EQ(a.value(2), 2);
  //   BOOST_TEST_EQ(a.value(3), 2.5);
  //   BOOST_TEST_EQ(a.value(4), 3);
  //   BOOST_TEST_EQ(a.bin(4).upper(), std::numeric_limits<double>::infinity());

  //   axis::unit_regular<> b(axis::step(0.5), 1, 3.1);
  //   BOOST_TEST_EQ(a, b);
  // }

  // // with circular option
  // {
  //   axis::circular<> a{4, 0, 1};
  //   BOOST_TEST_EQ(a.bin(-1).lower(), a.bin(a.size() - 1).lower() - 1);
  //   BOOST_TEST_EQ(a.index(-1.0 * 3), 0);
  //   BOOST_TEST_EQ(a.index(0.0), 0);
  //   BOOST_TEST_EQ(a.index(0.25), 1);
  //   BOOST_TEST_EQ(a.index(0.5), 2);
  //   BOOST_TEST_EQ(a.index(0.75), 3);
  //   BOOST_TEST_EQ(a.index(1.0), 0);
  //   BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 4);
  //   BOOST_TEST_EQ(a.index(-std::numeric_limits<double>::infinity()), 4);
  //   BOOST_TEST_EQ(a.index(std::numeric_limits<double>::quiet_NaN()), 4);
  // }

  // // with growth
  // {
  //   using pii_t = std::pair<axis::index_type, axis::index_type>;
  //   axis::unit_regular<double, def, def, op::growth_t> a{1, 0, 1};
  //   BOOST_TEST_EQ(a.size(), 1);
  //   BOOST_TEST_EQ(a.update(0), pii_t(0, 0));
  //   BOOST_TEST_EQ(a.size(), 1);
  //   BOOST_TEST_EQ(a.update(1), pii_t(1, -1));
  //   BOOST_TEST_EQ(a.size(), 2);
  //   BOOST_TEST_EQ(a.value(0), 0);
  //   BOOST_TEST_EQ(a.value(2), 2);
  //   BOOST_TEST_EQ(a.update(-1), pii_t(0, 1));
  //   BOOST_TEST_EQ(a.size(), 3);
  //   BOOST_TEST_EQ(a.value(0), -1);
  //   BOOST_TEST_EQ(a.value(3), 2);
  //   BOOST_TEST_EQ(a.update(-10), pii_t(0, 9));
  //   BOOST_TEST_EQ(a.size(), 12);
  //   BOOST_TEST_EQ(a.value(0), -10);
  //   BOOST_TEST_EQ(a.value(12), 2);
  //   BOOST_TEST_EQ(a.update(std::numeric_limits<double>::infinity()), pii_t(a.size(),
  //   0)); BOOST_TEST_EQ(a.update(std::numeric_limits<double>::quiet_NaN()),
  //   pii_t(a.size(), 0));
  //   BOOST_TEST_EQ(a.update(-std::numeric_limits<double>::infinity()), pii_t(-1, 0));
  // }

  // // axis with overflow bin represents open interval
  // {
  //   axis::unit_regular<double, def, def, op::overflow_t> a{2, 0, 1};
  //   BOOST_TEST_EQ(a.index(0), 0);
  //   BOOST_TEST_EQ(a.index(0.49), 0);
  //   BOOST_TEST_EQ(a.index(0.50), 1);
  //   BOOST_TEST_EQ(a.index(0.99), 1);
  //   BOOST_TEST_EQ(a.index(1), 2);   // overflow bin
  //   BOOST_TEST_EQ(a.index(1.1), 2); // overflow bin
  // }

  // // axis without overflow bin represents a closed interval
  // {
  //   axis::unit_regular<double, def, def, op::none_t> a{2, 0, 1};
  //   BOOST_TEST_EQ(a.index(0), 0);
  //   BOOST_TEST_EQ(a.index(0.49), 0);
  //   BOOST_TEST_EQ(a.index(0.50), 1);
  //   BOOST_TEST_EQ(a.index(0.99), 1);
  //   BOOST_TEST_EQ(a.index(1), 1);   // last ordinary bin
  //   BOOST_TEST_EQ(a.index(1.1), 2); // out of range
  // }

  // // iterators
  // {
  //   test_axis_iterator(axis::unit_regular<>(5, 0, 1), 0, 5);
  //   test_axis_iterator(axis::unit_regular<double, def, def, op::none_t>(5, 0, 1), 0,
  //   5); test_axis_iterator(axis::circular<>(5, 0, 1), 0, 5);
  // }

  // // bin_type streamable
  // {
  //   auto test = [](const auto& x, const char* ref) {
  //     std::ostringstream os;
  //     os << x;
  //     BOOST_TEST_EQ(os.str(), std::string(ref));
  //   };

  //   auto a = axis::unit_regular<>(2, 0, 1);
  //   test(a.bin(0), "[0, 0.5)");
  // }

  // // null_type streamable
  // {
  //   auto a = axis::unit_regular<float, def, axis::null_type>(2, 0, 1);
  //   BOOST_TEST_EQ(str(a), "regular(2, 0, 1, options=underflow | overflow)");
  // }

  // // shrink and rebin
  // {
  //   using A = axis::unit_regular<>;
  //   auto a = A(5, 0, 5);
  //   auto b = A(a, 1, 4, 1);
  //   BOOST_TEST_EQ(b.size(), 3);
  //   BOOST_TEST_EQ(b.value(0), 1);
  //   BOOST_TEST_EQ(b.value(3), 4);
  //   auto c = A(a, 0, 4, 2);
  //   BOOST_TEST_EQ(c.size(), 2);
  //   BOOST_TEST_EQ(c.value(0), 0);
  //   BOOST_TEST_EQ(c.value(2), 4);
  //   auto e = A(a, 1, 5, 2);
  //   BOOST_TEST_EQ(e.size(), 2);
  //   BOOST_TEST_EQ(e.value(0), 1);
  //   BOOST_TEST_EQ(e.value(2), 5);
  // }

  // // shrink and rebin with circular option
  // {
  //   using A = axis::circular<>;
  //   auto a = A(4, 1, 5);
  //   BOOST_TEST_THROWS(A(a, 1, 4, 1), std::invalid_argument);
  //   BOOST_TEST_THROWS(A(a, 0, 3, 1), std::invalid_argument);
  //   auto b = A(a, 0, 4, 2);
  //   BOOST_TEST_EQ(b.size(), 2);
  //   BOOST_TEST_EQ(b.value(0), 1);
  //   BOOST_TEST_EQ(b.value(2), 5);
  // }

  return boost::report_errors();
}
