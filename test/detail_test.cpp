// Copyright 2015-2017 Hans Dembinski
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
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/literals.hpp>
#include <tuple>
#include <vector>
#include "utility_meta.hpp"

using namespace boost::histogram;
using namespace boost::histogram::literals;

int main() {
  BOOST_TEST_EQ(detail::cat("foo", 1, "bar"), "foo1bar");

  // literals
  {
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<std::integral_constant<unsigned, 0>, decltype(0_c)>));
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<std::integral_constant<unsigned, 3>, decltype(3_c)>));
    BOOST_TEST_EQ(decltype(10_c)::value, 10);
    BOOST_TEST_EQ(decltype(213_c)::value, 213);
  }

  // axes_size
  {
    std::tuple<int, int> a;
    std::vector<int> b(3);
    std::array<int, 4> c;
    BOOST_TEST_EQ(detail::axes_size(a), 2);
    BOOST_TEST_EQ(detail::axes_size(b), 3);
    BOOST_TEST_EQ(detail::axes_size(c), 4);
  }

  // sequence equality
  {
    using R = axis::regular<>;
    using I = axis::integer<>;
    using V = axis::variable<>;
    auto r = R(2, -1, 1);
    auto i = I(-1, 1);
    auto v = V{-1, 0, 1};

    std::vector<axis::variant<R, I, V>> v1 = {r, i};
    std::vector<axis::variant<R, I>> v2 = {r, i};
    std::vector<axis::variant<R, I>> v3 = {i, r};
    std::vector<axis::variant<I, R>> v4 = {r, i};
    std::vector<axis::variant<R, I>> v5 = {r, r};
    std::vector<R> v6 = {r, r};

    BOOST_TEST(detail::axes_equal(v1, v2));
    BOOST_TEST(detail::axes_equal(v1, v4));
    BOOST_TEST(detail::axes_equal(v5, v6));
    BOOST_TEST_NOT(detail::axes_equal(v1, v3));
    BOOST_TEST_NOT(detail::axes_equal(v2, v3));
    BOOST_TEST_NOT(detail::axes_equal(v3, v4));
    BOOST_TEST_NOT(detail::axes_equal(v1, v5));

    auto t1 = std::make_tuple(r, i);
    auto t2 = std::make_tuple(i, r);
    auto t3 = std::make_tuple(v, i);
    auto t4 = std::make_tuple(r, r);

    BOOST_TEST(detail::axes_equal(t1, v1));
    BOOST_TEST(detail::axes_equal(t1, v2));
    BOOST_TEST(detail::axes_equal(t1, v4));
    BOOST_TEST(detail::axes_equal(v1, t1));
    BOOST_TEST(detail::axes_equal(v2, t1));
    BOOST_TEST(detail::axes_equal(v4, t1));
    BOOST_TEST(detail::axes_equal(t2, v3));
    BOOST_TEST(detail::axes_equal(v3, t2));
    BOOST_TEST(detail::axes_equal(t4, v5));
    BOOST_TEST(detail::axes_equal(t4, v6));
    BOOST_TEST_NOT(detail::axes_equal(t1, t2));
    BOOST_TEST_NOT(detail::axes_equal(t2, t3));
    BOOST_TEST_NOT(detail::axes_equal(t1, v3));
    BOOST_TEST_NOT(detail::axes_equal(t1, v3));
    BOOST_TEST_NOT(detail::axes_equal(t3, v1));
    BOOST_TEST_NOT(detail::axes_equal(t3, v2));
    BOOST_TEST_NOT(detail::axes_equal(t3, v3));
    BOOST_TEST_NOT(detail::axes_equal(t3, v4));
  }

  // sequence assign
  {
    using R = axis::regular<>;
    using I = axis::integer<>;
    using V = axis::variable<>;
    auto r = R(2, -1, 1);
    auto i = I(-1, 1);
    auto v = V{-1, 0, 1};

    std::vector<axis::variant<R, V, I>> v1 = {r, i};
    std::vector<axis::variant<I, R>> v2;
    std::vector<R> v3 = {r, r};

    BOOST_TEST_NOT(detail::axes_equal(v2, v1));
    detail::axes_assign(v2, v1);
    BOOST_TEST(detail::axes_equal(v2, v1));
    detail::axes_assign(v2, v3);
    BOOST_TEST(detail::axes_equal(v2, v3));

    auto t1 = std::make_tuple(r);
    detail::axes_assign(v3, t1);
    BOOST_TEST(detail::axes_equal(v3, t1));

    auto t2 = std::make_tuple(r, i);
    detail::axes_assign(v2, t2);
    BOOST_TEST(detail::axes_equal(v2, t2));

    auto t3 = std::make_tuple(R{3, -1, 1}, i);
    BOOST_TEST_NOT(detail::axes_equal(t2, t3));
    detail::axes_assign(t2, t3);
    BOOST_TEST(detail::axes_equal(t2, t3));
  }

  // make_sub_axes
  {
    using I = axis::integer<>;
    using boost::mp11::mp_list;

    I a0(0, 1), a1(1, 2), a2(2, 3);
    auto ax = std::make_tuple(a0, a1, a2);
    BOOST_TEST_EQ(detail::make_sub_axes(ax, 0_c), std::make_tuple(a0));
    BOOST_TEST_EQ(detail::make_sub_axes(ax, 1_c), std::make_tuple(a1));
    BOOST_TEST_EQ(detail::make_sub_axes(ax, 2_c), std::make_tuple(a2));
    BOOST_TEST_EQ(detail::make_sub_axes(ax, 0_c, 1_c), std::make_tuple(a0, a1));
    BOOST_TEST_EQ(detail::make_sub_axes(ax, 0_c, 2_c), std::make_tuple(a0, a2));
    BOOST_TEST_EQ(detail::make_sub_axes(ax, 1_c, 2_c), std::make_tuple(a1, a2));
    BOOST_TEST_EQ(detail::make_sub_axes(ax, 0_c, 1_c, 2_c), std::make_tuple(a0, a1, a2));
  }

  return boost::report_errors();
}
