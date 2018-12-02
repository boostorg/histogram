// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/axis/category.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/is_set.hpp>
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

  // sequence equality
  {
    enum { A, B, C };
    std::vector<axis::variant<axis::regular<>, axis::variable<>, axis::category<>,
                              axis::integer<>>>
        std_vector1 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{A, B, C}};

    std::vector<axis::variant<axis::regular<>, axis::variable<>, axis::category<>>>
        std_vector2 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{{A, B, C}}};

    std::vector<axis::variant<axis::regular<>, axis::variable<>>> std_vector3 = {
        axis::variable<>{-1, 0, 1}, axis::regular<>{2, -1, 1}};

    std::vector<axis::variant<axis::variable<>, axis::regular<>>> std_vector4 = {
        axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1}};

    BOOST_TEST(detail::axes_equal(std_vector1, std_vector2));
    BOOST_TEST_NOT(detail::axes_equal(std_vector2, std_vector3));
    BOOST_TEST_NOT(detail::axes_equal(std_vector3, std_vector4));

    auto tuple1 = std::make_tuple(axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                                  axis::category<>{{A, B, C}});

    auto tuple2 = std::make_tuple(axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                                  axis::category<>{{A, B}});

    auto tuple3 = std::make_tuple(axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1});

    BOOST_TEST(detail::axes_equal(std_vector1, tuple1));
    BOOST_TEST(detail::axes_equal(tuple1, std_vector1));
    BOOST_TEST_NOT(detail::axes_equal(tuple1, tuple2));
    BOOST_TEST_NOT(detail::axes_equal(tuple2, tuple3));
    BOOST_TEST_NOT(detail::axes_equal(std_vector3, tuple3));
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

  // sequence assign
  {
    enum { A, B, C, D };
    std::vector<axis::variant<axis::regular<>, axis::variable<>, axis::category<>,
                              axis::integer<>>>
        std_vector1 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{A, B, C}};

    std::vector<axis::variant<axis::regular<>, axis::variable<>, axis::category<>>>
        std_vector2 = {axis::regular<>{2, -2, 2}, axis::variable<>{-2, 0, 2},
                       axis::category<>{A, B}};

    detail::axes_assign(std_vector2, std_vector1);
    BOOST_TEST(detail::axes_equal(std_vector2, std_vector1));

    auto tuple1 = std::make_tuple(axis::regular<>{2, -3, 3}, axis::variable<>{-3, 0, 3},
                                  axis::category<>{A, B, C, D});

    detail::axes_assign(tuple1, std_vector1);
    BOOST_TEST(detail::axes_equal(tuple1, std_vector1));

    decltype(std_vector1) std_vector3;
    BOOST_TEST_NOT(detail::axes_equal(std_vector3, tuple1));
    detail::axes_assign(std_vector3, tuple1);
    BOOST_TEST(detail::axes_equal(std_vector3, tuple1));

    auto tuple2 = std::make_tuple(axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                                  axis::category<>{A, B});

    detail::axes_assign(tuple2, tuple1);
    BOOST_TEST(detail::axes_equal(tuple2, tuple1));
  }

  // make_sub_axes
  {
    using boost::mp11::mp_list;
    axis::integer<> a0(0, 1), a1(1, 2), a2(2, 3);
    auto axes = std::make_tuple(a0, a1, a2);
    BOOST_TEST_EQ(detail::make_sub_axes(axes, 0_c), std::make_tuple(a0));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, 1_c), std::make_tuple(a1));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, 2_c), std::make_tuple(a2));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, 0_c, 1_c), std::make_tuple(a0, a1));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, 0_c, 2_c), std::make_tuple(a0, a2));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, 1_c, 2_c), std::make_tuple(a1, a2));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, 0_c, 1_c, 2_c),
                  std::make_tuple(a0, a1, a2));
  }

  // is_set
  {
    std::vector<int> v({3, 1, 2});
    BOOST_TEST(detail::is_set(v.begin(), v.end()));
    v = {3, 1, 2, 1};
    BOOST_TEST_NOT(detail::is_set(v.begin(), v.end()));
  }

  return boost::report_errors();
}
