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
#include <boost/histogram/histogram_fwd.hpp>
#include <tuple>
#include <vector>
#include "utility_meta.hpp"

using namespace boost::histogram;

int main() {
  BOOST_TEST_EQ(detail::cat("foo", 1, "bar"), "foo1bar");

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

  // sub_axes
  {
    using ra = axis::regular<>;
    using ia = axis::integer<>;
    using ca = axis::category<>;
    using T = std::tuple<ra, ia, ca>;
    BOOST_TEST_TRAIT_TRUE((std::is_same<detail::sub_axes<T, i0>, std::tuple<ra>>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<detail::sub_axes<T, i1>, std::tuple<ia>>));
    BOOST_TEST_TRAIT_TRUE((std::is_same<detail::sub_axes<T, i2>, std::tuple<ca>>));
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<detail::sub_axes<T, i0, i1, i2>, std::tuple<ra, ia, ca>>));
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<detail::sub_axes<T, i0, i1>, std::tuple<ra, ia>>));
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<detail::sub_axes<T, i0, i2>, std::tuple<ra, ca>>));
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<detail::sub_axes<T, i1, i2>, std::tuple<ia, ca>>));
  }

  // make_sub_tuple
  {
    using ia = axis::integer<>;
    using T = std::tuple<ia, ia, ia>;
    auto axes = T(ia(0, 1), ia(1, 2), ia(2, 3));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, i1(), i2()),
                  (std::tuple<ia, ia>(ia(1, 2), ia(2, 3))));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, i0(), i1()),
                  (std::tuple<ia, ia>(ia(0, 1), ia(1, 2))));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, i1()), (std::tuple<ia>(ia(1, 2))));
    BOOST_TEST_EQ(detail::make_sub_axes(axes, i0(), i1(), i2()), axes);
  }

  return boost::report_errors();
}
