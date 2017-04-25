// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/utility.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/variant.hpp>
#include <limits>

#define BOOST_TEST_NOT(expr) BOOST_TEST(!(expr))
#define BOOST_TEST_IS_CLOSE(a, b, eps) BOOST_TEST(std::abs(a - b) < eps)

template <typename Axis>
void test_real_axis_iterator(Axis &&a, int begin, int end) {
  for (const auto &bin : a) {
    BOOST_TEST_EQ(bin.idx, begin);
    BOOST_TEST_EQ(bin.left, left(a, begin));
    BOOST_TEST_EQ(bin.right, right(a, begin));
    ++begin;
  }
  BOOST_TEST_EQ(begin, end);
}

template <typename Axis>
void test_axis_iterator(const Axis &a, int begin, int end) {
  for (const auto &bin : a) {
    BOOST_TEST_EQ(bin.idx, begin);
    BOOST_TEST_EQ(bin.value, a[begin]);
    ++begin;
  }
  BOOST_TEST_EQ(begin, end);
}

int main() {
  using namespace boost::histogram;
  using axis_t = typename boost::make_variant_over<default_axes>::type;

  // bad_ctors
  {
    BOOST_TEST_THROWS(regular_axis<>(0, 0, 1), std::logic_error);
    BOOST_TEST_THROWS(regular_axis<>(1, 1, -1), std::logic_error);
    BOOST_TEST_THROWS(circular_axis<>(0), std::logic_error);
    BOOST_TEST_THROWS(variable_axis<>({}), std::logic_error);
    BOOST_TEST_THROWS(variable_axis<>({1.0}), std::logic_error);
    BOOST_TEST_THROWS(integer_axis(1, -1), std::logic_error);
    BOOST_TEST_THROWS(category_axis({}), std::logic_error);
  }

  // regular_axis
  {
    regular_axis<> a{4, -2, 2};
    BOOST_TEST_EQ(a[-1], -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.bins() + 1], std::numeric_limits<double>::infinity());
    regular_axis<> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    regular_axis<> c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    regular_axis<> d;
    BOOST_TEST_NOT(c == d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a.index(-10.), -1);
    BOOST_TEST_EQ(a.index(-2.1), -1);
    BOOST_TEST_EQ(a.index(-2.0), 0);
    BOOST_TEST_EQ(a.index(-1.1), 0);
    BOOST_TEST_EQ(a.index(0.0), 2);
    BOOST_TEST_EQ(a.index(0.9), 2);
    BOOST_TEST_EQ(a.index(1.0), 3);
    BOOST_TEST_EQ(a.index(10.), 4);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 4);
    BOOST_TEST_EQ(a.index(-std::numeric_limits<double>::infinity()), -1);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::quiet_NaN()), -1);
  }

  // regular_axis with transform
  {
    regular_axis<double, transform::log> b{2, 1e0, 1e2};
    BOOST_TEST_EQ(b[-1], 0.0);
    BOOST_TEST_IS_CLOSE(b[0], 1.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[1], 10.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[2], 100.0, 1e-9);
    BOOST_TEST_EQ(b[3], std::numeric_limits<double>::infinity());

    BOOST_TEST_EQ(b.index(-1), -1);
    BOOST_TEST_EQ(b.index(0), -1);
    BOOST_TEST_EQ(b.index(1), 0);
    BOOST_TEST_EQ(b.index(9), 0);
    BOOST_TEST_EQ(b.index(10), 1);
    BOOST_TEST_EQ(b.index(90), 1);
    BOOST_TEST_EQ(b.index(100), 2);
    BOOST_TEST_EQ(b.index(std::numeric_limits<double>::infinity()), 2);
  }

  // circular_axis
  {
    circular_axis<> a{4};
    BOOST_TEST_EQ(a[-1], a[a.bins() - 1] - a.perimeter());
    circular_axis<> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    circular_axis<> c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    circular_axis<> d;
    BOOST_TEST_NOT(c == d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a.index(-1.0 * a.perimeter()), 0);
    BOOST_TEST_EQ(a.index(0.0), 0);
    BOOST_TEST_EQ(a.index(0.25 * a.perimeter()), 1);
    BOOST_TEST_EQ(a.index(0.5 * a.perimeter()), 2);
    BOOST_TEST_EQ(a.index(0.75 * a.perimeter()), 3);
    BOOST_TEST_EQ(a.index(1.00 * a.perimeter()), 0);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 0);
    BOOST_TEST_EQ(a.index(-std::numeric_limits<double>::infinity()), 0);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::quiet_NaN()), 0);
  }

  // variable_axis
  {
    variable_axis<> a{-1, 0, 1};
    BOOST_TEST_EQ(a[-1], -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.bins() + 1], std::numeric_limits<double>::infinity());
    variable_axis<> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    variable_axis<> c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    variable_axis<> d;
    BOOST_TEST_NOT(c == d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    variable_axis<> e{-2, 0, 2};
    BOOST_TEST_NOT(a == e);
    BOOST_TEST_EQ(a.index(-10.), -1);
    BOOST_TEST_EQ(a.index(-1.), 0);
    BOOST_TEST_EQ(a.index(0.), 1);
    BOOST_TEST_EQ(a.index(1.), 2);
    BOOST_TEST_EQ(a.index(10.), 2);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 2);
    BOOST_TEST_EQ(a.index(-std::numeric_limits<double>::infinity()), -1);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::quiet_NaN()), 2);
  }

  // integer_axis
  {
    integer_axis a{-1, 1};
    integer_axis b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    integer_axis c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    integer_axis d;
    BOOST_TEST_NOT(c == d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a.index(-10), -1);
    BOOST_TEST_EQ(a.index(-2), -1);
    BOOST_TEST_EQ(a.index(-1), 0);
    BOOST_TEST_EQ(a.index(0), 1);
    BOOST_TEST_EQ(a.index(1), 2);
    BOOST_TEST_EQ(a.index(2), 3);
    BOOST_TEST_EQ(a.index(10), 3);
  }

  // category_axis
  {
    category_axis a{{"A", "B", "C"}};
    category_axis b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    category_axis c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    category_axis d;
    BOOST_TEST_NOT(c == d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a.index(0), 0);
    BOOST_TEST_EQ(a.index(1), 1);
    BOOST_TEST_EQ(a.index(2), 2);
  }

  // iterators
  {
    test_real_axis_iterator(regular_axis<>(5, 0, 1, "", false), 0, 5);
    test_real_axis_iterator(regular_axis<>(5, 0, 1, "", true), -1, 6);
    test_real_axis_iterator(circular_axis<>(5, 0, 1, ""), 0, 5);
    test_real_axis_iterator(variable_axis<>({1, 2, 3}, "", false), 0, 2);
    test_real_axis_iterator(variable_axis<>({1, 2, 3}, "", true), -1, 3);
    test_axis_iterator(integer_axis(0, 4, "", false), 0, 5);
    test_axis_iterator(integer_axis(0, 4, "", true), -1, 6);
    test_axis_iterator(category_axis({"A", "B", "C"}), 0, 3);
  }

  // axis_t_copyable
  {
    axis_t a(regular_axis<>(2, -1, 1));
    axis_t b(a);
    BOOST_TEST(a == b);
    axis_t c;
    BOOST_TEST_NOT(a == c);
    c = a;
    BOOST_TEST(a == c);
  }

  // axis_t_movable
  {
    axis_t a(regular_axis<>(2, -1, 1));
    axis_t r(a);
    axis_t b(std::move(a));
    BOOST_TEST(b == r);
    axis_t c;
    BOOST_TEST_NOT(a == c);
    c = std::move(b);
    BOOST_TEST(c == r);
  }

  // axis_t_streamable
  {
    std::vector<axis_t> axes;
    axes.push_back(regular_axis<>{2, -1, 1, "regular", false});
    axes.push_back(circular_axis<>{4, 0.1, 1.0, "polar"});
    axes.push_back(variable_axis<>{{-1, 0, 1}, "variable", false});
    axes.push_back(category_axis{{"A", "B", "C"}, "category"});
    axes.push_back(integer_axis{-1, 1, "integer", false});
    std::ostringstream os;
    for (const auto &a : axes) {
      os << a;
    }
    const std::string ref =
        "regular_axis(2, -1, 1, label='regular', uoflow=False)"
        "circular_axis(4, phase=0.1, perimeter=1, label='polar')"
        "variable_axis(-1, 0, 1, label='variable', uoflow=False)"
        "category_axis('A', 'B', 'C', label='category')"
        "integer_axis(-1, 1, label='integer', uoflow=False)";
    BOOST_TEST_EQ(os.str(), ref);
  }

  // axis_t_equal_comparable
  {
    std::vector<axis_t> axes;
    axes.push_back(regular_axis<>{2, -1, 1});
    axes.push_back(circular_axis<>{4});
    axes.push_back(variable_axis<>{-1, 0, 1});
    axes.push_back(category_axis{"A", "B", "C"});
    axes.push_back(integer_axis{-1, 1});
    for (const auto &a : axes) {
      BOOST_TEST(!(a == axis_t()));
      BOOST_TEST_EQ(a, a);
    }
    BOOST_TEST_NOT(axes == std::vector<axis_t>());
    BOOST_TEST(axes == std::vector<axis_t>(axes));
  }

  // sequence equality
  {
    std::vector<boost::variant<regular_axis<>, variable_axis<>, category_axis,
                               integer_axis>>
        std_vector1 = {regular_axis<>{2, -1, 1}, variable_axis<>{-1, 0, 1},
                       category_axis{"A", "B", "C"}};

    std::vector<boost::variant<regular_axis<>, variable_axis<>, category_axis>>
        std_vector2 = {regular_axis<>{2, -1, 1}, variable_axis<>{-1, 0, 1},
                       category_axis{"A", "B", "C"}};

    std::vector<boost::variant<regular_axis<>, variable_axis<>>> std_vector3 = {
        variable_axis<>{-1, 0, 1}, regular_axis<>{2, -1, 1},
        variable_axis<>{-1, 0, 1}};

    std::vector<boost::variant<regular_axis<>, variable_axis<>>> std_vector4 = {
        regular_axis<>{2, -1, 1}, variable_axis<>{-1, 0, 1},
    };

    BOOST_TEST(detail::axes_equal(std_vector1, std_vector2));
    BOOST_TEST_NOT(detail::axes_equal(std_vector2, std_vector3));
    BOOST_TEST_NOT(detail::axes_equal(std_vector3, std_vector4));

    auto fusion_vector1 = boost::fusion::make_vector(
        regular_axis<>{2, -1, 1}, variable_axis<>{-1, 0, 1},
        category_axis{"A", "B", "C"});

    auto fusion_vector2 = boost::fusion::make_vector(regular_axis<>{2, -1, 1},
                                                     variable_axis<>{-1, 0, 1},
                                                     category_axis{"A", "B"});

    auto fusion_vector3 = boost::fusion::make_vector(regular_axis<>{2, -1, 1},
                                                     variable_axis<>{-1, 0, 1});

    BOOST_TEST(detail::axes_equal(std_vector1, fusion_vector1));
    BOOST_TEST(detail::axes_equal(fusion_vector1, std_vector1));
    BOOST_TEST_NOT(detail::axes_equal(fusion_vector1, fusion_vector2));
    BOOST_TEST_NOT(detail::axes_equal(fusion_vector2, fusion_vector3));
    BOOST_TEST_NOT(detail::axes_equal(std_vector3, fusion_vector3));
  }

  // sequence assign
  {
    std::vector<boost::variant<regular_axis<>, variable_axis<>, category_axis,
                               integer_axis>>
        std_vector1 = {regular_axis<>{2, -1, 1}, variable_axis<>{-1, 0, 1},
                       category_axis{"A", "B", "C"}};

    std::vector<boost::variant<regular_axis<>, variable_axis<>, category_axis>>
        std_vector2 = {regular_axis<>{2, -2, 2}, variable_axis<>{-2, 0, 2},
                       category_axis{"A", "B"}};

    detail::axes_assign(std_vector2, std_vector1);
    BOOST_TEST(detail::axes_equal(std_vector2, std_vector1));

    auto fusion_vector1 = boost::fusion::make_vector(
        regular_axis<>{2, -3, 3}, variable_axis<>{-3, 0, 3},
        category_axis{"A", "B", "C", "D"});

    detail::axes_assign(fusion_vector1, std_vector1);
    BOOST_TEST(detail::axes_equal(fusion_vector1, std_vector1));

    auto fusion_vector2 = boost::fusion::make_vector(regular_axis<>{2, -1, 1},
                                                     variable_axis<>{-1, 0, 1},
                                                     category_axis{"A", "B"});

    detail::axes_assign(fusion_vector2, fusion_vector1);
    BOOST_TEST(detail::axes_equal(fusion_vector2, fusion_vector1));
  }

  return boost::report_errors();
}
