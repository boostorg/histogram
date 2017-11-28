// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/fusion/container/generation/make_vector.hpp>
#include <boost/fusion/include/make_vector.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/axis/axis.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/variant.hpp>
#include <limits>

#define BOOST_TEST_NOT(expr) BOOST_TEST(!(expr))
#define BOOST_TEST_IS_CLOSE(a, b, eps) BOOST_TEST(std::abs(a - b) < eps)

namespace boost {
namespace histogram {
namespace axis {
template <typename T>
std::ostream &operator<<(std::ostream &os, const interval<T> &i) {
  os << "[" << i.lower() << ", " << i.upper() << ")";
  return os;
}
} // namespace axis
} // namespace histogram
} // namespace boost

template <typename Axis>
void test_axis_iterator(const Axis &a, int begin, int end) {
  for (const auto &bin : a) {
    BOOST_TEST_EQ(bin.first, begin);
    BOOST_TEST_EQ(bin.second, a[begin]);
    ++begin;
  }
  BOOST_TEST_EQ(begin, end);
}

int main() {
  using namespace boost::histogram;
  using any_axis_type = axis::any<>;

  // bad_ctors
  {
    BOOST_TEST_THROWS(axis::regular<>(0, 0, 1), std::logic_error);
    BOOST_TEST_THROWS(axis::regular<>(1, 1, -1), std::logic_error);
    BOOST_TEST_THROWS(axis::circular<>(0), std::logic_error);
    BOOST_TEST_THROWS(axis::variable<>({}), std::logic_error);
    BOOST_TEST_THROWS(axis::variable<>({1.0}), std::logic_error);
    BOOST_TEST_THROWS(axis::integer<>(1, -1), std::logic_error);
    BOOST_TEST_THROWS(axis::category<>({}), std::logic_error);
  }

  // axis::regular
  {
    axis::regular<> a{4, -2, 2};
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<double>::infinity());
    axis::regular<> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::regular<> c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    axis::regular<> d;
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

  // axis::regular with transform
  {
    axis::regular<double, axis::transform::log> b{2, 1e0, 1e2};
    BOOST_TEST_EQ(b[-1].lower(), 0.0);
    BOOST_TEST_IS_CLOSE(b[0].lower(), 1.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[1].lower(), 10.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[2].lower(), 100.0, 1e-9);
    BOOST_TEST_EQ(b[2].upper(), std::numeric_limits<double>::infinity());

    BOOST_TEST_EQ(b.index(-1), -1);
    BOOST_TEST_EQ(b.index(0), -1);
    BOOST_TEST_EQ(b.index(1), 0);
    BOOST_TEST_EQ(b.index(9), 0);
    BOOST_TEST_EQ(b.index(10), 1);
    BOOST_TEST_EQ(b.index(90), 1);
    BOOST_TEST_EQ(b.index(100), 2);
    BOOST_TEST_EQ(b.index(std::numeric_limits<double>::infinity()), 2);
  }

  // axis::circular
  {
    axis::circular<> a{4};
    BOOST_TEST_EQ(a[-1].lower(), a[a.size() - 1].lower() - a.perimeter());
    axis::circular<> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::circular<> c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    axis::circular<> d;
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

  // axis::variable
  {
    axis::variable<> a{-1, 0, 1};
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<double>::infinity());
    axis::variable<> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::variable<> c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    axis::variable<> d;
    BOOST_TEST_NOT(c == d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    axis::variable<> e{-2, 0, 2};
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

  // axis::integer
  {
    axis::integer<> a{-1, 2};
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<int>::max());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<int>::max());
    axis::integer<> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::integer<> c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    axis::integer<> d;
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

  // axis::category
  {
    std::string A("A"), B("B"), C("C");
    axis::category<std::string> a{{A, B, C}};
    axis::category<std::string> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = b;
    BOOST_TEST_EQ(a, b);
    axis::category<std::string> c = std::move(b);
    BOOST_TEST(c == a);
    BOOST_TEST_NOT(b == a);
    axis::category<std::string> d;
    BOOST_TEST_NOT(c == d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(a.index(A), 0);
    BOOST_TEST_EQ(a.index(B), 1);
    BOOST_TEST_EQ(a.index(C), 2);
    BOOST_TEST_EQ(a[0], A);
    BOOST_TEST_EQ(a[1], B);
    BOOST_TEST_EQ(a[2], C);
  }

  // iterators
  {
    enum { A, B, C };
    test_axis_iterator(axis::regular<>(5, 0, 1, "", axis::uoflow::off), 0, 5);
    test_axis_iterator(axis::regular<>(5, 0, 1, "", axis::uoflow::on), 0, 5);
    test_axis_iterator(axis::circular<>(5, 0, 1, ""), 0, 5);
    test_axis_iterator(axis::variable<>({1, 2, 3}, ""), 0, 2);
    test_axis_iterator(axis::integer<>(0, 4, ""), 0, 4);
    test_axis_iterator(axis::category<>({A, B, C}, ""), 0, 3);
    test_axis_iterator(any_axis_type(axis::regular<>(5, 0, 1)), 0, 5);
  }

  // any_axis_type_copyable
  {
    any_axis_type a(axis::regular<>(2, -1, 1));
    any_axis_type b(a);
    BOOST_TEST(a == b);
    any_axis_type c;
    BOOST_TEST_NOT(a == c);
    c = a;
    BOOST_TEST(a == c);
  }

  // any_axis_type_movable
  {
    any_axis_type a(axis::regular<>(2, -1, 1));
    any_axis_type r(a);
    any_axis_type b(std::move(a));
    BOOST_TEST_EQ(b, r);
    any_axis_type c;
    BOOST_TEST_NOT(a == c);
    c = std::move(b);
    BOOST_TEST(c == r);
  }

  // any_axis_type_streamable
  {
    enum { A, B, C };
    std::vector<any_axis_type> axes;
    axes.push_back(axis::regular<>{2, -1, 1, "regular1"});
    axes.push_back(axis::regular<double, axis::transform::log>{
        2, 1, 10, "regular2", axis::uoflow::off});
    axes.push_back(axis::regular<double, axis::transform::pow>{
        2, 1, 10, "regular3", axis::uoflow::on, 0.5});
    axes.push_back(axis::circular<>{4, 0.1, 1.0, "polar"});
    axes.push_back(axis::variable<>{{-1, 0, 1}, "variable", axis::uoflow::off});
    axes.push_back(axis::category<>{{A, B, C}, "category"});
    axes.push_back(axis::integer<>{-1, 1, "integer", axis::uoflow::off});
    std::ostringstream os;
    for (const auto &a : axes) {
      os << a;
    }
    const std::string ref =
        "regular(2, -1, 1, label='regular1')"
        "regular_log(2, 1, 10, label='regular2', uoflow=False)"
        "regular_pow(2, 1, 10, 0.5, label='regular3')"
        "circular(4, phase=0.1, perimeter=1, label='polar')"
        "variable(-1, 0, 1, label='variable', uoflow=False)"
        "category(0, 1, 2, label='category')"
        "integer(-1, 1, label='integer', uoflow=False)";
    BOOST_TEST_EQ(os.str(), ref);
  }

  // any_axis_type_equal_comparable
  {
    enum { A, B, C };
    std::vector<any_axis_type> axes;
    axes.push_back(axis::regular<>{2, -1, 1});
    axes.push_back(axis::circular<>{4});
    axes.push_back(axis::variable<>{-1, 0, 1});
    axes.push_back(axis::category<>{A, B, C});
    axes.push_back(axis::integer<>{-1, 1});
    for (const auto &a : axes) {
      BOOST_TEST(!(a == any_axis_type()));
      BOOST_TEST_EQ(a, a);
    }
    BOOST_TEST_NOT(axes == std::vector<any_axis_type>());
    BOOST_TEST(axes == std::vector<any_axis_type>(axes));
  }

  // sequence equality
  {
    enum { A, B, C };
    std::vector<boost::variant<axis::regular<>, axis::variable<>,
                               axis::category<>, axis::integer<>>>
        std_vector1 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{A, B, C}};

    std::vector<
        boost::variant<axis::regular<>, axis::variable<>, axis::category<>>>
        std_vector2 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{{A, B, C}}};

    std::vector<boost::variant<axis::regular<>, axis::variable<>>> std_vector3 =
        {axis::variable<>{-1, 0, 1}, axis::regular<>{2, -1, 1}};

    std::vector<boost::variant<axis::variable<>, axis::regular<>>> std_vector4 =
        {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1}};

    BOOST_TEST(detail::axes_equal(std_vector1, std_vector2));
    BOOST_TEST_NOT(detail::axes_equal(std_vector2, std_vector3));
    BOOST_TEST_NOT(detail::axes_equal(std_vector3, std_vector4));

    auto fusion_vector1 = boost::fusion::make_vector(
        axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
        axis::category<>{{A, B, C}});

    auto fusion_vector2 = boost::fusion::make_vector(axis::regular<>{2, -1, 1},
                                                     axis::variable<>{-1, 0, 1},
                                                     axis::category<>{{A, B}});

    auto fusion_vector3 = boost::fusion::make_vector(
        axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1});

    BOOST_TEST(detail::axes_equal(std_vector1, fusion_vector1));
    BOOST_TEST(detail::axes_equal(fusion_vector1, std_vector1));
    BOOST_TEST_NOT(detail::axes_equal(fusion_vector1, fusion_vector2));
    BOOST_TEST_NOT(detail::axes_equal(fusion_vector2, fusion_vector3));
    BOOST_TEST_NOT(detail::axes_equal(std_vector3, fusion_vector3));
  }

  // sequence assign
  {
    enum { A, B, C, D };
    std::vector<boost::variant<axis::regular<>, axis::variable<>,
                               axis::category<>, axis::integer<>>>
        std_vector1 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{A, B, C}};

    std::vector<
        boost::variant<axis::regular<>, axis::variable<>, axis::category<>>>
        std_vector2 = {axis::regular<>{2, -2, 2}, axis::variable<>{-2, 0, 2},
                       axis::category<>{A, B}};

    detail::axes_assign(std_vector2, std_vector1);
    BOOST_TEST(detail::axes_equal(std_vector2, std_vector1));

    auto fusion_vector1 = boost::fusion::make_vector(
        axis::regular<>{2, -3, 3}, axis::variable<>{-3, 0, 3},
        axis::category<>{A, B, C, D});

    detail::axes_assign(fusion_vector1, std_vector1);
    BOOST_TEST(detail::axes_equal(fusion_vector1, std_vector1));

    decltype(std_vector1) std_vector3;
    BOOST_TEST_NOT(detail::axes_equal(std_vector3, fusion_vector1));
    detail::axes_assign(std_vector3, fusion_vector1);
    BOOST_TEST(detail::axes_equal(std_vector3, fusion_vector1));

    auto fusion_vector2 = boost::fusion::make_vector(axis::regular<>{2, -1, 1},
                                                     axis::variable<>{-1, 0, 1},
                                                     axis::category<>{A, B});

    detail::axes_assign(fusion_vector2, fusion_vector1);
    BOOST_TEST(detail::axes_equal(fusion_vector2, fusion_vector1));
  }

  return boost::report_errors();
}
