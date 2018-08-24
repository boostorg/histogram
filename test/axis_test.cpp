// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <limits>
#include <sstream>
#include <string>
#include "utility.hpp"

using namespace boost::histogram;

#define BOOST_TEST_IS_CLOSE(a, b, eps) BOOST_TEST(std::abs(a - b) < eps)

template <typename Axis>
void test_axis_iterator(const Axis& a, int begin, int end) {
  for (auto bin : a) {
    BOOST_TEST_EQ(bin.idx(), begin);
    BOOST_TEST_EQ(bin, a[begin]);
    ++begin;
  }
  BOOST_TEST_EQ(begin, end);
  auto rit = a.rbegin();
  for (; rit != a.rend(); ++rit) {
    BOOST_TEST_EQ(rit->idx(), --begin);
    BOOST_TEST_EQ(*rit, a[begin]);
  }
}

int main() {
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
    BOOST_TEST_EQ(a.index(-std::numeric_limits<double>::infinity()), -1);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 4);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::quiet_NaN()), 4);
  }

  // axis::regular with log transform
  {
    axis::regular<axis::transform::log> b{2, 1e0, 1e2};
    BOOST_TEST_EQ(b[-1].lower(), 0.0);
    BOOST_TEST_IS_CLOSE(b[0].lower(), 1.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[1].lower(), 10.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[2].lower(), 100.0, 1e-9);
    BOOST_TEST_EQ(b[2].upper(), std::numeric_limits<double>::infinity());

    BOOST_TEST_EQ(b.index(-1), 2); // produces NaN in conversion
    BOOST_TEST_EQ(b.index(0), -1);
    BOOST_TEST_EQ(b.index(1), 0);
    BOOST_TEST_EQ(b.index(9), 0);
    BOOST_TEST_EQ(b.index(10), 1);
    BOOST_TEST_EQ(b.index(90), 1);
    BOOST_TEST_EQ(b.index(100), 2);
    BOOST_TEST_EQ(b.index(std::numeric_limits<double>::infinity()), 2);
  }

  // axis::regular with sqrt transform
  {
    axis::regular<axis::transform::sqrt> b{2, 0, 4};
    // this is weird: -inf * -inf = inf, thus the lower bound
    BOOST_TEST_EQ(b[-1].lower(), std::numeric_limits<double>::infinity());
    BOOST_TEST_IS_CLOSE(b[0].lower(), 0.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[1].lower(), 1.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[2].lower(), 4.0, 1e-9);
    BOOST_TEST_EQ(b[2].upper(), std::numeric_limits<double>::infinity());

    BOOST_TEST_EQ(b.index(-1), 2); // produces NaN in conversion
    BOOST_TEST_EQ(b.index(0), 0);
    BOOST_TEST_EQ(b.index(0.99), 0);
    BOOST_TEST_EQ(b.index(1), 1);
    BOOST_TEST_EQ(b.index(3.99), 1);
    BOOST_TEST_EQ(b.index(4), 2);
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
    BOOST_TEST_EQ(a.index(-std::numeric_limits<double>::infinity()), -1);
    BOOST_TEST_EQ(a.index(std::numeric_limits<double>::infinity()), 2);
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
    std::string A("A"), B("B"), C("C"), other;
    axis::category<std::string> a{{A, B, C}};
    axis::category<std::string> b;
    BOOST_TEST_NOT(a == b);
    b = a;
    BOOST_TEST_EQ(a, b);
    b = axis::category<std::string>{{B, A, C}};
    BOOST_TEST_NOT(a == b);
    b = a;
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
    BOOST_TEST_EQ(a.index(other), 3);
    BOOST_TEST_EQ(a.value(0), A);
    BOOST_TEST_EQ(a.value(1), B);
    BOOST_TEST_EQ(a.value(2), C);
    BOOST_TEST_THROWS(a.value(3), std::out_of_range);
  }

  // iterators
  {
    enum { A, B, C };
    test_axis_iterator(axis::regular<>(5, 0, 1, "", axis::uoflow_type::off), 0, 5);
    test_axis_iterator(axis::regular<>(5, 0, 1, "", axis::uoflow_type::on), 0, 5);
    test_axis_iterator(axis::circular<>(5, 0, 1, ""), 0, 5);
    test_axis_iterator(axis::variable<>({1, 2, 3}, ""), 0, 2);
    test_axis_iterator(axis::integer<>(0, 4, ""), 0, 4);
    test_axis_iterator(axis::category<>({A, B, C}, ""), 0, 3);
    test_axis_iterator(axis::any_std(axis::regular<>(5, 0, 1)), 0, 5);
    BOOST_TEST_THROWS(axis::any_std(axis::category<>({A, B, C})).lower(0),
                      std::runtime_error);
  }

  // axis::any copyable
  {
    axis::any_std a1(axis::regular<>(2, -1, 1));
    axis::any_std a2(a1);
    BOOST_TEST_EQ(a1, a2);
    axis::any_std a3;
    BOOST_TEST_NE(a3, a1);
    a3 = a1;
    BOOST_TEST_EQ(a3, a1);
    axis::any<axis::regular<>> a4(axis::regular<>(3, -2, 2));
    axis::any_std a5(a4);
    BOOST_TEST_EQ(a4, a5);
    axis::any<axis::regular<>> a6;
    a6 = a1;
    BOOST_TEST_EQ(a6, a1);
    axis::any<axis::regular<>, axis::integer<>> a7(axis::integer<>(0, 2));
    BOOST_TEST_THROWS(axis::any<axis::regular<>> a8(a7), std::invalid_argument);
    BOOST_TEST_THROWS(a4 = a7, std::invalid_argument);
  }

  // axis::any movable
  {
    axis::any_std a(axis::regular<>(2, -1, 1));
    axis::any_std r(a);
    axis::any_std b(std::move(a));
    BOOST_TEST_EQ(b, r);
    axis::any_std c;
    BOOST_TEST_NOT(a == c);
    c = std::move(b);
    BOOST_TEST(c == r);
  }

  // axis::any streamable
  {
    enum { A, B, C };
    std::string a = "A";
    std::string b = "B";
    std::vector<axis::any_std> axes;
    axes.push_back(axis::regular<>{2, -1, 1, "regular1"});
    axes.push_back(axis::regular<axis::transform::log>(2, 1, 10, "regular2",
                                                       axis::uoflow_type::off));
    axes.push_back(axis::regular<axis::transform::pow>(2, 1, 10, "regular3",
                                                       axis::uoflow_type::on, 0.5));
    axes.push_back(axis::regular<axis::transform::pow>(2, 1, 10, "regular4",
                                                       axis::uoflow_type::off, -0.5));
    axes.push_back(axis::circular<>(4, 0.1, 1.0, "polar"));
    axes.push_back(axis::variable<>({-1, 0, 1}, "variable", axis::uoflow_type::off));
    axes.push_back(axis::category<>({A, B, C}, "category"));
    axes.push_back(axis::category<std::string>({a, b}, "category2"));
    axes.push_back(axis::integer<>(-1, 1, "integer", axis::uoflow_type::off));
    std::ostringstream os;
    for (const auto& a : axes) { os << a << "\n"; }
    os << axes.back()[0];
    const std::string ref =
        "regular(2, -1, 1, label='regular1')\n"
        "regular_log(2, 1, 10, label='regular2', uoflow=False)\n"
        "regular_pow(2, 1, 10, 0.5, label='regular3')\n"
        "regular_pow(2, 1, 10, -0.5, label='regular4', uoflow=False)\n"
        "circular(4, phase=0.1, perimeter=1, label='polar')\n"
        "variable(-1, 0, 1, label='variable', uoflow=False)\n"
        "category(0, 1, 2, label='category')\n"
        "category('A', 'B', label='category2')\n"
        "integer(-1, 1, label='integer', uoflow=False)\n"
        "[-1, 0)";
    BOOST_TEST_EQ(os.str(), ref);
  }

  // axis::any equal_comparable
  {
    enum { A, B, C };
    std::vector<axis::any_std> axes;
    axes.push_back(axis::regular<>{2, -1, 1});
    axes.push_back(
        axis::regular<axis::transform::pow>(2, 1, 4, "", axis::uoflow_type::on, 0.5));
    axes.push_back(axis::circular<>{4});
    axes.push_back(axis::variable<>{-1, 0, 1});
    axes.push_back(axis::category<>{A, B, C});
    axes.push_back(axis::integer<>{-1, 1});
    for (const auto& a : axes) {
      BOOST_TEST(!(a == axis::any_std()));
      BOOST_TEST_EQ(a, axis::any_std(a));
    }
    BOOST_TEST_NOT(axes == std::vector<axis::any_std>());
    BOOST_TEST(axes == std::vector<axis::any_std>(axes));
  }

  // axis::any value_to_index_failure
  {
    std::string a = "A", b = "B";
    axis::any_std x = axis::category<std::string>({a, b}, "category");
    BOOST_TEST_THROWS(x.index(1.5), std::runtime_error);
    auto cx = static_cast<const axis::category<std::string>&>(x);
    BOOST_TEST_EQ(cx.index(b), 1);
  }

  // sequence equality
  {
    enum { A, B, C };
    std::vector<
        axis::any<axis::regular<>, axis::variable<>, axis::category<>, axis::integer<>>>
        std_vector1 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{A, B, C}};

    std::vector<axis::any<axis::regular<>, axis::variable<>, axis::category<>>>
        std_vector2 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{{A, B, C}}};

    std::vector<axis::any<axis::regular<>, axis::variable<>>> std_vector3 = {
        axis::variable<>{-1, 0, 1}, axis::regular<>{2, -1, 1}};

    std::vector<axis::any<axis::variable<>, axis::regular<>>> std_vector4 = {
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
    std::vector<
        axis::any<axis::regular<>, axis::variable<>, axis::category<>, axis::integer<>>>
        std_vector1 = {axis::regular<>{2, -1, 1}, axis::variable<>{-1, 0, 1},
                       axis::category<>{A, B, C}};

    std::vector<axis::any<axis::regular<>, axis::variable<>, axis::category<>>>
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

  // vector of axes with custom allocators
  {
    using T1 = axis::regular<axis::transform::identity, double, tracing_allocator<char>>;
    using T2 = axis::circular<double, tracing_allocator<char>>;
    using T3 = axis::variable<double, tracing_allocator<char>>;
    using T4 = axis::integer<int, tracing_allocator<char>>;
    using T5 = axis::category<long, tracing_allocator<char>>;
    using axis_type = axis::any<T1, T2, T3, T4, T5>; // no heap allocation
    using axes_type = std::vector<axis_type, tracing_allocator<axis_type>>;

    tracing_allocator_db db;
    {
      auto a = tracing_allocator<char>(db);
      const auto label = std::string(512, 'c');
      axes_type axes(a);
      axes.reserve(5);
      axes.emplace_back(T1(1, 0, 1, label, axis::uoflow_type::on, {}, a));
      axes.emplace_back(T2(2, 0, T2::two_pi(), label, a));
      axes.emplace_back(T3({0., 1., 2.}, label, axis::uoflow_type::on, a));
      axes.emplace_back(T4(0, 4, label, axis::uoflow_type::on, a));
      axes.emplace_back(T5({1, 2, 3, 4, 5}, label, axis::uoflow_type::on, a));
    }
    // 5 axis::any objects
    BOOST_TEST_EQ(db[typeid(axis_type)].first, db[typeid(axis_type)].second);
    BOOST_TEST_EQ(db[typeid(axis_type)].first, 5);

    // 5 labels
    BOOST_TEST_EQ(db[typeid(char)].first, db[typeid(char)].second);
    BOOST_TEST_GE(db[typeid(char)].first, 5 * 512u);

    // nothing to allocate for T1
    // nothing to allocate for T2
    // T3 allocates storage for bin edges
    BOOST_TEST_EQ(db[typeid(double)].first, db[typeid(double)].second);
    BOOST_TEST_EQ(db[typeid(double)].first, 3u);
    // nothing to allocate for T4
    // T5 allocates storage for long array
    BOOST_TEST_EQ(db[typeid(long)].first, db[typeid(long)].second);
    BOOST_TEST_EQ(db[typeid(long)].first, 5u);

#if (BOOST_MSVC)
    BOOST_TEST_EQ(db.size(), 5); // axis_type, char, double, long + ???
#else
    BOOST_TEST_EQ(db.size(), 4); // axis_type, char, double, long
#endif
  }

  return boost::report_errors();
}
