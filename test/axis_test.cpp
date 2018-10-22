// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
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

// quantity with unit for testing
template <typename Unit>
struct quantity {
  double value;
};

struct length {
  double value;
};

auto meter = length{1.0};

template <typename Unit>
double operator/(const quantity<Unit>& a, const Unit& u) {
  return a.value / u.value;
}

template <typename Unit>
quantity<Unit> operator*(double x, const Unit& u) {
  return quantity<Unit>{x * u.value};
}

template <typename Unit>
quantity<Unit> operator-(const quantity<Unit>& a, const quantity<Unit>& b) {
  return quantity<Unit>{a.value - b.value};
}

int main() {
  // bad_ctors
  {
    BOOST_TEST_THROWS(axis::regular<>(0, 0, 1), std::invalid_argument);
    // BOOST_TEST_THROWS(axis::regular<>(1, 1, -1), std::invalid_argument);
    BOOST_TEST_THROWS(axis::circular<>(0), std::invalid_argument);
    BOOST_TEST_THROWS(axis::variable<>(std::vector<double>()), std::invalid_argument);
    BOOST_TEST_THROWS(axis::variable<>({1.0}), std::invalid_argument);
    BOOST_TEST_THROWS(axis::integer<>(1, -1), std::invalid_argument);
    BOOST_TEST_THROWS(axis::category<>(std::vector<int>()), std::invalid_argument);
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
    axis::regular<> a(2, 1, -1);
    BOOST_TEST_EQ(a[0].lower(), 1);
    BOOST_TEST_EQ(a[1].lower(), 0);
    BOOST_TEST_EQ(a[2].lower(), -1);
    BOOST_TEST_EQ(a(2), -1);
    BOOST_TEST_EQ(a(1), 0);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(-1), 2);
    BOOST_TEST_EQ(a(-2), 2);
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

  // axis::regular with quantity transform
  {
    axis::regular<axis::transform::quantity<quantity<length>, length>> b{
      2, 0*meter, 2*meter, {}, axis::option_type::underflow_and_overflow, meter
    };
    BOOST_TEST_EQ(b[-1].lower()/meter, -std::numeric_limits<double>::infinity());
    BOOST_TEST_IS_CLOSE(b[0].lower()/meter, 0.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[1].lower()/meter, 1.0, 1e-9);
    BOOST_TEST_IS_CLOSE(b[2].lower()/meter, 2.0, 1e-9);
    BOOST_TEST_EQ(b[2].upper()/meter, std::numeric_limits<double>::infinity());

    BOOST_TEST_EQ(b(-1*meter), -1); // produces NaN in conversion
    BOOST_TEST_EQ(b(0*meter), 0);
    BOOST_TEST_EQ(b(0.99*meter), 0);
    BOOST_TEST_EQ(b(1*meter), 1);
    BOOST_TEST_EQ(b(1.99*meter), 1);
    BOOST_TEST_EQ(b(2*meter), 2);
    BOOST_TEST_EQ(b(100*meter), 2);
    BOOST_TEST_EQ(b(std::numeric_limits<double>::infinity()*meter), 2);
  }

  // axis::circular
  {
    axis::circular<> a{4, 0, 1};
    BOOST_TEST_EQ(a[-1].lower(), a[a.size() - 1].lower() - 1);
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
    BOOST_TEST_EQ(a(-1.0 * 3), 0);
    BOOST_TEST_EQ(a(0.0), 0);
    BOOST_TEST_EQ(a(0.25), 1);
    BOOST_TEST_EQ(a(0.5), 2);
    BOOST_TEST_EQ(a(0.75), 3);
    BOOST_TEST_EQ(a(1.0), 0);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::infinity()), 4);
    BOOST_TEST_EQ(a(-std::numeric_limits<double>::infinity()), 4);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::quiet_NaN()), 4);
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
    BOOST_TEST_EQ(a(-10.), -1);
    BOOST_TEST_EQ(a(-1.), 0);
    BOOST_TEST_EQ(a(0.), 1);
    BOOST_TEST_EQ(a(1.), 2);
    BOOST_TEST_EQ(a(10.), 2);
    BOOST_TEST_EQ(a(-std::numeric_limits<double>::infinity()), -1);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::infinity()), 2);
    BOOST_TEST_EQ(a(std::numeric_limits<double>::quiet_NaN()), 2);
  }

  // axis::integer
  {
    axis::integer<> a{-1, 2};
    BOOST_TEST_EQ(a[-1].lower(), std::numeric_limits<int>::min());
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
    BOOST_TEST_EQ(a(-10), -1);
    BOOST_TEST_EQ(a(-2), -1);
    BOOST_TEST_EQ(a(-1), 0);
    BOOST_TEST_EQ(a(0), 1);
    BOOST_TEST_EQ(a(1), 2);
    BOOST_TEST_EQ(a(2), 3);
    BOOST_TEST_EQ(a(10), 3);
  }

  // axis::category
  {
    std::string A("A"), B("B"), C("C"), other;
    axis::category<std::string> a({A, B, C});
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
    BOOST_TEST_EQ(a(A), 0);
    BOOST_TEST_EQ(a(B), 1);
    BOOST_TEST_EQ(a(C), 2);
    BOOST_TEST_EQ(a(other), 3);
    BOOST_TEST_EQ(a.value(0), A);
    BOOST_TEST_EQ(a.value(1), B);
    BOOST_TEST_EQ(a.value(2), C);
    BOOST_TEST_THROWS(a.value(3), std::out_of_range);
  }

  // iterators
  {
    enum { A, B, C };
    test_axis_iterator(axis::regular<>(5, 0, 1, "", axis::option_type::none), 0, 5);
    test_axis_iterator(axis::regular<>(5, 0, 1, "", axis::option_type::underflow_and_overflow), 0, 5);
    test_axis_iterator(axis::circular<>(5, 0, 1, ""), 0, 5);
    test_axis_iterator(axis::variable<>({1, 2, 3}, ""), 0, 2);
    test_axis_iterator(axis::integer<>(0, 4, ""), 0, 4);
    test_axis_iterator(axis::category<>({A, B, C}, ""), 0, 3);
    test_axis_iterator(axis::variant<axis::regular<>>(axis::regular<>(5, 0, 1)), 0, 5);
    // BOOST_TEST_THROWS(axis::variant<axis::category<>>(axis::category<>({A, B, C}))[0].lower(),
    //                   std::runtime_error);
  }

  // axis::variant copyable
  {
    axis::variant<axis::regular<>> a1(axis::regular<>(2, -1, 1));
    axis::variant<axis::regular<>> a2(a1);
    BOOST_TEST_EQ(a1, a2);
    axis::variant<axis::regular<>> a3;
    BOOST_TEST_NE(a3, a1);
    a3 = a1;
    BOOST_TEST_EQ(a3, a1);
    axis::variant<axis::regular<>> a4(axis::regular<>(3, -2, 2));
    axis::variant<axis::regular<>, axis::integer<>> a5(a4);
    BOOST_TEST_EQ(a4, a5);
    axis::variant<axis::regular<>> a6;
    a6 = a1;
    BOOST_TEST_EQ(a6, a1);
    axis::variant<axis::regular<>, axis::integer<>> a7(axis::integer<>(0, 2));
    BOOST_TEST_THROWS(axis::variant<axis::regular<>> a8(a7), std::runtime_error);
    BOOST_TEST_THROWS(a4 = a7, std::runtime_error);
  }

  // axis::variant movable
  {
    axis::variant<axis::regular<>> a(axis::regular<>(2, -1, 1));
    axis::variant<axis::regular<>> r(a);
    axis::variant<axis::regular<>> b(std::move(a));
    BOOST_TEST_EQ(b, r);
    axis::variant<axis::regular<>> c;
    BOOST_TEST_NOT(a == c);
    c = std::move(b);
    BOOST_TEST(c == r);
  }

  // axis::variant streamable
  {
    auto test = [](auto&& a, const char* ref) {
      using T = detail::rm_cvref<decltype(a)>;
      axis::variant<T> axis(std::move(a));
      std::ostringstream os;
      os << axis;
      BOOST_TEST_EQ(os.str(), std::string(ref));
    };

    struct user_defined {};

    test(axis::regular<>{2, -1, 1, "regular1"},
         "regular(2, -1, 1, metadata=\"regular1\", options=underflow_and_overflow)");
    test(axis::regular<axis::transform::log<>>(2, 1, 10, "regular2", axis::option_type::none),
         "regular_log(2, 1, 10, metadata=\"regular2\", options=none)");
    test(axis::regular<axis::transform::pow<>>(2, 1, 10, "regular3", axis::option_type::overflow, 0.5),
         "regular_pow(2, 1, 10, metadata=\"regular3\", options=overflow, power=0.5)");
    test(axis::regular<axis::transform::pow<>>(2, 1, 10, "regular4", axis::option_type::none, -0.5),
         "regular_pow(2, 1, 10, metadata=\"regular4\", options=none, power=-0.5)");
    test(axis::circular<double, axis::empty_metadata_type>(4, 0.1, 1.0),
         "circular(4, 0.1, 1.1, options=overflow)");
    test(axis::variable<>({-1, 0, 1}, "variable", axis::option_type::none),
         "variable(-1, 0, 1, metadata=\"variable\", options=none)");
    test(axis::category<>({0, 1, 2}, "category"),
         "category(0, 1, 2, metadata=\"category\", options=overflow)");
    test(axis::category<std::string>({"A", "B"}, "category2"),
         "category(\"A\", \"B\", metadata=\"category2\", options=overflow)");
    test(axis::integer<int, user_defined>(-1, 1, {}, axis::option_type::none),
         "integer(-1, 1, metadata=main::user_defined, options=none)");
  }

  // axis::variant support for minimal_axis
  {
    struct minimal_axis {
      int operator()(double) const { return 0; }
      unsigned size() const { return 1; }
    };

    axis::variant<minimal_axis> axis;
    BOOST_TEST_EQ(axis(0), 0);
    BOOST_TEST_EQ(axis(10), 0);
    BOOST_TEST_EQ(axis.size(), 1);
    BOOST_TEST_THROWS(std::ostringstream() << axis, std::runtime_error);
    BOOST_TEST_THROWS(axis.lower(0), std::runtime_error);
    BOOST_TEST_TRAIT_TRUE((std::is_same<decltype(axis.metadata()), axis::empty_metadata_type&>));
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

    auto b = axis::category<>({1, 2});
    test(b[0], "1");
  }

  // axis::variant equal_comparable
  {
    enum { A, B, C };
    using variant = axis::variant<
      axis::regular<>,
      axis::regular<axis::transform::pow<>>,
      axis::circular<>,
      axis::variable<>,
      axis::category<>,
      axis::integer<>
    >;
    std::vector<variant> axes;
    axes.push_back(axis::regular<>{2, -1, 1});
    axes.push_back(
        axis::regular<axis::transform::pow<>>(2, 1, 4, "", axis::option_type::underflow_and_overflow, 0.5));
    axes.push_back(axis::circular<>{4});
    axes.push_back(axis::variable<>{-1, 0, 1});
    axes.push_back(axis::category<>({A, B, C}));
    axes.push_back(axis::integer<>{-1, 1});
    for (const auto& a : axes) {
      BOOST_TEST(!(a == variant()));
      BOOST_TEST_EQ(a, variant(a));
    }
    BOOST_TEST_NOT(axes == std::vector<variant>());
    BOOST_TEST(axes == std::vector<variant>(axes));
  }

  // axis::variant value_to_index_failure
  {
    axis::variant<axis::category<std::string>> x = axis::category<std::string>({"A", "B"}, "category");
    auto cx = axis::get<axis::category<std::string>>(x);
    // BOOST_TEST_EQ(cx(b), 1);
  }

  // sequence equality
  {
    enum { A, B, C };
    std::vector<
        axis::variant<axis::regular<>, axis::variable<>, axis::category<>, axis::integer<>>>
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
    std::vector<
        axis::variant<axis::regular<>, axis::variable<>, axis::category<>, axis::integer<>>>
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

  // vector of axes with custom allocators
  {
    struct null {};
    using M = std::vector<char, tracing_allocator<char>>;
    using T1 = axis::regular<axis::transform::identity<>, M>;
    using T2 = axis::circular<double, null>;
    using T3 = axis::variable<double, tracing_allocator<double>, null>;
    using T4 = axis::integer<int, null>;
    using T5 = axis::category<long, tracing_allocator<long>, null>;
    using axis_type = axis::variant<T1, T2, T3, T4, T5>; // no heap allocation
    using axes_type = std::vector<axis_type, tracing_allocator<axis_type>>;

    tracing_allocator_db db;
    {
      auto a = tracing_allocator<char>(db);
      axes_type axes(a);
      axes.reserve(5);
      axes.emplace_back(T1(1, 0, 1, M(3, 'c', a)));
      axes.emplace_back(T2(2));
      axes.emplace_back(T3({0., 1., 2.}, {}, axis::option_type::underflow_and_overflow, a));
      axes.emplace_back(T4(0, 4));
      axes.emplace_back(T5({1, 2, 3, 4, 5}, {}, axis::option_type::overflow, a));
    }
    // 5 axis::variant objects
    BOOST_TEST_EQ(db[typeid(axis_type)].first, db[typeid(axis_type)].second);
    BOOST_TEST_EQ(db[typeid(axis_type)].first, 5);

    // label
    BOOST_TEST_EQ(db[typeid(char)].first, db[typeid(char)].second);
    BOOST_TEST_EQ(db[typeid(char)].first, 3u);

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
