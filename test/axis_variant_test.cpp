// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/axis/category.hpp>
#include <boost/histogram/axis/circular.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <functional> // for std::ref
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "utility_allocator.hpp"
#include "utility_axis.hpp"

using namespace boost::histogram;

int main() {
  {
    axis::variant<axis::integer<>, axis::category<std::string>> a{
        axis::integer<>(0, 2, "int")};
    BOOST_TEST_EQ(a(-10), -1);
    BOOST_TEST_EQ(a(-1), -1);
    BOOST_TEST_EQ(a(0), 0);
    BOOST_TEST_EQ(a(0.5), 0);
    BOOST_TEST_EQ(a(1), 1);
    BOOST_TEST_EQ(a(2), 2);
    BOOST_TEST_EQ(a(10), 2);
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[-10].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size() + 10].upper(), std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a.metadata(), "int");
    BOOST_TEST_EQ(a.options(), axis::option_type::underflow_and_overflow);

    a = axis::category<std::string>({"A", "B"}, "cat");
    BOOST_TEST_EQ(a("A"), 0);
    BOOST_TEST_EQ(a("B"), 1);
    BOOST_TEST_EQ(a.metadata(), "cat");
    BOOST_TEST_EQ(a.options(), axis::option_type::overflow);
  }

  // axis::variant with reference
  {
    auto a = axis::integer<double, axis::null_type>(0, 3, {}, axis::option_type::none);
    using V = axis::variant<axis::integer<double, axis::null_type>&>;
    V v(a);
    BOOST_TEST_EQ(v.size(), 3);
    BOOST_TEST_EQ(v[0], a[0]);
    BOOST_TEST_EQ(v.metadata(), a.metadata());
    BOOST_TEST_EQ(v.options(), a.options());
    BOOST_TEST_EQ(v(1), a(1));
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
    BOOST_TEST_THROWS(axis.value(0), std::runtime_error);
    BOOST_TEST_TRAIT_TRUE((std::is_same<decltype(axis.metadata()), axis::null_type&>));
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
      using T = detail::unqual<decltype(a)>;
      axis::variant<T> axis(std::move(a));
      std::ostringstream os;
      os << axis;
      BOOST_TEST_EQ(os.str(), std::string(ref));
    };

    struct user_defined {};

    namespace tr = axis::transform;
    test(axis::regular<>(2, -1, 1, "regular1"),
         "regular(2, -1, 1, metadata=\"regular1\", options=underflow_and_overflow)");
    test(axis::regular<tr::log<>>(2, 1, 10, "regular2", axis::option_type::none),
         "regular_log(2, 1, 10, metadata=\"regular2\", options=none)");
    test(axis::regular<tr::pow<>>(1.5, 2, 1, 10, "regular3", axis::option_type::overflow),
         "regular_pow(2, 1, 10, metadata=\"regular3\", options=overflow, power=1.5)");
    test(axis::regular<tr::pow<>>(-1.5, 2, 1, 10, "regular4", axis::option_type::none),
         "regular_pow(2, 1, 10, metadata=\"regular4\", options=none, power=-1.5)");
    test(axis::circular<double, axis::null_type>(4, 0.1, 1.0),
         "circular(4, 0.1, 1.1, options=overflow)");
    test(axis::variable<>({-1, 0, 1}, "variable", axis::option_type::none),
         "variable(-1, 0, 1, metadata=\"variable\", options=none)");
    test(axis::category<>({0, 1, 2}, "category"),
         "category(0, 1, 2, metadata=\"category\", options=overflow)");
    test(axis::category<std::string>({"A", "B"}, "category2"),
         "category(\"A\", \"B\", metadata=\"category2\", options=overflow)");
    const auto ref = detail::cat(
        "integer(-1, 1, metadata=",
        boost::core::demangled_name(BOOST_CORE_TYPEID(user_defined)), ", options=none)");
    test(axis::integer<int, user_defined>(-1, 1, {}, axis::option_type::none),
         ref.c_str());
  }

  // bin_type operator<<
  {
    auto test = [](const auto& x, const char* ref) {
      std::ostringstream os;
      os << x;
      BOOST_TEST_EQ(os.str(), std::string(ref));
    };

    auto b = axis::category<>({1, 2});
    test(b[0], "1");
  }

  // axis::variant operator==
  {
    enum { A, B, C };
    using variant = axis::variant<axis::regular<>, axis::regular<axis::transform::pow<>>,
                                  axis::circular<>, axis::variable<>, axis::category<>,
                                  axis::integer<>>;
    std::vector<variant> axes;
    axes.push_back(axis::regular<>{2, -1, 1});
    axes.push_back(axis::regular<axis::transform::pow<>>(
        0.5, 2, 1, 4, "", axis::option_type::underflow_and_overflow));
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

  // axis::variant with unusual args
  {
    struct minimal_axis {
      int operator()(int x) const { return x % 2; }
      unsigned size() const { return 2; }
    };
    axis::variant<axis::category<std::string>, minimal_axis> x =
        axis::category<std::string>({"A", "B"}, "category");
    BOOST_TEST_EQ(x("B"), 1);
    x = minimal_axis();
    BOOST_TEST_EQ(x(4), 0);
    BOOST_TEST_EQ(x(5), 1);
  }

  // axis::variant with axis that has incompatible bin type
  {
    auto a = axis::variant<axis::category<std::string>>(
        axis::category<std::string>({"A", "B", "C"}));
    BOOST_TEST_THROWS(a[0].lower(), std::runtime_error);
    auto b = axis::variant<axis::category<int>>(axis::category<int>({2, 1, 3}));
    BOOST_TEST_THROWS(b[0].lower(), std::runtime_error);
  }

  // vector of axes with custom allocators
  {
    using M = std::vector<char, tracing_allocator<char>>;
    using T1 = axis::regular<axis::transform::identity<>, M>;
    using T2 = axis::circular<double, axis::null_type>;
    using T3 = axis::variable<double, tracing_allocator<double>, axis::null_type>;
    using T4 = axis::integer<int, axis::null_type>;
    using T5 = axis::category<long, tracing_allocator<long>, axis::null_type>;
    using axis_type = axis::variant<T1, T2, T3, T4, T5>; // no heap allocation
    using axes_type = std::vector<axis_type, tracing_allocator<axis_type>>;

    tracing_allocator_db db;
    {
      auto a = tracing_allocator<char>(db);
      axes_type axes(a);
      axes.reserve(5);
      axes.emplace_back(T1(1, 0, 1, M(3, 'c', a)));
      axes.emplace_back(T2(2));
      axes.emplace_back(
          T3({0., 1., 2.}, {}, axis::option_type::underflow_and_overflow, a));
      axes.emplace_back(T4(0, 4));
      axes.emplace_back(T5({1, 2, 3, 4, 5}, {}, axis::option_type::overflow, a));
    }
    // 5 axis::variant objects
    BOOST_TEST_EQ(db.at<axis_type>().first, db.at<axis_type>().second);
    BOOST_TEST_EQ(db.at<axis_type>().first, 5);

    // label
    BOOST_TEST_EQ(db.at<char>().first, db.at<char>().second);
    BOOST_TEST_EQ(db.at<char>().first, 3u);

    // nothing to allocate for T1
    // nothing to allocate for T2
    // T3 allocates storage for bin edges
    BOOST_TEST_EQ(db.at<double>().first, db.at<double>().second);
    BOOST_TEST_EQ(db.at<double>().first, 3u);
    // nothing to allocate for T4
    // T5 allocates storage for long array
    BOOST_TEST_EQ(db.at<long>().first, db.at<long>().second);
    BOOST_TEST_EQ(db.at<long>().first, 5u);
  }

  // testing pass-through versions of get and visit
  {
    axis::regular<> a(10, 0, 1);
    axis::integer<> b(0, 3);
    const auto& ta = axis::get<axis::regular<>>(a);
    BOOST_TEST_EQ(ta, a);
    const auto* tb = axis::get<axis::integer<>>(&b);
    BOOST_TEST_EQ(tb, &b);
    const auto* tc = axis::get<axis::regular<>>(&b);
    BOOST_TEST_EQ(tc, nullptr);

    axis::visit([&](const auto& x) { BOOST_TEST_EQ(a, x); }, a);
  }

  // iterators
  test_axis_iterator(axis::variant<axis::regular<>>(axis::regular<>(5, 0, 1)), 0, 5);

  // variant of references
  {
    using A = axis::integer<int, axis::null_type>;
    using VARef = axis::variant<A&>;
    auto a = A(1, 5);
    VARef ref(a);
    BOOST_TEST_EQ(ref.size(), 4);
    BOOST_TEST_EQ(ref.value(0), 1);
    // change a through ref
    axis::get<A>(ref) = A(7, 14);
    BOOST_TEST_EQ(a.size(), 7);
    BOOST_TEST_EQ(a.value(0), 7);
  }

  return boost::report_errors();
}
