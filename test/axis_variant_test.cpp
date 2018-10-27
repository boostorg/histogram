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
#include <boost/histogram/histogram_fwd.hpp>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "utility.hpp"

using namespace boost::histogram;

int main() {
  {
    BOOST_TEST_THROWS(axis::integer<>(1, 1), std::invalid_argument);
  }

  {
    axis::variant<
      axis::integer<>,
      axis::category<std::string>
    > a{axis::integer<>(0, 2, "int")};
    BOOST_TEST_EQ(a(-10), -1);
    BOOST_TEST_EQ(a(-1), -1);
    BOOST_TEST_EQ(a(0), 0);
    BOOST_TEST_EQ(a(0.5), 0);
    BOOST_TEST_EQ(a(1), 1);
    BOOST_TEST_EQ(a(2), 2);
    BOOST_TEST_EQ(a(10), 2);
    BOOST_TEST_EQ(a[-1].lower(), -std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a[a.size()].upper(), std::numeric_limits<double>::infinity());
    BOOST_TEST_EQ(a.metadata(), std::string("int"));
    BOOST_TEST_EQ(a.options(), axis::option_type::underflow_and_overflow);

    a = axis::category<std::string>({"A", "B"}, "cat");
    BOOST_TEST_EQ(a("A"), 0);
    BOOST_TEST_EQ(a("B"), 1);
    BOOST_TEST_EQ(a.metadata(), std::string("cat"));
    BOOST_TEST_EQ(a.options(), axis::option_type::overflow);
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
    BOOST_TEST_TRAIT_TRUE(
        (std::is_same<decltype(axis.metadata()), axis::empty_metadata_type&>));
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

    namespace tr = axis::transform;
    test(axis::regular<>(2, -1, 1, "regular1"),
         "regular(2, -1, 1, metadata=\"regular1\", options=underflow_and_overflow)");
    test(axis::regular<tr::log<>>(2, 1, 10, "regular2", axis::option_type::none),
         "regular_log(2, 1, 10, metadata=\"regular2\", options=none)");
    test(axis::regular<tr::pow<>>(1.5, 2, 1, 10, "regular3", axis::option_type::overflow),
         "regular_pow(2, 1, 10, metadata=\"regular3\", options=overflow, power=1.5)");
    test(axis::regular<tr::pow<>>(-1.5, 2, 1, 10, "regular4", axis::option_type::none),
         "regular_pow(2, 1, 10, metadata=\"regular4\", options=none, power=-1.5)");
    test(axis::circular<double, axis::empty_metadata_type>(4, 0.1, 1.0),
         "circular(4, 0.1, 1.1, options=overflow)");
    test(axis::variable<>({-1, 0, 1}, "variable", axis::option_type::none),
         "variable(-1, 0, 1, metadata=\"variable\", options=none)");
    test(axis::category<>({0, 1, 2}, "category"),
         "category(0, 1, 2, metadata=\"category\", options=overflow)");
    test(axis::category<std::string>({"A", "B"}, "category2"),
         "category(\"A\", \"B\", metadata=\"category2\", options=overflow)");
#ifndef _MSC_VER // fails on MSVC because demagnled name for user_defined looks different
    test(axis::integer<int, user_defined>(-1, 1, {}, axis::option_type::none),
         "integer(-1, 1, metadata=main::user_defined, options=none)");
#endif
  }

  // bin_type streamable
  {
    auto test = [](const auto& x, const char* ref) {
      std::ostringstream os;
      os << x;
      BOOST_TEST_EQ(os.str(), std::string(ref));
    };

    auto b = axis::category<>({1, 2});
    test(b[0], "1");
  }

  // axis::variant equal_comparable
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
    axis::variant<axis::category<std::string>> x =
        axis::category<std::string>({"A", "B"}, "category");
    BOOST_TEST_EQ(x("B"), 1);
  }

  {
    auto a = axis::variant<axis::category<>>(axis::category<>({2, 1, 3}));
    BOOST_TEST_THROWS(a[0].lower(), std::runtime_error);    
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
      axes.emplace_back(
          T3({0., 1., 2.}, {}, axis::option_type::underflow_and_overflow, a));
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

  // iterators
  {
    test_axis_iterator(axis::variant<axis::regular<>>(axis::regular<>(5, 0, 1)), 0, 5);
  }

  return boost::report_errors();
}
