// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include "utility.hpp"

using namespace boost::histogram;

template <typename T1, typename T2>
void run_tests() {
  // compare
  {
    auto a = make(T1{}, axis::regular<>{3, 0, 3}, axis::integer<>(0, 2));
    auto b = make_s(T2{}, array_storage<int>(), axis::regular<>{3, 0, 3},
                    axis::integer<>(0, 2));
    BOOST_TEST_EQ(a, b);
    auto b2 = make(T2{}, axis::integer<>{0, 3}, axis::integer<>(0, 2));
    BOOST_TEST_NE(a, b2);
    auto b3 = make(T2{}, axis::regular<>(3, 0, 4), axis::integer<>(0, 2));
    BOOST_TEST_NE(a, b3);
  }

  // add
  {
    auto a = make(T1{}, axis::integer<>{0, 2});
    auto b = make(T2{}, axis::integer<>{0, 2});
    BOOST_TEST_EQ(a, b);
    a(0);   // 1 0
    b(1);   // 0 1
    a += b; // 1 1
    BOOST_TEST_EQ(a[0], 1);
    BOOST_TEST_EQ(a[1], 1);

    auto c = make(T2{}, axis::integer<>{0, 3});
    BOOST_TEST_THROWS(a += c, std::invalid_argument);
  }

  // copy_assign
  {
    auto a = make(T1{}, axis::regular<>{3, 0, 3}, axis::integer<>(0, 2));
    auto b = make_s(T2{}, array_storage<int>(), axis::regular<>{3, 0, 3},
                    axis::integer<>(0, 2));
    a(1, 1);
    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
  }
}

int main() {
  run_tests<static_tag, dynamic_tag>();
  run_tests<dynamic_tag, static_tag>();

  return boost::report_errors();
}
