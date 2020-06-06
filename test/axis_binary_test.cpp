// Copyright 2020 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/binary.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <limits>
#include <sstream>
#include <type_traits>
#include "std_ostream.hpp"
#include "throw_exception.hpp"
#include "utility_axis.hpp"
#include "utility_str.hpp"

int main() {
  using namespace boost::histogram;

  BOOST_TEST(std::is_nothrow_move_assignable<axis::binary<>>::value);
  BOOST_TEST(std::is_nothrow_move_constructible<axis::binary<>>::value);

  // axis::integer with double type
  {
    axis::binary<> a{"foo"};
    BOOST_TEST_EQ(a.metadata(), "foo");
    a.metadata() = "bar";
    const auto& cref = a;
    BOOST_TEST_EQ(cref.metadata(), "bar");
    cref.metadata() = "foo";
    BOOST_TEST_EQ(cref.metadata(), "foo");

    BOOST_TEST_EQ(a.index(true), 1);
    BOOST_TEST_EQ(a.index(false), 0);
    BOOST_TEST_EQ(a.index(1), 1);
    BOOST_TEST_EQ(a.index(0), 0);

    BOOST_TEST_CSTR_EQ(str(a).c_str(), "binary(metadata=\"foo\")");

    axis::binary<> b;
    BOOST_TEST_CSTR_EQ(str(b).c_str(), "binary()");

    BOOST_TEST_NE(a, b);
    b = a;
    BOOST_TEST_EQ(a, b);
    axis::binary<> c = std::move(b);
    BOOST_TEST_EQ(c, a);
    axis::binary<> d;
    BOOST_TEST_NE(c, d);
    d = std::move(c);
    BOOST_TEST_EQ(d, a);
  }

  // iterators
  test_axis_iterator(axis::binary<>(), 0, 2);

  return boost::report_errors();
}
