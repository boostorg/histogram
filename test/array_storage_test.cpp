// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <limits>

int main() {
  using namespace boost::histogram;

  // ctor and reset
  {
    array_storage<unsigned> a;
    BOOST_TEST_EQ(a.size(), 0);
    a.reset(1);
    BOOST_TEST_EQ(a.size(), 1);
    a.increase(0);
    BOOST_TEST_EQ(a[0], 1);
    a.reset(1);
    BOOST_TEST_EQ(a[0], 0);
  }

  // increase
  {
    array_storage<unsigned> a, b;
    a.reset(1);
    b.reset(1);
    array_storage<unsigned char> c, d;
    c.reset(1);
    d.reset(2);
    a.increase(0);
    b.increase(0);
    c.increase(0);
    c.increase(0);
    d.increase(0);
    d.add(1, 5);
    d.add(0, 2);
    BOOST_TEST_EQ(a[0], 1);
    BOOST_TEST_EQ(b[0], 1);
    BOOST_TEST_EQ(c[0], 2);
    BOOST_TEST_EQ(d[0], 3);
    BOOST_TEST_EQ(d[1], 5);
    BOOST_TEST(a == a);
    BOOST_TEST(a == b);
    BOOST_TEST(!(a == c));
    BOOST_TEST(!(a == d));
  }

  // multiply
  {
    array_storage<double> a;
    a.reset(2);
    a.increase(0);
    a *= 3;
    BOOST_TEST_EQ(a[0], 3);
    BOOST_TEST_EQ(a[1], 0);
    a.add(1, 2);
    BOOST_TEST_EQ(a[0], 3);
    BOOST_TEST_EQ(a[1], 2);
    a *= 3;
    BOOST_TEST_EQ(a[0], 9);
    BOOST_TEST_EQ(a[1], 6);
  }

  // copy
  {
    array_storage<unsigned> a;
    a.reset(1);
    a.increase(0);
    decltype(a) b;
    b.reset(2);
    BOOST_TEST(!(a == b));
    b = a;
    BOOST_TEST(a == b);
    BOOST_TEST_EQ(b.size(), 1);
    BOOST_TEST_EQ(b[0], 1);

    decltype(a) c(a);
    BOOST_TEST(a == c);
    BOOST_TEST_EQ(c.size(), 1);
    BOOST_TEST_EQ(c[0], 1);

    array_storage<unsigned char> d;
    d.reset(1);
    BOOST_TEST(!(a == d));
    d = a;
    BOOST_TEST(a == d);
    decltype(d) e(a);
    BOOST_TEST(a == e);
  }

  // move
  {
    array_storage<unsigned> a;
    a.reset(1);
    a.increase(0);
    decltype(a) b;
    BOOST_TEST(!(a == b));
    b = std::move(a);
    BOOST_TEST_EQ(a.size(), 0);
    BOOST_TEST_EQ(b.size(), 1);
    BOOST_TEST_EQ(b[0], 1);
    decltype(a) c(std::move(b));
    BOOST_TEST_EQ(c.size(), 1);
    BOOST_TEST_EQ(c[0], 1);
    BOOST_TEST_EQ(b.size(), 0);
  }

  // with weight_counter
  {
    array_storage<weight_counter<double>> a;
    a.reset(1);
    a.increase(0);
    a.add(0, 1);
    a.add(0, weight_counter<double>(1, 0));
    BOOST_TEST_EQ(a[0].value(), 3);
    BOOST_TEST_EQ(a[0].variance(), 2);
    a.add(0, weight(2));
    BOOST_TEST_EQ(a[0].value(), 5);
    BOOST_TEST_EQ(a[0].variance(), 6);
  }

  return boost::report_errors();
}
