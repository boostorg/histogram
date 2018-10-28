// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/adaptive_storage.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/weight_counter.hpp>
#include <limits>
#include <vector>

using namespace boost::histogram;

template <typename T>
using vector_storage = storage_adaptor<std::vector<T>>;
template <typename T>
using array_storage = storage_adaptor<std::array<100, T>>;

int main() {
  // ctor, copy, move
  {
    vector_storage<unsigned> a;
    vector_storage<unsigned> b(a);
    vector_storage<unsigned> c;
    c = a;
    vector_storage<unsigned> d(std::move(a));
    vector_storage<unsigned> e;
    e = std::move(b);
    vector_storage<int> f(e);

    vector_storage<unsigned> g(std::vector<unsigned>(10, 0));

    array_storage<unsigned> h;
    h = g;
    array_storage<unsigned> i(g);
  }

  // increment and reset
  {
    vector_storage<unsigned> a, b;
    a.reset(1);
    b.reset(1);
    vector_storage<unsigned char> c, d;
    c.reset(1);
    d.reset(2);
    a.increment(0);
    b.increment(0);
    c.increment(0);
    c.increment(0);
    d.increment(0);
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
    vector_storage<double> a;
    a.reset(2);
    a.increment(0);
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
    vector_storage<unsigned> a;
    a.reset(1);
    a.increment(0);
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

    vector_storage<unsigned char> d;
    d.reset(1);
    BOOST_TEST(!(a == d));
    d = a;
    BOOST_TEST(a == d);
    decltype(d) e(a);
    BOOST_TEST(a == e);
  }

  // move
  {
    vector_storage<unsigned> a;
    a.reset(1);
    a.increment(0);
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
    vector_storage<weight_counter<double>> a;
    a.reset(1);
    a.increment(0);
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
