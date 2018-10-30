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

// TODO: test array, map, boost::accumulator as bin type
template <typename T>
void tests() {
  // ctor, copy, move
  {
    storage_adaptor<T> a;
    a.reset(2);
    storage_adaptor<T> b(a);
    storage_adaptor<T> c;
    c = a;
    BOOST_TEST_EQ(a.size(), 2);
    BOOST_TEST_EQ(b.size(), 2);
    BOOST_TEST_EQ(c.size(), 2);

    storage_adaptor<T> d(std::move(a));
    BOOST_TEST_EQ(d.size(), 2);
    storage_adaptor<T> e;
    e = std::move(d);
    BOOST_TEST_EQ(e.size(), 2);

    T t;
    storage_adaptor<T> g(t); // tests converting ctor
  }

  // increment and reset
  {
    storage_adaptor<T> a, b;
    a.reset(1);
    b.reset(2);
    a(0);
    a(0);
    b(0);
    b(0, 2);
    b(1, 5);
    BOOST_TEST_EQ(a[0], 2);
    BOOST_TEST_EQ(b[0], 3);
    BOOST_TEST_EQ(b[1], 5);
    a.reset(2);
    b.reset(1);
    BOOST_TEST_EQ(a.size(), 2);
    BOOST_TEST_EQ(b.size(), 1);
    BOOST_TEST_EQ(a[0], 0);
    BOOST_TEST_EQ(a[1], 0);
    BOOST_TEST_EQ(b[0], 0);
  }

  // multiply
  {
    storage_adaptor<T> a;
    a.reset(2);
    a(0);
    a *= 3;
    BOOST_TEST_EQ(a[0], 3);
    BOOST_TEST_EQ(a[1], 0);
    a(1, 2);
    BOOST_TEST_EQ(a[0], 3);
    BOOST_TEST_EQ(a[1], 2);
    a *= 3;
    BOOST_TEST_EQ(a[0], 9);
    BOOST_TEST_EQ(a[1], 6);
  }

  // copy
  {
    storage_adaptor<T> a;
    a.reset(1);
    a(0);
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
  }

  // move
  {
    storage_adaptor<T> a;
    a.reset(1);
    a(0);
    decltype(a) b;
    BOOST_TEST(!(a == b));
    b = std::move(a);
    BOOST_TEST_EQ(b.size(), 1);
    BOOST_TEST_EQ(b[0], 1);
    decltype(a) c(std::move(b));
    BOOST_TEST_EQ(c.size(), 1);
    BOOST_TEST_EQ(c[0], 1);
  }

  // add
  {
    storage_adaptor<T> a;
    a.reset(2);
    a(1);
    auto b = a;
    b += a;
    BOOST_TEST_EQ(b[0], 0);
    BOOST_TEST_EQ(b[1], 2);
    a += a;
    // also test self-add
    BOOST_TEST_EQ(a[0], 0);
    BOOST_TEST_EQ(a[1], 2);
  }

  // multiply
  {
    storage_adaptor<T> a;
    a.reset(2);
    a(1);
    a *= 2;
    BOOST_TEST_EQ(a[0], 0);
    BOOST_TEST_EQ(a[1], 2);
  }
}

template <typename A, typename B>
void mixed_tests() {
  // comparison
  {
    A a, b;
    a.reset(1);
    b.reset(1);
    B c, d;
    c.reset(1);
    d.reset(2);
    a(0);
    b(0);
    c(0);
    c(0);
    d(0);
    d(1, 5);
    d(0, 2);
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

  // ctor, copy, move
  {
    A a;
    a.reset(2);
    a(1);
    B b(a);
    B c;
    c = a;
    BOOST_TEST_EQ(c[0], 0);
    BOOST_TEST_EQ(c[1], 1);
    B d(std::move(a));
    B e;
    e = std::move(d);
    BOOST_TEST_EQ(e[0], 0);
    BOOST_TEST_EQ(e[1], 1);
  }
}

int main() {
  tests<std::vector<unsigned>>();
  tests<std::array<unsigned, 100>>();
  // tests<std::map<std::size_t, unsigned>>();

  mixed_tests<storage_adaptor<std::vector<unsigned>>,
              storage_adaptor<std::array<double, 100>>>();
  mixed_tests<adaptive_storage<>, storage_adaptor<std::vector<unsigned>>>();
  mixed_tests<storage_adaptor<std::vector<unsigned>>, adaptive_storage<>>();

  // with weight_counter
  {
    storage_adaptor<std::vector<weight_counter<double>>> a;
    a.reset(1);
    a(0);
    a(0, 1);
    a(0, weight_counter<double>(1, 0));
    BOOST_TEST_EQ(a[0].value(), 3);
    BOOST_TEST_EQ(a[0].variance(), 2);
    a(0, weight(2));
    BOOST_TEST_EQ(a[0].value(), 5);
    BOOST_TEST_EQ(a[0].variance(), 6);
  }

  return boost::report_errors();
}
