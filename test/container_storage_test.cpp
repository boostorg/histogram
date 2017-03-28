// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/array.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/container_storage.hpp>
#include <deque>
#include <limits>
#include <vector>

int main() {
  using namespace boost::histogram;

  // ctor
  {
    container_storage<std::vector<unsigned>> a(1);
    BOOST_TEST_EQ(a.size(), 1u);
    BOOST_TEST_EQ(a.value(0), 0u);
    container_storage<std::deque<unsigned>> b(1);
    BOOST_TEST_EQ(b.size(), 1u);
    BOOST_TEST_EQ(b.value(0), 0u);
    container_storage<std::array<unsigned, 1>> c(1);
    BOOST_TEST_EQ(c.size(), 1u);
    BOOST_TEST_EQ(c.value(0), 0u);
    container_storage<boost::array<unsigned, 1>> d(1);
    BOOST_TEST_EQ(d.size(), 1u);
    BOOST_TEST_EQ(d.value(0), 0u);
  }

  // increase
  {
    container_storage<std::vector<unsigned>> a(1), b(1);
    container_storage<std::vector<unsigned char>> c(1), d(2);
    a.increase(0);
    b.increase(0);
    c.increase(0);
    c.increase(0);
    d.increase(0);
    BOOST_TEST_EQ(a.value(0), 1u);
    BOOST_TEST_EQ(b.value(0), 1u);
    BOOST_TEST_EQ(c.value(0), 2u);
    BOOST_TEST_EQ(d.value(0), 1u);
    BOOST_TEST_EQ(d.value(1), 0u);
    BOOST_TEST(a == a);
    BOOST_TEST(a == b);
    BOOST_TEST(!(a == c));
    BOOST_TEST(!(a == d));

    container_storage<std::array<unsigned, 2>> e;
    e.increase(0);
    BOOST_TEST(d == e);
    e.increase(1);
    BOOST_TEST(!(d == e));
  }

  // copy
  {
    container_storage<std::vector<unsigned>> a(1);
    a.increase(0);
    decltype(a) b(2);
    BOOST_TEST(!(a == b));
    b = a;
    BOOST_TEST(a == b);
    BOOST_TEST_EQ(b.size(), 1u);
    BOOST_TEST_EQ(b.value(0), 1u);

    decltype(a) c(a);
    BOOST_TEST(a == c);
    BOOST_TEST_EQ(c.size(), 1u);
    BOOST_TEST_EQ(c.value(0), 1u);

    container_storage<std::vector<unsigned char>> d(1);
    BOOST_TEST(!(a == d));
    d = a;
    BOOST_TEST(a == d);
    decltype(d) e(a);
    BOOST_TEST(a == e);

    container_storage<std::array<unsigned, 1>> f;
    BOOST_TEST(!(a == f));
    f = a;
    BOOST_TEST(f == a);
    decltype(f) g(a);
    BOOST_TEST(g == a);

    container_storage<std::array<unsigned char, 1>> h;
    BOOST_TEST(!(h == f));
    h = f;
    BOOST_TEST(h == f);
    decltype(h) i(f);
    BOOST_TEST(i == f);
  }

  // move
  {
    container_storage<std::vector<unsigned>> a(1);
    a.increase(0);
    decltype(a) b;
    BOOST_TEST(!(a == b));
    b = std::move(a);
    BOOST_TEST_EQ(a.size(), 0u);
    BOOST_TEST_EQ(b.size(), 1u);
    BOOST_TEST_EQ(b.value(0), 1u);
    decltype(a) c(std::move(b));
    BOOST_TEST_EQ(c.size(), 1u);
    BOOST_TEST_EQ(c.value(0), 1u);
    BOOST_TEST_EQ(b.size(), 0u);
  }

  return boost::report_errors();
}
