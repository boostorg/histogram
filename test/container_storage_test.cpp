// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE container_storage_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/container_storage.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <limits>
#include <vector>
#include <deque>
#include <array>
#include <boost/array.hpp>
using namespace boost::histogram;

BOOST_AUTO_TEST_CASE(ctor)
{
    container_storage<std::vector<unsigned>> a(1);
    BOOST_CHECK_EQUAL(a.size(), 1);
    BOOST_CHECK_EQUAL(a.value(0), 0);
    container_storage<std::deque<unsigned>> b(1);
    BOOST_CHECK_EQUAL(b.size(), 1);
    BOOST_CHECK_EQUAL(b.value(0), 0);
    container_storage<std::array<unsigned, 1>> c(1);
    BOOST_CHECK_EQUAL(c.size(), 1);
    BOOST_CHECK_EQUAL(c.value(0), 0);
    container_storage<boost::array<unsigned, 1>> d(1);
    BOOST_CHECK_EQUAL(d.size(), 1);
    BOOST_CHECK_EQUAL(d.value(0), 0);
}

BOOST_AUTO_TEST_CASE(increase)
{
    container_storage<std::vector<unsigned>> a(1), b(1);
    container_storage<std::vector<unsigned char>> c(1), d(2);
    a.increase(0);
    b.increase(0);
    c.increase(0); c.increase(0);
    d.increase(0);
    BOOST_CHECK_EQUAL(a.value(0), 1.0);
    BOOST_CHECK_EQUAL(b.value(0), 1.0);
    BOOST_CHECK_EQUAL(c.value(0), 2.0);
    BOOST_CHECK_EQUAL(d.value(0), 1.0);
    BOOST_CHECK_EQUAL(d.value(1), 0.0);
    BOOST_CHECK(a == a);
    BOOST_CHECK(a == b);
    BOOST_CHECK(!(a == c));
    BOOST_CHECK(!(a == d));

    container_storage<std::array<unsigned, 2>> e;
    e.increase(0);
    BOOST_CHECK(d == e);
    e.increase(1);
    BOOST_CHECK(!(d == e));
}

BOOST_AUTO_TEST_CASE(copy)
{
    container_storage<std::vector<unsigned>> a(1);
    a.increase(0);
    decltype(a) b(2);
    BOOST_CHECK(!(a == b));
    b = a;
    BOOST_CHECK(a == b);
    BOOST_CHECK_EQUAL(b.size(), 1);
    BOOST_CHECK_EQUAL(b.value(0), 1.0);

    decltype(a) c(a);
    BOOST_CHECK(a == c);
    BOOST_CHECK_EQUAL(c.size(), 1);
    BOOST_CHECK_EQUAL(c.value(0), 1.0);

    container_storage<std::vector<unsigned char>> d(1);
    BOOST_CHECK(!(a == d));
    d = a;
    BOOST_CHECK(a == d);
    decltype(d) e(a);
    BOOST_CHECK(a == e);

    container_storage<std::array<unsigned, 1>> f;
    BOOST_CHECK(!(a == f));
    f = a;
    BOOST_CHECK(f == a);
    decltype(f) g(a);
    BOOST_CHECK(g == a);

    container_storage<std::array<unsigned char, 1>> h;
    BOOST_CHECK(!(h == f));
    h = f;
    BOOST_CHECK(h == f);
    decltype(h) i(f);
    BOOST_CHECK(i == f);
}

BOOST_AUTO_TEST_CASE(move)
{
    container_storage<std::vector<unsigned>> a(1);
    a.increase(0);
    decltype(a) b;
    BOOST_CHECK(!(a == b));
    b = std::move(a);
    BOOST_CHECK_EQUAL(a.size(), 0);
    BOOST_CHECK_EQUAL(b.size(), 1);
    BOOST_CHECK_EQUAL(b.value(0), 1.0);
    decltype(a) c(std::move(b));
    BOOST_CHECK_EQUAL(c.size(), 1);
    BOOST_CHECK_EQUAL(c.value(0), 1.0);
}
