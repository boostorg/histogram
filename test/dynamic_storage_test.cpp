// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE dynamic_storage_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <boost/histogram/static_storage.hpp>
#include <sstream>
#include <string>
#include <limits>
using namespace boost::histogram;

template <typename Storage1, typename Storage2>
bool operator==(const Storage1& s1, const Storage2& s2)
{
    if (s1.size() != s2.size())
        return false;
    for (std::size_t i = 0, n = s1.size(); i < n; ++i)
        if (s1.value(i) != s2.value(i) || s1.variance(i) != s2.variance(i))
            return false;
    return true;
}

BOOST_AUTO_TEST_CASE(wtype_streamer)
{
    std::ostringstream os;
    detail::wtype w;
    w.w = 1.2;
    w.w2 = 2.3;
    os << w;
    BOOST_CHECK_EQUAL(os.str(), std::string("(1.2,2.3)"));
}

BOOST_AUTO_TEST_CASE(dynamic_storage_grow_1)
{
    double x = 1.0;
    dynamic_storage n(1), p(1);
    n.increase(0);
    for (unsigned i = 0; i < 100; ++i) {
        p = n;
        n += p;
        x += x;
        n.increase(0);
        ++x;
        BOOST_CHECK_EQUAL(n.value(0), x);
        BOOST_CHECK_EQUAL(n.variance(0), x);
    }
}

BOOST_AUTO_TEST_CASE(dynamic_storage_grow_2)
{
    dynamic_storage n(1), p(1);
    n.increase(0);
    for (unsigned i = 0; i < 100; ++i) {
        p = n;
        n += p;
        dynamic_storage a(1);
        a.increase(0, 0.0); // converts to wtype
        a = n;
        BOOST_CHECK_EQUAL(a.value(0), n.value(0));
        BOOST_CHECK_EQUAL(a.variance(0), n.variance(0));
    }
}

BOOST_AUTO_TEST_CASE(dynamic_storage_grow_3)
{
    dynamic_storage n(1), p(1);
    n.increase(0);
    for (unsigned i = 0; i < 100; ++i) {
        p = n;
        n += p;
        dynamic_storage a(1);
        a += n;
        a.increase(0, 0.0); // converts to wtype
        BOOST_CHECK_EQUAL(a.value(0), n.value(0));
        BOOST_CHECK_EQUAL(a.variance(0), n.variance(0));
    }
}

BOOST_AUTO_TEST_CASE(dynamic_storage_add_with_growth)
{
    dynamic_storage a(1), b(1);
    a.increase(0);
    for (unsigned i = 0; i < 100; ++i)
        a += a;
    b.increase(0);
    b += a;
    BOOST_CHECK_EQUAL(b.value(0), a.value(0) + 1.0);
}

BOOST_AUTO_TEST_CASE(dynamic_storage_add_wtype)
{
    dynamic_storage a(1), b(1);
    a.increase(0, 1.0);
    b.increase(0);
    a += b;
    BOOST_CHECK_EQUAL(a.value(0), b.value(0) + 1.0);
}

BOOST_AUTO_TEST_CASE(dynamic_storage_equality)
{
    dynamic_storage a(1), b(1), c(1);
    BOOST_CHECK(a == b);
    a.increase(0);
    BOOST_CHECK(!(a == b));
    c.increase(0, 1.0);
    BOOST_CHECK(a == c);
}

BOOST_AUTO_TEST_CASE(convert_static_storage_1)
{
    dynamic_storage a(2), b(2);
    static_storage<uint8_t> c(2);
    c.increase(0);
    for (unsigned i = 0; i < 7; ++i)
        c += c;
    a = c;
    BOOST_CHECK(a == c);
    BOOST_CHECK(c == a);
    BOOST_CHECK_EQUAL(a.value(0), 128);
    BOOST_CHECK_EQUAL(a.value(1), 0);
    b = std::move(c);
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(convert_static_storage_2)
{
    dynamic_storage a(2), b(2);
    static_storage<uint16_t> c(2);
    c.increase(0);
    for (unsigned i = 0; i < 15; ++i)
        c += c;
    a = c;
    BOOST_CHECK(a == c);
    BOOST_CHECK(c == a);
    BOOST_CHECK_EQUAL(a.value(0), 32768);
    BOOST_CHECK_EQUAL(a.value(1), 0);
    b = std::move(c);
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(convert_static_storage_3)
{
    dynamic_storage a(2), b(2);
    static_storage<uint32_t> c(2);
    c.increase(0);
    for (unsigned i = 0; i < 31; ++i)
        c += c;
    a = c;
    BOOST_CHECK(a == c);
    BOOST_CHECK_EQUAL(a.value(0), 2147483648);
    BOOST_CHECK_EQUAL(a.value(1), 0);
    b = std::move(c);
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(convert_static_storage_4)
{
    dynamic_storage a(2), b(2);
    static_storage<uint64_t> c(2);
    c.increase(0);
    for (unsigned i = 0; i < 63; ++i)
        c += c;
    a = c;
    BOOST_CHECK(a == c);
    BOOST_CHECK_EQUAL(a.value(0), 9223372036854775808lu);
    BOOST_CHECK_EQUAL(a.value(1), 0);
    b = std::move(c);
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(convert_static_storage_5)
{
    dynamic_storage a(2), b(2);
    a.increase(0, 0.0);
    b.increase(0, 0.0);
    static_storage<uint64_t> c(2);
    c.increase(0);
    for (unsigned i = 0; i < 63; ++i)
        c += c;
    a = c;
    BOOST_CHECK(a == c);
    BOOST_CHECK_EQUAL(a.value(0), 9223372036854775808lu);
    BOOST_CHECK_EQUAL(a.value(1), 0);
    b = std::move(c);
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(convert_static_storage_6)
{
    dynamic_storage a(2), b(2);
    static_storage<float> c(2);
    c.increase(0);
    for (unsigned i = 0; i < 15; ++i)
        c += c;
    a = c;
    BOOST_CHECK(a == c);
    BOOST_CHECK_EQUAL(a.value(0), 32768);
    BOOST_CHECK_EQUAL(a.value(1), 0);
    b = std::move(c);
    BOOST_CHECK_EQUAL(b.value(0), 32768);
    BOOST_CHECK_EQUAL(b.value(1), 0);
}
