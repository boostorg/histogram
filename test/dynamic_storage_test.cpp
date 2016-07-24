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

BOOST_AUTO_TEST_CASE(equal_operator)
{
    dynamic_storage a(1), b(1), c(1), d(2);
    a.increase(0);
    b.increase(0);
    c.increase(0);
    c.increase(0);
    d.increase(0);
    BOOST_CHECK(a == a);
    BOOST_CHECK(a == b);
    BOOST_CHECK(!(a == c));
    BOOST_CHECK(!(a == d));
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

BOOST_AUTO_TEST_CASE(dynamic_storage_increase_and_grow)
{
    dynamic_storage n(1);
    for (unsigned b = 1; b <= 8; b *= 2) {
        uint64_t x = 0;
        for (unsigned t = 1; t < (8 * b); ++t) { ++x; x <<= 1; }
        void* buf = const_cast<void*>(n.data());
        switch (b) {
            case 1: (static_cast<uint8_t*>(buf))[0] = x;
            case 2: (static_cast<uint16_t*>(buf))[0] = x;
            case 4: (static_cast<uint32_t*>(buf))[0] = x;
            case 8: (static_cast<uint64_t*>(buf))[0] = x;
        }
        ++x;
        n.increase(0);
        double v = x;
        BOOST_CHECK_EQUAL(n.value(0), v);
        n.increase(0);
        ++v;
        BOOST_CHECK_EQUAL(n.value(0), v);
        n.increase(0);
        ++v;
        BOOST_CHECK_EQUAL(n.value(0), v);
    }
}

BOOST_AUTO_TEST_CASE(dynamic_storage_add_and_grow)
{
    dynamic_storage a(1);
    a.increase(0);
    double x = 1.0;
    for (unsigned i = 0; i < 100; ++i) {
        a += a;
        x += x;
        dynamic_storage b(1);
        b += a;
        BOOST_CHECK_EQUAL(a.value(0), x);
        BOOST_CHECK_EQUAL(a.variance(0), x);
        BOOST_CHECK_EQUAL(b.value(0), x);
        BOOST_CHECK_EQUAL(b.variance(0), x);
        b.increase(0, 0.0);
        BOOST_CHECK_EQUAL(b.value(0), x);
        BOOST_CHECK_EQUAL(b.variance(0), x);
        dynamic_storage c(1);
        c.increase(0, 0.0);
        c += a;
        BOOST_CHECK_EQUAL(c.value(0), x);
        BOOST_CHECK_EQUAL(c.variance(0), x);
    }
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

template <typename T>
void convert_static_storage_test() {
    dynamic_storage a(2), b(2), c(2);
    static_storage<T> s(2);
    a.increase(0);
    b.increase(0, 1.0);
    s.increase(0);
    for (unsigned i = 0; i < (8 * sizeof(T) - 1); ++i)
        s += s;
    a = s;
    b = s;
    BOOST_CHECK(a == s);
    BOOST_CHECK(s == a);
    BOOST_CHECK(b == s);
    BOOST_CHECK(s == b);
    a = dynamic_storage(2);
    b = dynamic_storage(2);
    a += s;
    b += s;
    BOOST_CHECK(a == s);
    BOOST_CHECK(s == a);
    BOOST_CHECK(b == s);
    BOOST_CHECK(s == b);
    dynamic_storage d;
    d = std::move(s);
    BOOST_CHECK(a == d);
}

BOOST_AUTO_TEST_CASE(convert_static_storage)
{
    convert_static_storage_test<uint8_t>();
    convert_static_storage_test<uint16_t>();
    convert_static_storage_test<uint32_t>();
    convert_static_storage_test<uint64_t>();
}

BOOST_AUTO_TEST_CASE(convert_static_storage_1)
{
    dynamic_storage a(2), b(2);
    static_storage<double> s(2);
    s.increase(0);
    for (unsigned i = 0; i < 100; ++i)
        s += s;
    a = s;
    BOOST_CHECK(a == s);
    BOOST_CHECK(s == a);
    BOOST_CHECK_EQUAL(a.value(0), 1.2676506002282294e+30);
    BOOST_CHECK_EQUAL(a.value(1), 0.);
    BOOST_CHECK_EQUAL(a.variance(0), 1.2676506002282294e+30);
    BOOST_CHECK_EQUAL(a.variance(1), 0.);
    b = std::move(s);
    BOOST_CHECK(b == s);
    BOOST_CHECK(s == b);
}
