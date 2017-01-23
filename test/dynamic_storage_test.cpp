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
#include <boost/histogram/utility.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <sstream>
#include <string>
#include <limits>
using namespace boost::histogram;

template <typename Storage1, typename Storage2>
bool operator==(const Storage1& a, const Storage2& b)
{
    if (a.size() != b.size())
        return false;
    return boost::histogram::detail::storage_content_equal(a, b);
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

template <typename T>
void dynamic_storage_increase_and_grow_impl()
{
    dynamic_storage n(1);
    BOOST_CHECK_EQUAL(n.value(0), 0.0);
    n.increase(0);
    while (n.depth() != sizeof(T))
        n += n;
    const auto tmax = std::numeric_limits<T>::max();
    void* buf = const_cast<void*>(n.data());
    static_cast<T*>(buf)[0] = tmax - 1;
    auto n2 = n;
    BOOST_CHECK_EQUAL(n.value(0), double(tmax - 1));
    BOOST_CHECK_EQUAL(n2.value(0), double(tmax - 1));
    dynamic_storage x(1);
    x.increase(0);
    n.increase(0);
    n2 += x;
    if (sizeof(T) == sizeof(uint64_t)) {
        BOOST_CHECK_THROW(n.increase(0), std::overflow_error);
        BOOST_CHECK_THROW(n2 += x, std::overflow_error);
    } else {
        n.increase(0);
        n2 += x;
        double v = tmax;
        ++v;
        BOOST_CHECK_EQUAL(n.value(0), v);
        BOOST_CHECK_EQUAL(n2.value(0), v);
        BOOST_CHECK(!(n.value(0) == double(tmax)));
        BOOST_CHECK(!(n2.value(0) == double(tmax)));
    }
}

BOOST_AUTO_TEST_CASE(dynamic_storage_increase_and_grow)
{
    dynamic_storage_increase_and_grow_impl<uint8_t>();
    dynamic_storage_increase_and_grow_impl<uint16_t>();
    dynamic_storage_increase_and_grow_impl<uint32_t>();
    dynamic_storage_increase_and_grow_impl<uint64_t>();
}

BOOST_AUTO_TEST_CASE(dynamic_storage_add_and_grow)
{
    dynamic_storage a(1);
    a.increase(0);
    double x = 1.0;
    dynamic_storage y(1);
    BOOST_CHECK_EQUAL(y.depth(), 0);
    BOOST_CHECK_EQUAL(y.value(0), 0.0);
    a += y;
    BOOST_CHECK_EQUAL(a.value(0), x);
    for (unsigned i = 0; i < 63; ++i) {
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
void convert_static_storage_impl() {
    dynamic_storage a(2), b(2);
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
    BOOST_CHECK(a == b);
    a += s;
    BOOST_CHECK(a == s);
    BOOST_CHECK(s == a);
    dynamic_storage c(s);
    BOOST_CHECK(c == s);
    BOOST_CHECK(s == c);
    dynamic_storage d;
    d = std::move(s); // cannot move, uses copy
    BOOST_CHECK(d == s);
    BOOST_CHECK(s == d);
}

BOOST_AUTO_TEST_CASE(convert_static_storage)
{
    convert_static_storage_impl<uint8_t>();
    convert_static_storage_impl<uint16_t>();
    convert_static_storage_impl<uint32_t>();
    convert_static_storage_impl<uint64_t>();
}
