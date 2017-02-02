// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE adaptive_storage_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/container_storage.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <sstream>
#include <limits>
using namespace boost::histogram;

namespace boost {
namespace histogram {

template <typename T>
adaptive_storage<> prepare(unsigned n=1) {
    adaptive_storage<> s(n);
    s.increase(0);
    const auto tmax = std::numeric_limits<T>::max();
    while (s.value(0) < 0.1 * tmax)
        s += s;
    return s;
}

template <>
adaptive_storage<> prepare<void>(unsigned n) {
    adaptive_storage<> s(n);
    return s;
}

template <>
adaptive_storage<> prepare<detail::weight>(unsigned n) {
    adaptive_storage<> s(n);
    s.increase(0, 1.0);
    return s;
}

template <>
adaptive_storage<> prepare<detail::mp_int>(unsigned n) {
    adaptive_storage<> s(n);
    s.increase(0);
    const auto tmax = std::numeric_limits<uint64_t>::max();
    while (s.value(0) <= tmax)
        s += s;
    return s;
}

struct storage_access {
    template <typename T>
    static adaptive_storage<>
    set_value(unsigned n, T x) {
        adaptive_storage<> s = prepare<T>(n);
        static_cast<T*>(s.buffer_.ptr_)[0] = x;
        return s;
    }
};

}
}

template <typename T>
void copy_impl() {
    const auto b = prepare<T>(1);
    auto a(b);
    BOOST_CHECK(a == b);
    a = b;
    BOOST_CHECK(a == b);
    a.increase(0);
    BOOST_CHECK(!(a == b));
    a = b;
    BOOST_CHECK(a == b);
    a = prepare<T>(2);
    BOOST_CHECK(!(a == b));
    a = b;
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(copy)
{
    copy_impl<detail::weight>();
    copy_impl<void>();
    copy_impl<uint8_t>();
    copy_impl<uint16_t>();
    copy_impl<uint32_t>();
    copy_impl<uint64_t>();
    copy_impl<detail::mp_int>();
}

template <typename T>
void equal_impl() {
    adaptive_storage<> a(1);
    auto b = storage_access::set_value(1, T(0));
    BOOST_CHECK_EQUAL(a.value(0), 0.0);
    BOOST_CHECK_EQUAL(a.variance(0), 0.0);
    BOOST_CHECK(a == b);
    b.increase(0);
    BOOST_CHECK(!(a == b));

    container_storage<std::vector<unsigned>> c(1);
    auto d = storage_access::set_value(1, T(0));
    BOOST_CHECK(c == d);
    c.increase(0);
    BOOST_CHECK(!(c == d));
}

template <>
void equal_impl<void>() {
    adaptive_storage<> a(1);
    adaptive_storage<> b(1);
    BOOST_CHECK_EQUAL(a.value(0), 0.0);
    BOOST_CHECK_EQUAL(a.variance(0), 0.0);
    BOOST_CHECK(a == b);
    b.increase(0);
    BOOST_CHECK(!(a == b));
}

BOOST_AUTO_TEST_CASE(equal_operator)
{
    equal_impl<detail::weight>();
    equal_impl<void>();
    equal_impl<uint8_t>();
    equal_impl<uint16_t>();
    equal_impl<uint32_t>();
    equal_impl<uint64_t>();
    equal_impl<detail::mp_int>();

    // special case
    adaptive_storage<> a = storage_access::set_value(1, unsigned(0));
    adaptive_storage<> b(1);
}

template <typename T>
void increase_and_grow_impl()
{
    auto tmax = std::numeric_limits<T>::max();
    adaptive_storage<> s = storage_access::set_value<T>(2, tmax - 1);
    auto n = s;
    auto n2 = s;

    n.increase(0);
    n.increase(0);

    adaptive_storage<> x(2);
    x.increase(0);
    n2 += x;
    n2 += x;

    double v = tmax;
    ++v;
    BOOST_CHECK_EQUAL(n.value(0), v);
    BOOST_CHECK_EQUAL(n2.value(0), v);
    BOOST_CHECK_EQUAL(n.value(1), 0.0);
    BOOST_CHECK_EQUAL(n2.value(1), 0.0);
}

template <>
void increase_and_grow_impl<void>()
{
    adaptive_storage<> s(2);
    s.increase(0);
    BOOST_CHECK_EQUAL(s.value(0), 1.0);
    BOOST_CHECK_EQUAL(s.value(1), 0.0);
}

BOOST_AUTO_TEST_CASE(increase_and_grow)
{
    increase_and_grow_impl<void>();
    increase_and_grow_impl<uint8_t>();
    increase_and_grow_impl<uint16_t>();
    increase_and_grow_impl<uint32_t>();
    increase_and_grow_impl<uint64_t>();

    // only increase for mp_int
    auto a = storage_access::set_value(2, detail::mp_int(1));
    BOOST_CHECK_EQUAL(a.value(0), 1.0);
    BOOST_CHECK_EQUAL(a.value(1), 0.0);
    a.increase(0);
    BOOST_CHECK_EQUAL(a.value(0), 2.0);
    BOOST_CHECK_EQUAL(a.value(1), 0.0);
}

BOOST_AUTO_TEST_CASE(add_and_grow)
{
    adaptive_storage<> a(1);
    a.increase(0);
    double x = 1.0;
    adaptive_storage<> y(1);
    BOOST_CHECK_EQUAL(y.value(0), 0.0);
    a += y;
    BOOST_CHECK_EQUAL(a.value(0), x);
    for (unsigned i = 0; i < 80; ++i) {
        a += a;
        x += x;
        adaptive_storage<> b(1);
        b += a;
        BOOST_CHECK_EQUAL(a.value(0), x);
        BOOST_CHECK_EQUAL(a.variance(0), x);
        BOOST_CHECK_EQUAL(b.value(0), x);
        BOOST_CHECK_EQUAL(b.variance(0), x);
        b.increase(0, 0.0);
        BOOST_CHECK_EQUAL(b.value(0), x);
        BOOST_CHECK_EQUAL(b.variance(0), x);
        adaptive_storage<> c(1);
        c.increase(0, 0.0);
        c += a;
        BOOST_CHECK_EQUAL(c.value(0), x);
        BOOST_CHECK_EQUAL(c.variance(0), x);
    }
}

template <typename T>
void convert_container_storage_impl() {
    const auto aref = storage_access::set_value(1, T(0));
    BOOST_CHECK_EQUAL(aref.value(0), 0.0);
    container_storage<std::vector<unsigned>> s(1);
    s.increase(0);

    auto a = aref;
    a = s;
    BOOST_CHECK_EQUAL(a.value(0), 1.0);
    BOOST_CHECK(a == s);
    a.increase(0);
    BOOST_CHECK(!(a == s));

    adaptive_storage<> b(s);
    BOOST_CHECK_EQUAL(b.value(0), 1.0);
    BOOST_CHECK(b == s);
    b.increase(0);
    BOOST_CHECK(!(b == s));

    auto c = aref;
    c += s;
    BOOST_CHECK_EQUAL(c.value(0), 1.0);
    BOOST_CHECK(c == s);
    BOOST_CHECK(s == c);

    container_storage<std::vector<unsigned>> t(2);
    t.increase(0);
    BOOST_CHECK(!(c == t));
}

template <>
void convert_container_storage_impl<void>() {
    adaptive_storage<> aref(1);
    BOOST_CHECK_EQUAL(aref.value(0), 0.0);
    container_storage<std::vector<unsigned>> s(1);
    s.increase(0);

    auto a = aref;
    a = s;
    BOOST_CHECK_EQUAL(a.value(0), 1.0);
    BOOST_CHECK(a == s);
    a.increase(0);
    BOOST_CHECK(!(a == s));

    auto c = aref;
    c += s;
    BOOST_CHECK_EQUAL(c.value(0), 1.0);
    BOOST_CHECK(c == s);
}

BOOST_AUTO_TEST_CASE(convert_container_storage)
{
    convert_container_storage_impl<detail::weight>();
    convert_container_storage_impl<void>();
    convert_container_storage_impl<uint8_t>();
    convert_container_storage_impl<uint16_t>();
    convert_container_storage_impl<uint32_t>();
    convert_container_storage_impl<uint64_t>();
    convert_container_storage_impl<detail::mp_int>();
}

template <typename T>
void serialization_impl()
{
    const auto a = storage_access::set_value(1, T(1));
    std::ostringstream os;
    std::string buf;
    {
        std::ostringstream os;
        boost::archive::text_oarchive oa(os);
        oa << a;
        buf = os.str();
    }
    adaptive_storage<> b;
    BOOST_CHECK(!(a == b));
    {
        std::istringstream is(buf);
        boost::archive::text_iarchive ia(is);
        ia >> b;
    }
    BOOST_CHECK(a == b);
}

template <>
void serialization_impl<void>()
{
    adaptive_storage<> a(1);
    std::ostringstream os;
    std::string buf;
    {
        std::ostringstream os;
        boost::archive::text_oarchive oa(os);
        oa << a;
        buf = os.str();
    }
    adaptive_storage<> b;
    BOOST_CHECK(!(a == b));
    {
        std::istringstream is(buf);
        boost::archive::text_iarchive ia(is);
        ia >> b;
    }
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(serialization_test)
{
    serialization_impl<detail::weight>();
    serialization_impl<void>();
    serialization_impl<uint8_t>();
    serialization_impl<uint16_t>();
    serialization_impl<uint32_t>();
    serialization_impl<uint64_t>();
    serialization_impl<detail::mp_int>();
}
