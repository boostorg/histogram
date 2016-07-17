// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE zero_suppression_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/histogram/detail/zero_suppression.hpp>
#include <boost/assign/std/vector.hpp>
#include <iostream>
using namespace boost::assign;
using namespace boost::histogram::detail;
namespace tt = boost::test_tools;

template <typename T>
void
print(const std::vector<T>& v) {
    std::cerr << "[";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cerr << v[i] << (i < v.size() - 1 ? ", " : "]");
    }
    std::cerr << "\n";
}

#define EQUAL_VECTORS(a, b) \
    BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end())

BOOST_AUTO_TEST_CASE( codec_no_zero )
{
    std::vector<unsigned> a, b, c(3, 0);
    a += 1, 1, 1;
    BOOST_CHECK(zero_suppression_encode<unsigned>(b, &a[0], a.size()));
    EQUAL_VECTORS(a, b);
    zero_suppression_decode<unsigned>(&c[0], c.size(), b);
    EQUAL_VECTORS(a, c);
}

BOOST_AUTO_TEST_CASE( codec_empty )
{
    std::vector<unsigned> a, b, c;
    BOOST_CHECK(zero_suppression_encode<unsigned>(b, &a[0], a.size()));
    EQUAL_VECTORS(a, b);
    zero_suppression_decode<unsigned>(&c[0], c.size(), b);
    EQUAL_VECTORS(a, c);
}

BOOST_AUTO_TEST_CASE( codec_zero_0a )
{
    std::vector<uint8_t> a, b;
    a += 0;
    BOOST_CHECK(zero_suppression_encode<uint8_t>(b, &a[0], a.size()) == false);
}

BOOST_AUTO_TEST_CASE( codec_zero_1a )
{
    std::vector<uint8_t> a, b, c, d(3, 0);
    a += 1, 0, 0;
    c += 1, 0, 2;
    BOOST_CHECK(zero_suppression_encode<uint8_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint8_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_2a )
{
    std::vector<uint8_t> a, b, c, d(3, 0);
    a += 0, 0, 1;
    c += 0, 2, 1;
    BOOST_CHECK(zero_suppression_encode<uint8_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint8_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_3a )
{
    std::vector<uint8_t> a, b, c, d(2, 0);
    a += 0, 0;
    c += 0, 2;
    BOOST_CHECK(zero_suppression_encode<uint8_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint8_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_0b )
{
    std::vector<unsigned> a, b;
    a += 0;
    BOOST_CHECK(zero_suppression_encode<unsigned>(b, &a[0], a.size()) == false);
}

BOOST_AUTO_TEST_CASE( codec_zero_1b )
{
    std::vector<unsigned> a, b, c, d(3, 0);
    a += 1, 0, 0;
    c += 1, 0, 2;
    BOOST_CHECK(zero_suppression_encode<unsigned>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<unsigned>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_2b )
{
    std::vector<unsigned> a, b, c, d(3, 0);
    a += 0, 0, 1;
    c += 0, 2, 1;
    BOOST_CHECK(zero_suppression_encode<unsigned>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<unsigned>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_3b )
{
    std::vector<unsigned> a, b, c, d(2, 0);
    a += 0, 0;
    c += 0, 2;
    BOOST_CHECK(zero_suppression_encode<unsigned>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<unsigned>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_0c )
{
    std::vector<uint64_t> a, b;
    a += 0;
    BOOST_CHECK(zero_suppression_encode<uint64_t>(b, &a[0], a.size()) == false);
}

BOOST_AUTO_TEST_CASE( codec_zero_1c )
{
    std::vector<uint64_t> a, b, c, d(3, 0);
    a += 1, 0, 0;
    c += 1, 0, 2;
    BOOST_CHECK(zero_suppression_encode<uint64_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint64_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_2c )
{
    std::vector<uint64_t> a, b, c, d(3, 0);
    a += 0, 0, 1;
    c += 0, 2, 1;
    BOOST_CHECK(zero_suppression_encode<uint64_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint64_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_3c )
{
    std::vector<uint64_t> a, b, c, d(2, 0);
    a += 0, 0;
    c += 0, 2;
    BOOST_CHECK(zero_suppression_encode<uint64_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint64_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_0d )
{
    std::vector<wtype> a, b;
    a += 0;
    BOOST_CHECK(zero_suppression_encode<wtype>(b, &a[0], a.size()) == false);
}

BOOST_AUTO_TEST_CASE( codec_zero_1d )
{
    std::vector<wtype> a, b, c, d(3, 0);
    a += 1, 0, 0;
    c += 1, 0, 2;
    BOOST_CHECK(zero_suppression_encode<wtype>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<wtype>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_2d )
{
    std::vector<wtype> a, b, c, d(3, 0);
    a += 0, 0, 1;
    c += 0, 2, 1;
    BOOST_CHECK(zero_suppression_encode<wtype>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<wtype>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_3d )
{
    std::vector<wtype> a, b, c, d(2, 0);
    a += 0, 0;
    c += 0, 2;
    BOOST_CHECK(zero_suppression_encode<wtype>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<wtype>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_4 )
{
    std::vector<unsigned> a, b, c, d(5, 0);
    a += 0, 0, 0, 1, 0;
    c += 0, 3, 1, 0, 1;
    BOOST_CHECK(zero_suppression_encode<unsigned>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<unsigned>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_5 )
{
    std::vector<unsigned> a, b, c, d(5, 0);
    a += 0, 1, 0, 0, 0;
    c += 0, 1, 1, 0, 3;
    BOOST_CHECK(zero_suppression_encode<unsigned>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<unsigned>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_6 )
{
    std::vector<uint8_t> a(255, 0), b, c, d(255, 0);
    c += 0, 255;
    BOOST_CHECK(zero_suppression_encode<uint8_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint8_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_7 )
{
    std::vector<uint8_t> a(256, 0), b, c, d(256, 0);
    c += 0, 0;
    BOOST_CHECK(zero_suppression_encode<uint8_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint8_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_8 )
{
    std::vector<uint8_t> a(257, 0), b, c, d(257, 0);
    c += 0, 0, 0, 1;
    BOOST_CHECK(zero_suppression_encode<uint8_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint8_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_9 )
{
    std::vector<uint8_t> a(1000, 0), b, c, d(1000, 0);
    c += 0, 0, 0, 0, 0, 0, 0, 232;
    BOOST_CHECK(zero_suppression_encode<uint8_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint8_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_10 )
{
    std::vector<uint16_t> a(1000, 0), b, c, d(1000, 0);
    c += 0, 1000;
    BOOST_CHECK(zero_suppression_encode<uint16_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint16_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_11 )
{
    std::vector<uint16_t> a(65536, 0), b, c, d(65536, 0);
    c += 0, 0;
    BOOST_CHECK(zero_suppression_encode<uint16_t>(b, &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode<uint16_t>(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}
