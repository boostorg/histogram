#include <boost/histogram/detail/zero_suppression.hpp>
#define BOOST_TEST_MODULE zero_suppression_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/assign/std/vector.hpp>
#include <iostream>
using namespace boost::assign;
using boost::histogram::detail::zero_suppression_encode;
using boost::histogram::detail::zero_suppression_decode;
namespace tt = boost::test_tools;

void
print(const std::vector<char>& v) {
    std::cerr << "[";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cerr << unsigned(v[i]) << (i < v.size() - 1 ? ", " : "]");
    }
    std::cerr << "\n";
}

#define EQUAL_VECTORS(a, b) \
    BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end())

BOOST_AUTO_TEST_CASE( codec_no_zero )
{
    std::vector<char> a, b, c(3, 0);
    a += 1, 1, 1;
    BOOST_CHECK(zero_suppression_encode(b, 3, &a[0], a.size()));
    EQUAL_VECTORS(a, b);
    zero_suppression_decode(&c[0], c.size(), b);
    EQUAL_VECTORS(a, c);
}

BOOST_AUTO_TEST_CASE( codec_fail )
{
    std::vector<char> a, b;
    a += 1, 1, 1;
    BOOST_CHECK(zero_suppression_encode(b, 2, &a[0], a.size()) == false);
}

BOOST_AUTO_TEST_CASE( codec_empty )
{
    std::vector<char> a, b, c;
    BOOST_CHECK(zero_suppression_encode(b, 3, &a[0], a.size()));
    EQUAL_VECTORS(a, b);
    zero_suppression_decode(&c[0], c.size(), b);
    EQUAL_VECTORS(a, c);
}

BOOST_AUTO_TEST_CASE( codec_zero_0 )
{
    std::vector<char> a, b, c, d(2, 0);
    a += 0, 1;
    c += 0, 1, 1;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_1 )
{
    std::vector<char> a, b, c, d(3, 0);
    a += 0, 0, 1;
    c += 0, 2, 1;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_2 )
{
    std::vector<char> a, b, c, d(2, 0);
    a += 0, 0;
    c += 0, 2;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_3 )
{
    std::vector<char> a, b, c, d(1, 0);
    a += 0;
    c += 0, 1;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_4 )
{
    std::vector<char> a, b, c, d(3, 0);
    a += 0, 1, 0;
    c += 0, 1, 1, 0, 1;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_5 )
{
    std::vector<char> a, b, c, d(4, 0);
    a += 0, 1, 0, 0;
    c += 0, 1, 1, 0, 2;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_6 )
{
    std::vector<char> a(255, 0), b, c, d(255, 0);
    c += 0, 255;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_7 )
{
    std::vector<char> a(256, 0), b, c, d(256, 0);
    c += 0, 0;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_8 )
{
    std::vector<char> a(257, 0), b, c, d(257, 0);
    c += 0, 0, 0, 1;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}

BOOST_AUTO_TEST_CASE( codec_zero_9 )
{
    std::vector<char> a(1000, 0), b, c, d(1000, 0);
    c += 0, 0, 0, 0, 0, 0, 0, 232;
    BOOST_CHECK(zero_suppression_encode(b, c.size(), &a[0], a.size()));
    EQUAL_VECTORS(b, c);
    zero_suppression_decode(&d[0], d.size(), b);
    EQUAL_VECTORS(a, d);
}
