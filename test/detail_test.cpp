// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE detail_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/weight.hpp>
#include <boost/histogram/detail/tiny_string.hpp>
#include <sstream>
#include <algorithm>
using namespace boost::histogram::detail;

// bool operator==(const buffer& a, const buffer& b)
// {
//     if (!(a.size() == b.size() &&
//           a.id() == b.id() &&
//           a.depth() == b.depth()))
//         return false;
//     switch (a.id()) {
//         case -1: return std::equal(&a.at<weight>(0),
//                                    &a.at<weight>(a.size()),
//                                    &b.at<weight>(0));
//         case 0: return true;
//         case 1: return std::equal(&a.at<uint8_t>(0),
//                                   &a.at<uint8_t>(a.size()),
//                                   &b.at<uint8_t>(0));
//         case 2: return std::equal(&a.at<uint16_t>(0),
//                                   &a.at<uint16_t>(a.size()),
//                                   &b.at<uint16_t>(0));
//         case 3: return std::equal(&a.at<uint32_t>(0),
//                                   &a.at<uint32_t>(a.size()),
//                                   &b.at<uint32_t>(0));
//         case 4: return std::equal(&a.at<uint64_t>(0),
//                                   &a.at<uint64_t>(a.size()),
//                                   &b.at<uint64_t>(0));
//     }
//     BOOST_ASSERT(!"never reach this");
//     return false;
// }

BOOST_AUTO_TEST_CASE(escape_0)
{
    std::ostringstream os;
    escape(os, "abc");
    BOOST_CHECK_EQUAL(os.str(), std::string("'abc'"));
}

BOOST_AUTO_TEST_CASE(escape_1)
{
    std::ostringstream os;
    escape(os, "abc\n");
    BOOST_CHECK_EQUAL(os.str(), std::string("'abc\n'"));
}

BOOST_AUTO_TEST_CASE(escape_2)
{
    std::ostringstream os;
    escape(os, "'abc'");
    BOOST_CHECK_EQUAL(os.str(), std::string("'\\\'abc\\\''"));
}

// BOOST_AUTO_TEST_CASE(buffer_ctor_and_get)
// {
//     auto a = buffer(3);
//     a.initialize<uint8_t>();
//     BOOST_CHECK_EQUAL(a.size(), 3);
//     BOOST_CHECK_EQUAL(a.depth(), 1);
//     a.at<uint8_t>(0) = 0;
//     a.at<uint8_t>(1) = 1;
//     a.at<uint8_t>(2) = 0;
//     BOOST_CHECK_EQUAL(a.at<uint8_t>(0), 0);
//     BOOST_CHECK_EQUAL(a.at<uint8_t>(1), 1);
//     BOOST_CHECK_EQUAL(a.at<uint8_t>(2), 0);
//     BOOST_CHECK(a == a);
//     auto b = buffer(3);
//     b.initialize<uint8_t>();
//     BOOST_CHECK(!(a == b));
//     auto c = buffer(1);
//     c.initialize<uint8_t>();
//     BOOST_CHECK(!(a == c));
//     auto d = buffer();
//     BOOST_CHECK(!(a == d));
// }

// BOOST_AUTO_TEST_CASE(buffer_copy_ctor)
// {
//     auto a = buffer(3);
//     a.initialize<uint8_t>();
//     a.at<uint8_t>(1) = 1;
//     auto b = a;
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(0), 0);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(1), 1);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(2), 0);
//     BOOST_CHECK(a == b);
// }

// BOOST_AUTO_TEST_CASE(buffer_move_ctor)
// {
//     auto a = buffer(3);
//     a.initialize<uint8_t>();
//     a.at<uint8_t>(1) = 1;
//     auto b = std::move(a);
//     BOOST_CHECK_EQUAL(a.size(), 0);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(0), 0);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(1), 1);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(2), 0);
//     BOOST_CHECK(!(a == b));
// }


// BOOST_AUTO_TEST_CASE(buffer_copy_assign)
// {
//     auto a = buffer(3);
//     a.initialize<uint8_t>();
//     a.at<uint8_t>(1) = 1;
//     auto b = buffer(3);
//     b.initialize<uint8_t>();
//     b = a;
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(0), 0);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(1), 1);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(2), 0);
//     BOOST_CHECK(a == b);
// }

// BOOST_AUTO_TEST_CASE(buffer_move_assign)
// {
//     auto a = buffer(3);
//     a.initialize<uint8_t>();
//     a.at<uint8_t>(1) = 1;
//     auto b = buffer(3);
//     b.initialize<uint8_t>();
//     b = std::move(a);
//     BOOST_CHECK_EQUAL(a.size(), 0);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(0), 0);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(1), 1);
//     BOOST_CHECK_EQUAL(b.at<uint8_t>(2), 0);
//     BOOST_CHECK(!(a == b));
// }

// BOOST_AUTO_TEST_CASE(buffer_realloc)
// {
//     auto a = buffer(3);
//     a.initialize<uint8_t>();
//     BOOST_CHECK_EQUAL(a.size(), 3);
//     a.at<uint8_t>(0) = 1;
//     a.at<uint8_t>(1) = 2;
//     a.at<uint8_t>(2) = 3;
//     a.grow<uint8_t>();
//     BOOST_CHECK_EQUAL(a.size(), 3);
//     BOOST_CHECK_EQUAL(a.depth(), 2);
//     BOOST_CHECK_EQUAL(a.at<int16_t>(0), 1);
//     BOOST_CHECK_EQUAL(a.at<int16_t>(1), 2);
//     BOOST_CHECK_EQUAL(a.at<int16_t>(2), 3);
// }

BOOST_AUTO_TEST_CASE(tiny_string_test)
{
    auto a = tiny_string();
    BOOST_CHECK_EQUAL(a.size(), 0u);
    BOOST_CHECK_EQUAL(a.c_str(), "");
    auto b = tiny_string("abc");
    BOOST_CHECK_EQUAL(b.c_str(), "abc");
    BOOST_CHECK_EQUAL(b.size(), 3u);
    std::ostringstream os;
    os << b;
    BOOST_CHECK_EQUAL(os.str(), std::string("abc"));
    auto c = b;
    BOOST_CHECK_EQUAL(c, b);
    auto d = std::move(c);
    BOOST_CHECK_EQUAL(c.c_str(), "");
    BOOST_CHECK_EQUAL(d, b);
    c = d;
    BOOST_CHECK_EQUAL(c, d);
    d = std::move(c);
    BOOST_CHECK_EQUAL(c.c_str(), "");
}
