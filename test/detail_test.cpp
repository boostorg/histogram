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

BOOST_AUTO_TEST_CASE(weight_test)
{
    BOOST_CHECK(weight(0) == weight());
    weight w(1);
    BOOST_CHECK(w == weight(1));
    BOOST_CHECK(w != weight());
    BOOST_CHECK(1 == w);
    BOOST_CHECK(w == 1);
    BOOST_CHECK(2 != w);
    BOOST_CHECK(w != 2);
}

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
