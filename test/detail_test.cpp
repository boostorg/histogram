// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/weight.hpp>
#include <boost/histogram/detail/tiny_string.hpp>
#include <sstream>
#include <cstring>
using namespace boost::histogram::detail;

#ifndef BOOST_TEST_CSTR_EQ
    #define BOOST_TEST_CSTR_EQ(x,y) BOOST_TEST(std::strcmp(x,y) == 0)
#endif

int main ()
{
    // weight
    {
        BOOST_TEST(weight(0) == weight());
        weight w(1);
        BOOST_TEST(w == weight(1));
        BOOST_TEST(w != weight());
        BOOST_TEST(1 == w);
        BOOST_TEST(w == 1);
        BOOST_TEST(2 != w);
        BOOST_TEST(w != 2);
    }

    // escape0
    {
        std::ostringstream os;
        escape(os, "abc");
        BOOST_TEST_EQ(os.str(), std::string("'abc'"));
    }

    // escape1
    {
        std::ostringstream os;
        escape(os, "abc\n");
        BOOST_TEST_EQ(os.str(), std::string("'abc\n'"));
    }

    // escape2
    {
        std::ostringstream os;
        escape(os, "'abc'");
        BOOST_TEST_EQ(os.str(), std::string("'\\\'abc\\\''"));
    }

    // tiny_string
    {
        auto a = tiny_string();
        BOOST_TEST_EQ(a.size(), 0u);
        BOOST_TEST_CSTR_EQ(a.c_str(), "");
        auto b = tiny_string("abc");
        BOOST_TEST_CSTR_EQ(b.c_str(), "abc");
        BOOST_TEST_EQ(b.size(), 3u);
        std::ostringstream os;
        os << b;
        BOOST_TEST_EQ(os.str(), std::string("abc"));
        auto c = b;
        BOOST_TEST_EQ(c, b);
        auto d = std::move(c);
        BOOST_TEST_CSTR_EQ(c.c_str(), "");
        BOOST_TEST_EQ(d, b);
        c = d;
        BOOST_TEST_EQ(c, d);
        d = std::move(c);
        BOOST_TEST_CSTR_EQ(c.c_str(), "");
    }
    return boost::report_errors();
}
