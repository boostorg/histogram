// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/weight.hpp>
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
        escape(os, std::string("abc"));
        BOOST_TEST_EQ(os.str(), std::string("'abc'"));
    }

    // escape1
    {
        std::ostringstream os;
        escape(os, std::string("abc\n"));
        BOOST_TEST_EQ(os.str(), std::string("'abc\n'"));
    }

    // escape2
    {
        std::ostringstream os;
        escape(os, std::string("'abc'"));
        BOOST_TEST_EQ(os.str(), std::string("'\\\'abc\\\''"));
    }

    return boost::report_errors();
}
