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

// hotfix for cstring comparison
namespace boost { namespace detail {
inline void test_eq_impl( char const * expr1, char const * expr2,
  char const * file, int line, char const * function, const char* t, const char* u )
{
    if( std::strcmp(t, u) == 0 )
    {
        report_errors_remind();
    }
    else
    {
        BOOST_LIGHTWEIGHT_TEST_OSTREAM
            << file << "(" << line << "): test '" << expr1 << " == " << expr2
            << "' failed in function '" << function << "': "
            << "'" << t << "' != '" << u << "'" << std::endl;
        ++test_errors();
    }
}
}}

int main ()
{
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

    {
        std::ostringstream os;
        escape(os, "abc");
        BOOST_TEST_EQ(os.str(), std::string("'abc'"));
    }

    {
        std::ostringstream os;
        escape(os, "abc\n");
        BOOST_TEST_EQ(os.str(), std::string("'abc\n'"));
    }

    {
        std::ostringstream os;
        escape(os, "'abc'");
        BOOST_TEST_EQ(os.str(), std::string("'\\\'abc\\\''"));
    }

    {
        auto a = tiny_string();
        BOOST_TEST_EQ(a.size(), 0u);
        BOOST_TEST_EQ(a.c_str(), "");
        auto b = tiny_string("abc");
        BOOST_TEST_EQ(b.c_str(), "abc");
        BOOST_TEST_EQ(b.size(), 3u);
        std::ostringstream os;
        os << b;
        BOOST_TEST_EQ(os.str(), std::string("abc"));
        auto c = b;
        BOOST_TEST_EQ(c, b);
        auto d = std::move(c);
        BOOST_TEST_EQ(c.c_str(), "");
        BOOST_TEST_EQ(d, b);
        c = d;
        BOOST_TEST_EQ(c, d);
        d = std::move(c);
        BOOST_TEST_EQ(c.c_str(), "");
    }
    return boost::report_errors();
}
