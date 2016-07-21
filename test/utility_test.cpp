// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE utility_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <string>
using namespace boost::histogram::detail;

BOOST_AUTO_TEST_CASE(escape_0)
{
    std::string s = escape("abc");
    BOOST_CHECK_EQUAL(s, std::string("'abc'"));
}

BOOST_AUTO_TEST_CASE(escape_1)
{
    std::string s = escape("abc\n");
    BOOST_CHECK_EQUAL(s, std::string("'abc\n'"));
}

BOOST_AUTO_TEST_CASE(escape_2)
{
    std::string s = escape("'abc'");
    BOOST_CHECK_EQUAL(s, std::string("'\\\'abc\\\''"));
}

BOOST_AUTO_TEST_CASE(buffer_ctor_and_get)
{
    auto a = buffer_t(3);
    BOOST_CHECK_EQUAL(a.nbytes(), 3);
    a.get<char>(0) = 0;
    a.get<char>(1) = 1;
    a.get<char>(2) = 0;
    BOOST_CHECK_EQUAL(a.get<char>(0), 0);
    BOOST_CHECK_EQUAL(a.get<char>(1), 1);
    BOOST_CHECK_EQUAL(a.get<char>(2), 0);
    BOOST_CHECK(a == a);
    auto b = buffer_t(3);
    BOOST_CHECK(!(a == b));
    auto c = buffer_t(1);
    BOOST_CHECK(!(a == c));
    auto d = buffer_t();
    BOOST_CHECK(!(a == d));    
}

BOOST_AUTO_TEST_CASE(buffer_copy_ctor)
{
    auto a = buffer_t(3);
    a.get<char>(1) = 1;
    auto b = a;
    BOOST_CHECK_EQUAL(b.get<char>(0), 0);
    BOOST_CHECK_EQUAL(b.get<char>(1), 1);
    BOOST_CHECK_EQUAL(b.get<char>(2), 0);
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(buffer_move_ctor)
{
    auto a = buffer_t(3);
    a.get<char>(1) = 1;
    auto b = std::move(a);
    BOOST_CHECK_EQUAL(a.nbytes(), 0);
    BOOST_CHECK_EQUAL(b.get<char>(0), 0);
    BOOST_CHECK_EQUAL(b.get<char>(1), 1);
    BOOST_CHECK_EQUAL(b.get<char>(2), 0);
    BOOST_CHECK(!(a == b));        
}


BOOST_AUTO_TEST_CASE(buffer_copy_assign)
{
    auto a = buffer_t(3);
    a.get<char>(1) = 1;
    auto b = buffer_t(3);
    b = a;
    BOOST_CHECK_EQUAL(b.get<char>(0), 0);
    BOOST_CHECK_EQUAL(b.get<char>(1), 1);
    BOOST_CHECK_EQUAL(b.get<char>(2), 0);
    BOOST_CHECK(a == b);
}

BOOST_AUTO_TEST_CASE(buffer_move_assign)
{
    auto a = buffer_t(3);
    a.get<char>(1) = 1;
    auto b = buffer_t(3);
    b = std::move(a);
    BOOST_CHECK_EQUAL(a.nbytes(), 0);
    BOOST_CHECK_EQUAL(b.get<char>(0), 0);
    BOOST_CHECK_EQUAL(b.get<char>(1), 1);
    BOOST_CHECK_EQUAL(b.get<char>(2), 0);
    BOOST_CHECK(!(a == b));
}

BOOST_AUTO_TEST_CASE(buffer_resize)
{
    auto a = buffer_t(3);
    BOOST_CHECK_EQUAL(a.nbytes(), 3);
    a.get<char>(0) = 1;
    a.get<char>(1) = 2;
    a.get<char>(2) = 3;
    a.resize(5);
    BOOST_CHECK_EQUAL(a.nbytes(), 5);
    BOOST_CHECK_EQUAL(a.get<char>(0), 1);
    BOOST_CHECK_EQUAL(a.get<char>(1), 2);
    BOOST_CHECK_EQUAL(a.get<char>(2), 3);
    a.resize(1);
    BOOST_CHECK_EQUAL(a.get<char>(0), 1);
    BOOST_CHECK_EQUAL(a.nbytes(), 1);
}
