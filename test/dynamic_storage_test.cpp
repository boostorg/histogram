// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE dynamic_storage_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <sstream>
#include <string>
#include <limits>
using namespace boost::histogram;

BOOST_AUTO_TEST_CASE(wtype_streamer)
{
    std::ostringstream os;
    detail::wtype w;
    w.w = 1.2;
    w.w2 = 2.3;
    os << w;
    BOOST_CHECK_EQUAL(os.str(), std::string("(1.2,2.3)"));
}

BOOST_AUTO_TEST_CASE(dynamic_storage_grow_1)
{
    double x = 1.0;
    dynamic_storage n(1);
    n.increase(0);
    for (unsigned i = 0; i < 64; ++i) {
        n += n;
        x += x;
        n.increase(0);
        ++x;
        BOOST_CHECK_EQUAL(n.value(0), x);
        BOOST_CHECK_EQUAL(n.variance(0), x);
    }
}

BOOST_AUTO_TEST_CASE(dynamic_storage_grow_2)
{
    dynamic_storage n(1);
    n.increase(0);
    for (unsigned i = 0; i < 64; ++i) {
        n += n;
        dynamic_storage a(1);
        a.increase(0, 0.0); // converts to wtype
        a = n;
        BOOST_CHECK_EQUAL(a.value(0), n.value(0));
        BOOST_CHECK_EQUAL(a.variance(0), n.variance(0));
    }
}

BOOST_AUTO_TEST_CASE(dynamic_storage_grow_3)
{
    dynamic_storage n(1);
    n.increase(0);
    for (unsigned i = 0; i < 64; ++i) {
        n += n;
        dynamic_storage a(1);
        a += n;
        a.increase(0, 0.0); // converts to wtype
        BOOST_CHECK_EQUAL(a.value(0), n.value(0));
        BOOST_CHECK_EQUAL(a.variance(0), n.variance(0));
    }
}

BOOST_AUTO_TEST_CASE(dynamic_storage_add_with_growth)
{
    dynamic_storage a(1), b(1);
    a.increase(0);
    for (unsigned i = 0; i < 64; ++i)
        a += a;
    b.increase(0);
    b += a;
    BOOST_CHECK_EQUAL(b.value(0), a.value(0) + 1.0);
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
