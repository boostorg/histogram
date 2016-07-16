// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE histogram_test
#include <boost/histogram/histogram.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <limits>
using namespace boost::histogram;

BOOST_AUTO_TEST_CASE(init_0)
{
    auto h = histogram();
    BOOST_CHECK_EQUAL(h.dim(), 0);
    BOOST_CHECK_EQUAL(h.size(), 0);
}

BOOST_AUTO_TEST_CASE(init_1)
{
    auto h = histogram(regular_axis{3, -1, 1});
    BOOST_CHECK_EQUAL(h.dim(), 1);
    BOOST_CHECK_EQUAL(h.size(), 5);
    BOOST_CHECK_EQUAL(h.shape(0), 5);
}

BOOST_AUTO_TEST_CASE(init_2)
{
    auto h = histogram(regular_axis{3, -1, 1},
                       integer_axis{-1, 1});
    BOOST_CHECK_EQUAL(h.dim(), 2);
    BOOST_CHECK_EQUAL(h.size(), 25);
    BOOST_CHECK_EQUAL(h.shape(0), 5);
    BOOST_CHECK_EQUAL(h.shape(1), 5);
}

BOOST_AUTO_TEST_CASE(init_3)
{
    auto h = histogram(regular_axis{3, -1, 1},
                       integer_axis{-1, 1},
                       polar_axis{3});
    BOOST_CHECK_EQUAL(h.dim(), 3);
    BOOST_CHECK_EQUAL(h.size(), 75);
}

BOOST_AUTO_TEST_CASE(init_4)
{
    auto h = histogram(regular_axis{3, -1, 1},
                       integer_axis{-1, 1},
                       polar_axis{3},
                       variable_axis{-1, 0, 1});
    BOOST_CHECK_EQUAL(h.dim(), 4);
    BOOST_CHECK_EQUAL(h.size(), 300);
}

BOOST_AUTO_TEST_CASE(init_5)
{
    auto h = histogram(regular_axis{3, -1, 1},
                       integer_axis{-1, 1},
                       polar_axis{3},
                       variable_axis{-1, 0, 1},
                       category_axis{"A", "B", "C"});
    BOOST_CHECK_EQUAL(h.dim(), 5);
    BOOST_CHECK_EQUAL(h.size(), 900);
}

BOOST_AUTO_TEST_CASE(copy_ctor)
{
    auto h = histogram(regular_axis(1, -1, 1),
                       regular_axis(2, -2, 2));
    h.fill(0.0, 0.0);
    auto h2 = histogram_t<2>(h);
    BOOST_CHECK(h2 == h);
    // auto h3 = histogram_t<Dynamic>(h);
    // BOOST_CHECK(h3 == h);
}

BOOST_AUTO_TEST_CASE(copy_assign)
{
    auto h = histogram(regular_axis(1, -1, 1),
                       regular_axis(2, -2, 2));
    h.fill(0.0, 0.0);
    auto h2 = histogram_t<2>();
    BOOST_CHECK(!(h == h2));
    h2 = h;
    BOOST_CHECK(h == h2);
    // test self-assign
    h2 = h2;
    BOOST_CHECK(h == h2);
    // auto h3 = histogram_t<Dynamic>();
    // h3 = h;
    // BOOST_CHECK(h == h3);
}

BOOST_AUTO_TEST_CASE(move_ctor)
{
    auto h = histogram(regular_axis(1, -1, 1),
                       regular_axis(2, -2, 2));
    h.fill(0.0, 0.0);
    auto h2 = histogram_t<2>(std::move(h));
    BOOST_CHECK_EQUAL(h2.sum(), 1);
    BOOST_CHECK_EQUAL(h2.dim(), 2);
    BOOST_CHECK_EQUAL(h.sum(), 0);
    BOOST_CHECK_EQUAL(h.dim(), 2);
    // auto h3 = histogram_t<Dynamic>(std::move(h2));
    // BOOST_CHECK_EQUAL(h3.sum(), 1);
    // BOOST_CHECK_EQUAL(h3.dim(), 2);
    // BOOST_CHECK_EQUAL(h2.sum(), 0);
    // BOOST_CHECK_EQUAL(h2.dim(), 0);
}

BOOST_AUTO_TEST_CASE(self_move_assign)
{
    auto h = histogram(regular_axis(1, -1, 1),
                       regular_axis(2, -2, 2));
    h = std::move(h);  
}

BOOST_AUTO_TEST_CASE(move_assign)
{
    auto h = histogram(regular_axis(1, -1, 1),
                       regular_axis(2, -2, 2));
    h.fill(0.0, 0.0);
    auto h2 = histogram_t<2>();
    h2 = std::move(h);
    BOOST_CHECK_EQUAL(h2.sum(), 1);
    BOOST_CHECK_EQUAL(h2.dim(), 2);
    BOOST_CHECK_EQUAL(h.sum(), 0);
    BOOST_CHECK_EQUAL(h.dim(), 2);
    // test self-move
    h2 = std::move(h2);
    BOOST_CHECK_EQUAL(h2.sum(), 1);
    BOOST_CHECK_EQUAL(h2.dim(), 2);
    // auto h3 = histogram_t<Dynamic>();
    // h3 = std::move(h2);
    // BOOST_CHECK_EQUAL(h3.sum(), 1);
    // BOOST_CHECK_EQUAL(h3.dim(), 2);
    // BOOST_CHECK_EQUAL(h2.sum(), 0);
    // BOOST_CHECK_EQUAL(h2.dim(), 0);
}

BOOST_AUTO_TEST_CASE(d1)
{
    auto h = histogram(regular_axis(2, -1, 1));
    h.fill(-1);
    h.fill(-1.0);
    h.fill(-2.0);
    h.fill(10.0);

    BOOST_CHECK_EQUAL(h.dim(), 1);
    BOOST_CHECK_EQUAL(h.bins(0), 2);
    BOOST_CHECK_EQUAL(h.shape(0), 4);
    BOOST_CHECK_EQUAL(h.sum(), 4);

    BOOST_CHECK_EQUAL(h.value(-1), 1.0);
    BOOST_CHECK_EQUAL(h.value(0), 2.0);
    BOOST_CHECK_EQUAL(h.value(1), 0.0);
    BOOST_CHECK_EQUAL(h.value(2), 1.0);

    BOOST_CHECK_EQUAL(h.variance(-1), 1.0);
    BOOST_CHECK_EQUAL(h.variance(0), 2.0);
    BOOST_CHECK_EQUAL(h.variance(1), 0.0);
    BOOST_CHECK_EQUAL(h.variance(2), 1.0);
}

// BOOST_AUTO_TEST_CASE(d1w)
// {
//     histogram h(regular_axis(2, -1, 1));
//     h.fill(0);
//     h.wfill(-1.0, 2.0);
//     h.fill(-1.0);
//     h.fill(-2.0);
//     h.wfill(10.0, 5.0);

//     BOOST_CHECK_EQUAL(h.sum(), 10);

//     BOOST_CHECK_EQUAL(h.value(-1), 1.0);
//     BOOST_CHECK_EQUAL(h.value(0), 3.0);
//     BOOST_CHECK_EQUAL(h.value(1), 1.0);
//     BOOST_CHECK_EQUAL(h.value(2), 5.0);

//     BOOST_CHECK_EQUAL(h.variance(-1), 1.0);
//     BOOST_CHECK_EQUAL(h.variance(0), 5.0);
//     BOOST_CHECK_EQUAL(h.variance(1), 1.0);
//     BOOST_CHECK_EQUAL(h.variance(2), 25.0);
// }

BOOST_AUTO_TEST_CASE(d2)
{
    auto h = histogram(regular_axis(2, -1, 1),
                       integer_axis(-1, 1, std::string(), false));
    h.fill(-1, -1);
    h.fill(-1, 0);
    h.fill(-1, -10);
    h.fill(-10, 0);

    // BOOST_CHECK_EQUAL(h.dim(), 2);
    // BOOST_CHECK_EQUAL(h.bins(0), 2);
    // BOOST_CHECK_EQUAL(h.shape(0), 4);
    // BOOST_CHECK_EQUAL(h.bins(1), 3);
    // BOOST_CHECK_EQUAL(h.shape(1), 3);
    // BOOST_CHECK_EQUAL(h.sum(), 3);

    // BOOST_CHECK_EQUAL(h.value(-1, 0), 0.0);
    // BOOST_CHECK_EQUAL(h.value(-1, 1), 1.0);
    // BOOST_CHECK_EQUAL(h.value(-1, 2), 0.0);

    // BOOST_CHECK_EQUAL(h.value(0, 0), 1.0);
    // BOOST_CHECK_EQUAL(h.value(0, 1), 1.0);
    // BOOST_CHECK_EQUAL(h.value(0, 2), 0.0);

    // BOOST_CHECK_EQUAL(h.value(1, 0), 0.0);
    // BOOST_CHECK_EQUAL(h.value(1, 1), 0.0);
    // BOOST_CHECK_EQUAL(h.value(1, 2), 0.0);

    // BOOST_CHECK_EQUAL(h.value(2, 0), 0.0);
    // BOOST_CHECK_EQUAL(h.value(2, 1), 0.0);
    // BOOST_CHECK_EQUAL(h.value(2, 2), 0.0);

    // BOOST_CHECK_EQUAL(h.variance(-1, 0), 0.0);
    // BOOST_CHECK_EQUAL(h.variance(-1, 1), 1.0);
    // BOOST_CHECK_EQUAL(h.variance(-1, 2), 0.0);

    // BOOST_CHECK_EQUAL(h.variance(0, 0), 1.0);
    // BOOST_CHECK_EQUAL(h.variance(0, 1), 1.0);
    // BOOST_CHECK_EQUAL(h.variance(0, 2), 0.0);

    // BOOST_CHECK_EQUAL(h.variance(1, 0), 0.0);
    // BOOST_CHECK_EQUAL(h.variance(1, 1), 0.0);
    // BOOST_CHECK_EQUAL(h.variance(1, 2), 0.0);

    // BOOST_CHECK_EQUAL(h.variance(2, 0), 0.0);
    // BOOST_CHECK_EQUAL(h.variance(2, 1), 0.0);
    // BOOST_CHECK_EQUAL(h.variance(2, 2), 0.0);
}

// BOOST_AUTO_TEST_CASE(d2w)
// {
//     histogram h(regular_axis(2, -1, 1),
//                 integer_axis(-1, 1, std::string(), false));
//     h.fill(-1, 0);       // -> 0, 1
//     h.wfill(-1, -1, 10); // -> 0, 0
//     h.wfill(-1, -10, 5); // is ignored
//     h.wfill(-10, 0, 7);  // -> -1, 1

//     BOOST_CHECK_EQUAL(h.sum(), 18);

//     BOOST_CHECK_EQUAL(h.value(-1, 0), 0.0);
//     BOOST_CHECK_EQUAL(h.value(-1, 1), 7.0);
//     BOOST_CHECK_EQUAL(h.value(-1, 2), 0.0);

//     BOOST_CHECK_EQUAL(h.value(0, 0), 10.0);
//     BOOST_CHECK_EQUAL(h.value(0, 1), 1.0);
//     BOOST_CHECK_EQUAL(h.value(0, 2), 0.0);

//     BOOST_CHECK_EQUAL(h.value(1, 0), 0.0);
//     BOOST_CHECK_EQUAL(h.value(1, 1), 0.0);
//     BOOST_CHECK_EQUAL(h.value(1, 2), 0.0);

//     BOOST_CHECK_EQUAL(h.value(2, 0), 0.0);
//     BOOST_CHECK_EQUAL(h.value(2, 1), 0.0);
//     BOOST_CHECK_EQUAL(h.value(2, 2), 0.0);

//     BOOST_CHECK_EQUAL(h.variance(-1, 0), 0.0);
//     BOOST_CHECK_EQUAL(h.variance(-1, 1), 49.0);
//     BOOST_CHECK_EQUAL(h.variance(-1, 2), 0.0);

//     BOOST_CHECK_EQUAL(h.variance(0, 0), 100.0);
//     BOOST_CHECK_EQUAL(h.variance(0, 1), 1.0);
//     BOOST_CHECK_EQUAL(h.variance(0, 2), 0.0);

//     BOOST_CHECK_EQUAL(h.variance(1, 0), 0.0);
//     BOOST_CHECK_EQUAL(h.variance(1, 1), 0.0);
//     BOOST_CHECK_EQUAL(h.variance(1, 2), 0.0);

//     BOOST_CHECK_EQUAL(h.variance(2, 0), 0.0);
//     BOOST_CHECK_EQUAL(h.variance(2, 1), 0.0);
//     BOOST_CHECK_EQUAL(h.variance(2, 2), 0.0);
// }

// BOOST_AUTO_TEST_CASE(add_0)
// {
//     histogram a(integer_axis(-1, 1));
//     histogram b(regular_axis(3, -1, 1));
//     BOOST_CHECK_THROW(a + b, std::logic_error);
// }

// BOOST_AUTO_TEST_CASE(add_1)
// {
//     histogram a(integer_axis(-1, 1));
//     histogram b(integer_axis(-1, 1));
//     a.fill(0);
//     b.fill(-1);
//     histogram c = a + b;
//     BOOST_CHECK_EQUAL(c.value(-1), 0);
//     BOOST_CHECK_EQUAL(c.value(0), 1);
//     BOOST_CHECK_EQUAL(c.value(1), 1);
//     BOOST_CHECK_EQUAL(c.value(2), 0);
//     BOOST_CHECK_EQUAL(c.value(3), 0);
// }

// BOOST_AUTO_TEST_CASE(add_2)
// {
//     histogram a(integer_axis(-1, 1));
//     histogram b(integer_axis(-1, 1));

//     a.fill(0);
//     b.wfill(-1, 3);
//     histogram c = a + b;
//     BOOST_CHECK_EQUAL(c.value(-1), 0);
//     BOOST_CHECK_EQUAL(c.value(0), 3);
//     BOOST_CHECK_EQUAL(c.value(1), 1);
//     BOOST_CHECK_EQUAL(c.value(2), 0);
//     BOOST_CHECK_EQUAL(c.value(3), 0);    
// }
