// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE dynamic_histogram_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/utility.hpp>
#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/container_storage.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <limits>
#include <sstream>
#include <array>
#include <vector>

using namespace boost::histogram;
namespace mpl = boost::mpl;

BOOST_AUTO_TEST_CASE(init_0)
{
    auto h = dynamic_histogram<default_axes, adaptive_storage>();
    BOOST_CHECK_EQUAL(h.dim(), 0);
    BOOST_CHECK_EQUAL(h.size(), 0);
    auto h2 = dynamic_histogram<
        default_axes,
        container_storage<std::vector<unsigned>>
    >();
    BOOST_CHECK(h2 == h);
}

BOOST_AUTO_TEST_CASE(init_1)
{
    auto h = dynamic_histogram<
        default_axes,
        adaptive_storage
    >(regular_axis{3, -1, 1});
    BOOST_CHECK_EQUAL(h.dim(), 1);
    BOOST_CHECK_EQUAL(h.size(), 5);
    BOOST_CHECK_EQUAL(shape(h.axis(0)), 5);
    auto h2 = dynamic_histogram<
        default_axes,
        container_storage<std::vector<unsigned>>
    >(regular_axis{3, -1, 1});
    BOOST_CHECK(h2 == h);
}

BOOST_AUTO_TEST_CASE(init_2)
{
    auto h = dynamic_histogram<
        default_axes,
        adaptive_storage
    >(regular_axis{3, -1, 1}, integer_axis{-1, 1});
    BOOST_CHECK_EQUAL(h.dim(), 2);
    BOOST_CHECK_EQUAL(h.size(), 25);
    BOOST_CHECK_EQUAL(shape(h.axis(0)), 5);
    BOOST_CHECK_EQUAL(shape(h.axis(1)), 5);
    auto h2 = dynamic_histogram<
        default_axes,
        container_storage<std::vector<unsigned>>
    >(regular_axis{3, -1, 1}, integer_axis{-1, 1});
    BOOST_CHECK(h2 == h);
}

BOOST_AUTO_TEST_CASE(init_3)
{
    auto h = dynamic_histogram<
        default_axes,
        adaptive_storage
    >(regular_axis{3, -1, 1}, integer_axis{-1, 1}, polar_axis{3});
    BOOST_CHECK_EQUAL(h.dim(), 3);
    BOOST_CHECK_EQUAL(h.size(), 75);
    auto h2 = dynamic_histogram<
        default_axes,
        container_storage<std::vector<unsigned>>
    >(regular_axis{3, -1, 1}, integer_axis{-1, 1}, polar_axis{3});
    BOOST_CHECK(h2 == h);
}

BOOST_AUTO_TEST_CASE(init_4)
{
    auto h = dynamic_histogram<
        default_axes,
        adaptive_storage
    >(regular_axis{3, -1, 1},
      integer_axis{-1, 1},
      polar_axis{3},
      variable_axis{-1, 0, 1});
    BOOST_CHECK_EQUAL(h.dim(), 4);
    BOOST_CHECK_EQUAL(h.size(), 300);
    auto h2 = dynamic_histogram<
        default_axes,
        container_storage<std::vector<unsigned>>
    >(regular_axis{3, -1, 1},
      integer_axis{-1, 1},
      polar_axis{3},
      variable_axis{-1, 0, 1});
    BOOST_CHECK(h2 == h);
}

BOOST_AUTO_TEST_CASE(init_5)
{
    auto h = make_dynamic_histogram(regular_axis{3, -1, 1},
                                    integer_axis{-1, 1},
                                    polar_axis{3},
                                    variable_axis{-1, 0, 1},
                                    category_axis{"A", "B", "C"});
    BOOST_CHECK_EQUAL(h.dim(), 5);
    BOOST_CHECK_EQUAL(h.size(), 900);
    auto h2 = make_dynamic_histogram(regular_axis{3, -1, 1},
                                     integer_axis{-1, 1},
                                     polar_axis{3},
                                     variable_axis{-1, 0, 1},
                                     category_axis{"A", "B", "C"});
    BOOST_CHECK(h2 == h);
}

BOOST_AUTO_TEST_CASE(copy_ctor)
{
    auto h = make_dynamic_histogram_with<adaptive_storage>(integer_axis(0, 1),
                                                          integer_axis(0, 2));
    h.fill(0, 0);
    auto h2 = decltype(h)(h);
    BOOST_CHECK(h2 == h);
    auto h3 = dynamic_histogram<
        default_axes,
        container_storage<std::vector<unsigned>>
    >(h);
    BOOST_CHECK(h3 == h);
}

BOOST_AUTO_TEST_CASE(copy_assign)
{
    auto h = make_dynamic_histogram_with<adaptive_storage>(integer_axis(0, 1),
                                                           integer_axis(0, 2));
    h.fill(0, 0);
    auto h2 = decltype(h)();
    BOOST_CHECK(!(h == h2));
    h2 = h;
    BOOST_CHECK(h == h2);
    // test self-assign
    h2 = h2;
    BOOST_CHECK(h == h2);
    auto h3 = dynamic_histogram<
        default_axes,
        container_storage<std::vector<unsigned>>
    >();
    h3 = h;
    BOOST_CHECK(h == h3);
}

BOOST_AUTO_TEST_CASE(move)
{
    auto h = make_dynamic_histogram(integer_axis(0, 1),
                                    integer_axis(0, 2));
    h.fill(0, 0);
    const auto href = h;
    decltype(h) h2(std::move(h));
    BOOST_CHECK_EQUAL(h.dim(), 0);
    BOOST_CHECK_EQUAL(h.sum(), 0);
    BOOST_CHECK_EQUAL(h.size(), 0);
    BOOST_CHECK(h2 == href);
    decltype(h) h3 = std::move(h2);
    BOOST_CHECK_EQUAL(h2.dim(), 0);
    BOOST_CHECK_EQUAL(h2.sum(), 0);
    BOOST_CHECK_EQUAL(h2.size(), 0);
    BOOST_CHECK(h3 == href);
}

BOOST_AUTO_TEST_CASE(equal_compare)
{
    auto a = dynamic_histogram<
        default_axes,
        adaptive_storage
    >(integer_axis(0, 1));
    auto b = dynamic_histogram<
        default_axes,
        adaptive_storage>(integer_axis(0, 1), integer_axis(0, 2));
    BOOST_CHECK(!(a == b));
    BOOST_CHECK(!(b == a));
    auto c = dynamic_histogram<
        mpl::vector<integer_axis>,
        container_storage<std::vector<unsigned>>
    >(integer_axis(0, 1));
    BOOST_CHECK(!(b == c));
    BOOST_CHECK(!(c == b));
    BOOST_CHECK(a == c);
    BOOST_CHECK(c == a);
    auto d = make_dynamic_histogram(regular_axis(2, 0, 1));
    BOOST_CHECK(!(c == d));
    BOOST_CHECK(!(d == c));
    c.fill(0);
    BOOST_CHECK(!(a == c));
    BOOST_CHECK(!(c == a));
    a.fill(0);
    BOOST_CHECK(a == c);
    BOOST_CHECK(c == a);
    a.fill(0);
    BOOST_CHECK(!(a == c));
    BOOST_CHECK(!(c == a));
}

BOOST_AUTO_TEST_CASE(d1)
{
    auto h = make_dynamic_histogram(integer_axis(0, 1));
    h.fill(0);
    h.fill(0);
    h.fill(-1);
    h.fill(10);

    BOOST_CHECK_EQUAL(h.dim(), 1);
    BOOST_CHECK_EQUAL(bins(h.axis(0)), 2);
    BOOST_CHECK_EQUAL(shape(h.axis(0)), 4);
    BOOST_CHECK_EQUAL(h.sum(), 4);

    BOOST_CHECK_THROW(h.value(-2), std::out_of_range);
    BOOST_CHECK_EQUAL(h.value(-1), 1.0);
    BOOST_CHECK_EQUAL(h.value(0), 2.0);
    BOOST_CHECK_EQUAL(h.value(1), 0.0);
    BOOST_CHECK_EQUAL(h.value(2), 1.0);
    BOOST_CHECK_THROW(h.value(3), std::out_of_range);

    BOOST_CHECK_THROW(h.variance(-2), std::out_of_range);
    BOOST_CHECK_EQUAL(h.variance(-1), 1.0);
    BOOST_CHECK_EQUAL(h.variance(0), 2.0);
    BOOST_CHECK_EQUAL(h.variance(1), 0.0);
    BOOST_CHECK_EQUAL(h.variance(2), 1.0);
    BOOST_CHECK_THROW(h.variance(3), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(d1_2)
{
    auto h = make_dynamic_histogram(integer_axis(0, 1, "", false));
    h.fill(0);
    h.fill(-0);
    h.fill(-1);
    h.fill(10);

    BOOST_CHECK_EQUAL(h.dim(), 1);
    BOOST_CHECK_EQUAL(bins(h.axis(0)), 2);
    BOOST_CHECK_EQUAL(shape(h.axis(0)), 2);
    BOOST_CHECK_EQUAL(h.sum(), 2);

    BOOST_CHECK_THROW(h.value(-1), std::out_of_range);
    BOOST_CHECK_EQUAL(h.value(0), 2.0);
    BOOST_CHECK_EQUAL(h.value(1), 0.0);
    BOOST_CHECK_THROW(h.value(2), std::out_of_range);

    BOOST_CHECK_THROW(h.variance(-1), std::out_of_range);
    BOOST_CHECK_EQUAL(h.variance(0), 2.0);
    BOOST_CHECK_EQUAL(h.variance(1), 0.0);
    BOOST_CHECK_THROW(h.variance(2), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(d1w)
{
    auto h = make_dynamic_histogram(regular_axis(2, -1, 1));
    h.fill(0);
    h.wfill(2, -1.0);
    h.fill(-1.0);
    h.fill(-2.0);
    h.wfill(5, 10);

    BOOST_CHECK_EQUAL(h.sum(), 10);

    BOOST_CHECK_EQUAL(h.value(-1), 1.0);
    BOOST_CHECK_EQUAL(h.value(0), 3.0);
    BOOST_CHECK_EQUAL(h.value(1), 1.0);
    BOOST_CHECK_EQUAL(h.value(2), 5.0);

    BOOST_CHECK_EQUAL(h.variance(-1), 1.0);
    BOOST_CHECK_EQUAL(h.variance(0), 5.0);
    BOOST_CHECK_EQUAL(h.variance(1), 1.0);
    BOOST_CHECK_EQUAL(h.variance(2), 25.0);
}

BOOST_AUTO_TEST_CASE(d2)
{
    auto h = make_dynamic_histogram(regular_axis(2, -1, 1),
                                    integer_axis(-1, 1, nullptr, false));
    h.fill(-1, -1);
    h.fill(-1, 0);
    std::array<double, 2> ai = {{-1., -10.}};
    h.fill(ai);
    double in[2] = {-10., 0.};
    h.fill(in, in+2);

    BOOST_CHECK_EQUAL(h.dim(), 2);
    BOOST_CHECK_EQUAL(bins(h.axis(0)), 2);
    BOOST_CHECK_EQUAL(shape(h.axis(0)), 4);
    BOOST_CHECK_EQUAL(bins(h.axis(1)), 3);
    BOOST_CHECK_EQUAL(shape(h.axis(1)), 3);
    BOOST_CHECK_EQUAL(h.sum(), 3);

    BOOST_CHECK_EQUAL(h.value(-1, 0), 0.0);
    BOOST_CHECK_EQUAL(h.value(-1, 1), 1.0);
    BOOST_CHECK_EQUAL(h.value(-1, 2), 0.0);

    BOOST_CHECK_EQUAL(h.value(0, 0), 1.0);
    BOOST_CHECK_EQUAL(h.value(0, 1), 1.0);
    BOOST_CHECK_EQUAL(h.value(0, 2), 0.0);

    BOOST_CHECK_EQUAL(h.value(1, 0), 0.0);
    BOOST_CHECK_EQUAL(h.value(1, 1), 0.0);
    BOOST_CHECK_EQUAL(h.value(1, 2), 0.0);

    BOOST_CHECK_EQUAL(h.value(2, 0), 0.0);
    BOOST_CHECK_EQUAL(h.value(2, 1), 0.0);
    BOOST_CHECK_EQUAL(h.value(2, 2), 0.0);

    BOOST_CHECK_EQUAL(h.variance(-1, 0), 0.0);
    BOOST_CHECK_EQUAL(h.variance(-1, 1), 1.0);
    BOOST_CHECK_EQUAL(h.variance(-1, 2), 0.0);

    BOOST_CHECK_EQUAL(h.variance(0, 0), 1.0);
    BOOST_CHECK_EQUAL(h.variance(0, 1), 1.0);
    BOOST_CHECK_EQUAL(h.variance(0, 2), 0.0);

    BOOST_CHECK_EQUAL(h.variance(1, 0), 0.0);
    BOOST_CHECK_EQUAL(h.variance(1, 1), 0.0);
    BOOST_CHECK_EQUAL(h.variance(1, 2), 0.0);

    BOOST_CHECK_EQUAL(h.variance(2, 0), 0.0);
    BOOST_CHECK_EQUAL(h.variance(2, 1), 0.0);
    BOOST_CHECK_EQUAL(h.variance(2, 2), 0.0);
}

BOOST_AUTO_TEST_CASE(d2w)
{
    auto h = make_dynamic_histogram(regular_axis(2, -1, 1),
                                    integer_axis(-1, 1, nullptr, false));
    h.fill(-1, 0);       // -> 0, 1
    h.wfill(10, -1, -1); // -> 0, 0
    h.wfill(5, -1, -10); // is ignored
    h.wfill(7, -10, 0);  // -> -1, 1

    BOOST_CHECK_EQUAL(h.sum(), 18);

    BOOST_CHECK_EQUAL(h.value(-1, 0), 0.0);
    BOOST_CHECK_EQUAL(h.value(-1, 1), 7.0);
    BOOST_CHECK_EQUAL(h.value(-1, 2), 0.0);

    BOOST_CHECK_EQUAL(h.value(0, 0), 10.0);
    BOOST_CHECK_EQUAL(h.value(0, 1), 1.0);
    BOOST_CHECK_EQUAL(h.value(0, 2), 0.0);

    BOOST_CHECK_EQUAL(h.value(1, 0), 0.0);
    BOOST_CHECK_EQUAL(h.value(1, 1), 0.0);
    BOOST_CHECK_EQUAL(h.value(1, 2), 0.0);

    BOOST_CHECK_EQUAL(h.value(2, 0), 0.0);
    BOOST_CHECK_EQUAL(h.value(2, 1), 0.0);
    BOOST_CHECK_EQUAL(h.value(2, 2), 0.0);

    BOOST_CHECK_EQUAL(h.variance(-1, 0), 0.0);
    BOOST_CHECK_EQUAL(h.variance(-1, 1), 49.0);
    BOOST_CHECK_EQUAL(h.variance(-1, 2), 0.0);

    BOOST_CHECK_EQUAL(h.variance(0, 0), 100.0);
    BOOST_CHECK_EQUAL(h.variance(0, 1), 1.0);
    BOOST_CHECK_EQUAL(h.variance(0, 2), 0.0);

    BOOST_CHECK_EQUAL(h.variance(1, 0), 0.0);
    BOOST_CHECK_EQUAL(h.variance(1, 1), 0.0);
    BOOST_CHECK_EQUAL(h.variance(1, 2), 0.0);

    BOOST_CHECK_EQUAL(h.variance(2, 0), 0.0);
    BOOST_CHECK_EQUAL(h.variance(2, 1), 0.0);
    BOOST_CHECK_EQUAL(h.variance(2, 2), 0.0);
}

BOOST_AUTO_TEST_CASE(d3w)
{
    auto h = make_dynamic_histogram(integer_axis(0, 3),
                                    integer_axis(0, 4),
                                    integer_axis(0, 5));
    for (auto i = 0; i < bins(h.axis(0)); ++i)
        for (auto j = 0; j < bins(h.axis(1)); ++j)
            for (auto k = 0; k < bins(h.axis(2)); ++k)
    {
        h.wfill(i+j+k, i, j, k);
    }

    for (auto i = 0; i < bins(h.axis(0)); ++i)
        for (auto j = 0; j < bins(h.axis(1)); ++j)
            for (auto k = 0; k < bins(h.axis(2)); ++k)
        BOOST_CHECK_EQUAL(h.value(i, j, k), i+j+k);
}

BOOST_AUTO_TEST_CASE(add_0)
{
    auto a = make_dynamic_histogram(integer_axis(-1, 1));
    auto b = make_dynamic_histogram(regular_axis(3, -1, 1));
    auto c = make_dynamic_histogram(regular_axis(3, -1.1, 1));
    BOOST_CHECK_THROW(a += b, std::logic_error);
    BOOST_CHECK_THROW(b += c, std::logic_error);
}

BOOST_AUTO_TEST_CASE(add_1)
{
    auto a = dynamic_histogram<
            mpl::vector<integer_axis>,
            adaptive_storage
        >(integer_axis(-1, 1));
    auto b = dynamic_histogram<
            mpl::vector<integer_axis, regular_axis>,
            container_storage<std::vector<unsigned>>
        >(integer_axis(-1, 1));
    a.fill(-1);
    b.fill(1);
    auto c = a;
    c += b;
    BOOST_CHECK_EQUAL(c.value(-1), 0);
    BOOST_CHECK_EQUAL(c.value(0), 1);
    BOOST_CHECK_EQUAL(c.value(1), 0);
    BOOST_CHECK_EQUAL(c.value(2), 1);
    BOOST_CHECK_EQUAL(c.value(3), 0);
    auto d = b;
    d += a;
    BOOST_CHECK_EQUAL(d.value(-1), 0);
    BOOST_CHECK_EQUAL(d.value(0), 1);
    BOOST_CHECK_EQUAL(d.value(1), 0);
    BOOST_CHECK_EQUAL(d.value(2), 1);
    BOOST_CHECK_EQUAL(d.value(3), 0);
}

BOOST_AUTO_TEST_CASE(add_2)
{
    auto a = make_dynamic_histogram(integer_axis(-1, 1));
    auto b = make_dynamic_histogram(integer_axis(-1, 1));

    a.fill(0);
    b.wfill(3, -1);
    auto c = a;
    c += b;
    BOOST_CHECK_EQUAL(c.value(-1), 0);
    BOOST_CHECK_EQUAL(c.value(0), 3);
    BOOST_CHECK_EQUAL(c.value(1), 1);
    BOOST_CHECK_EQUAL(c.value(2), 0);
    BOOST_CHECK_EQUAL(c.value(3), 0);    
    auto d = b;
    d += a;
    BOOST_CHECK_EQUAL(d.value(-1), 0);
    BOOST_CHECK_EQUAL(d.value(0), 3);
    BOOST_CHECK_EQUAL(d.value(1), 1);
    BOOST_CHECK_EQUAL(d.value(2), 0);
    BOOST_CHECK_EQUAL(d.value(3), 0);
}

BOOST_AUTO_TEST_CASE(add_3)
{
    auto a = make_dynamic_histogram_with<container_storage<std::vector<char>>>(integer_axis(-1, 1));
    auto b = make_dynamic_histogram_with<container_storage<std::vector<unsigned>>>(integer_axis(-1, 1));
    a.fill(-1);
    b.fill(1);
    auto c = a;
    c += b;
    BOOST_CHECK_EQUAL(c.value(-1), 0);
    BOOST_CHECK_EQUAL(c.value(0), 1);
    BOOST_CHECK_EQUAL(c.value(1), 0);
    BOOST_CHECK_EQUAL(c.value(2), 1);
    BOOST_CHECK_EQUAL(c.value(3), 0);
    auto d = b;
    d += a;
    BOOST_CHECK_EQUAL(d.value(-1), 0);
    BOOST_CHECK_EQUAL(d.value(0), 1);
    BOOST_CHECK_EQUAL(d.value(1), 0);
    BOOST_CHECK_EQUAL(d.value(2), 1);
    BOOST_CHECK_EQUAL(d.value(3), 0);
}

BOOST_AUTO_TEST_CASE(bad_add)
{
    auto a = make_dynamic_histogram(integer_axis(0, 1));
    auto b = make_dynamic_histogram(integer_axis(0, 2));
    BOOST_CHECK_THROW(a += b, std::logic_error);
}

BOOST_AUTO_TEST_CASE(bad_index)
{
    auto a = make_dynamic_histogram(integer_axis(0, 1));
    BOOST_CHECK_THROW(a.value(5), std::out_of_range);
    BOOST_CHECK_THROW(a.value(-5), std::out_of_range);
    BOOST_CHECK_THROW(a.variance(5), std::out_of_range);
    BOOST_CHECK_THROW(a.variance(-5), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(histogram_serialization)
{
    auto a = make_dynamic_histogram(regular_axis(3, -1, 1, "r"),
                                    polar_axis(4, 0.0, "p"),
                                    variable_axis({0.1, 0.2, 0.3, 0.4, 0.5}, "v"),
                                    category_axis{"A", "B", "C"},
                                    integer_axis(0, 1, "i"));
    a.fill(0.5, 0.1, 0.25, 1, 0);
    std::string buf;
    {
        std::ostringstream os;
        boost::archive::text_oarchive oa(os);
        oa << a;
        buf = os.str();
    }
    auto b = make_dynamic_histogram();
    BOOST_CHECK(!(a == b));
    {
        std::istringstream is(buf);
        boost::archive::text_iarchive ia(is);
        ia >> b;
    }
    BOOST_CHECK(a == b);
}
