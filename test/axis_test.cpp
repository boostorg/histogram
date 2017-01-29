// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE axis_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/variant.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/math/constants/constants.hpp>
#include <limits>
using namespace boost::histogram;

using axis_t = boost::variant<
    regular_axis, polar_axis, variable_axis,
    category_axis, integer_axis
>;

BOOST_AUTO_TEST_CASE(bad_ctors) {
    BOOST_CHECK_THROW(regular_axis(0, 0, 1), std::logic_error);
    BOOST_CHECK_THROW(regular_axis(1, 1, -1), std::logic_error);
    BOOST_CHECK_THROW(polar_axis(0), std::logic_error);
    BOOST_CHECK_THROW(variable_axis({}), std::logic_error);
    BOOST_CHECK_THROW(variable_axis({1.0}), std::logic_error);
    BOOST_CHECK_THROW(integer_axis(1, -1), std::logic_error);
    BOOST_CHECK_THROW(category_axis({}), std::logic_error);
}

BOOST_AUTO_TEST_CASE(regular_axis_operators) {
    regular_axis a{4, -2, 2};
    BOOST_CHECK_EQUAL(a[-1], -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(a[a.bins() + 1], std::numeric_limits<double>::infinity());
    regular_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
    BOOST_CHECK_EQUAL(a.index(-10.), -1);
    BOOST_CHECK_EQUAL(a.index(-2.1), -1);
    BOOST_CHECK_EQUAL(a.index(-2.0), 0);
    BOOST_CHECK_EQUAL(a.index(-1.1), 0);
    BOOST_CHECK_EQUAL(a.index(0.0), 2);
    BOOST_CHECK_EQUAL(a.index(0.9), 2);
    BOOST_CHECK_EQUAL(a.index(1.0), 3);
    BOOST_CHECK_EQUAL(a.index(10.), 4);
    BOOST_CHECK_EQUAL(a.index(std::numeric_limits<double>::infinity()), 4);
    BOOST_CHECK_EQUAL(a.index(-std::numeric_limits<double>::infinity()), -1);
    BOOST_CHECK_EQUAL(a.index(std::numeric_limits<double>::quiet_NaN()), -1);
}

BOOST_AUTO_TEST_CASE(polar_axis_operators) {
    using namespace boost::math::double_constants;
    polar_axis a{4};
    BOOST_CHECK_EQUAL(a[-1], a[a.bins() - 1] - two_pi);
    polar_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
    BOOST_CHECK_EQUAL(a.index(-1.0 * two_pi), 0);
    BOOST_CHECK_EQUAL(a.index(0.0), 0);
    BOOST_CHECK_EQUAL(a.index(0.25 * two_pi), 1);
    BOOST_CHECK_EQUAL(a.index(0.5 * two_pi), 2);
    BOOST_CHECK_EQUAL(a.index(0.75 * two_pi), 3);
    BOOST_CHECK_EQUAL(a.index(1.00 * two_pi), 0);
    BOOST_CHECK_EQUAL(a.index(std::numeric_limits<double>::infinity()), 0);
    BOOST_CHECK_EQUAL(a.index(-std::numeric_limits<double>::infinity()), 0);
    BOOST_CHECK_EQUAL(a.index(std::numeric_limits<double>::quiet_NaN()), 0);
}

BOOST_AUTO_TEST_CASE(variable_axis_operators) {
    variable_axis a{-1, 0, 1};
    BOOST_CHECK_EQUAL(a[-1], -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(a[a.bins() + 1], std::numeric_limits<double>::infinity());
    variable_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
    variable_axis c{-2, 0, 2};
    BOOST_CHECK_NE(a, c);
    BOOST_CHECK_EQUAL(a.index(-10.), -1);
    BOOST_CHECK_EQUAL(a.index(-1.), 0);
    BOOST_CHECK_EQUAL(a.index(0.), 1);
    BOOST_CHECK_EQUAL(a.index(1.), 2);
    BOOST_CHECK_EQUAL(a.index(10.), 2);
    BOOST_CHECK_EQUAL(a.index(std::numeric_limits<double>::infinity()), 2);
    BOOST_CHECK_EQUAL(a.index(-std::numeric_limits<double>::infinity()), -1);
    BOOST_CHECK_EQUAL(a.index(std::numeric_limits<double>::quiet_NaN()), 2);
}

BOOST_AUTO_TEST_CASE(category_axis_operators) {
    category_axis a{"A", "B", "C"};
    category_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
    BOOST_CHECK_EQUAL(a.index(0), 0);
    BOOST_CHECK_EQUAL(a.index(1), 1);
    BOOST_CHECK_EQUAL(a.index(2), 2);

    BOOST_CHECK_THROW(a.index(-1), std::out_of_range);
    BOOST_CHECK_THROW(a.index(3), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(integer_axis_operators) {
    integer_axis a{-1, 1};
    integer_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
    BOOST_CHECK_EQUAL(a.index(-10), -1);
    BOOST_CHECK_EQUAL(a.index(-2), -1);
    BOOST_CHECK_EQUAL(a.index(-1), 0);
    BOOST_CHECK_EQUAL(a.index(0), 1);
    BOOST_CHECK_EQUAL(a.index(1), 2);
    BOOST_CHECK_EQUAL(a.index(2), 3);
    BOOST_CHECK_EQUAL(a.index(10), 3);
}

BOOST_AUTO_TEST_CASE(axis_t_streamable) {
    std::vector<axis_t> axes;
    axes.push_back(regular_axis{2, -1, 1, "regular", false});
    axes.push_back(polar_axis{4, 0.1, "polar"});
    axes.push_back(variable_axis{{-1, 0, 1}, "variable", false});
    axes.push_back(category_axis{"A", "B", "C"});
    axes.push_back(integer_axis{-1, 1, "integer", false});
    std::ostringstream os;
    for(const auto& a : axes)
        os << a;
    BOOST_CHECK(!os.str().empty());
}

BOOST_AUTO_TEST_CASE(axis_t_equal_comparable) {
    std::vector<axis_t> axes;
    axes.push_back(regular_axis{2, -1, 1});
    axes.push_back(polar_axis{4});
    axes.push_back(variable_axis{-1, 0, 1});
    axes.push_back(category_axis{"A", "B", "C"});
    axes.push_back(integer_axis{-1, 1});
    for (const auto& a : axes) {
        BOOST_CHECK(!(a == axis_t()));
        BOOST_CHECK_EQUAL(a, a);
    }
    BOOST_CHECK(!(axes == std::vector<axis_t>()));
    BOOST_CHECK(axes == std::vector<axis_t>(axes));
}
