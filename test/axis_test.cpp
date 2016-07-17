// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_TEST_MODULE axis_test
#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis_ostream_operators.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/math/constants/constants.hpp>
#include <sstream>
#include <limits>
using namespace boost::histogram;

// only test things not already covered by python_test_suite

BOOST_AUTO_TEST_CASE(regular_axis_operators) {
    regular_axis a{3, -1, 1};
    BOOST_CHECK_EQUAL(a[-1], -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(a[a.bins() + 1], std::numeric_limits<double>::infinity());
    regular_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
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
}

BOOST_AUTO_TEST_CASE(category_axis_operators) {
    category_axis a{"A", "B", "C"};
    category_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
}

BOOST_AUTO_TEST_CASE(integer_axis_operators) {
    integer_axis a{-1, 1};
    integer_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
}

BOOST_AUTO_TEST_CASE(axis_t_streamable) {
    std::vector<axis_t> axes;
    axes.push_back(regular_axis{2, -1, 1});
    axes.push_back(polar_axis{4});
    axes.push_back(variable_axis{-1, 0, 1});
    axes.push_back(category_axis{"A", "B", "C"});
    axes.push_back(integer_axis{-1, 1});
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
