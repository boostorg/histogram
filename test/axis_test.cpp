#include <boost/histogram/axis.hpp>
#define BOOST_TEST_MODULE axis_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/assign/std/vector.hpp>
#include <limits>
using namespace boost::assign;
using namespace boost::histogram;

// only test things not already covered by python_test_suite

BOOST_AUTO_TEST_CASE(regular_axis_operators) {
    regular_axis a(3, -1, 1);
    BOOST_CHECK_EQUAL(a[-1], -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(a[a.bins() + 1], std::numeric_limits<double>::infinity());
    regular_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
}

BOOST_AUTO_TEST_CASE(variable_axis_operators) {
    std::vector<double> x;
    x += -1, 0, 1;
    variable_axis a(x);
    BOOST_CHECK_EQUAL(a[-1], -std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(a[a.bins() + 1], std::numeric_limits<double>::infinity());
    variable_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
}
