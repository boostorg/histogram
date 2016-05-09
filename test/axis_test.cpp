#include <boost/histogram/axis.hpp>
#define BOOST_TEST_MODULE axis_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/foreach.hpp>
#include <sstream>
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

BOOST_AUTO_TEST_CASE(polar_axis_operators) {
    using namespace boost::math::double_constants;
    polar_axis a(4);
    BOOST_CHECK_EQUAL(a[-1], a[a.bins() - 1] - two_pi);
    polar_axis b;
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

BOOST_AUTO_TEST_CASE(category_axis_operators) {
    category_axis a("A;B;C");
    category_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
}

BOOST_AUTO_TEST_CASE(integer_axis_operators) {
    integer_axis a(-1, 1);
    integer_axis b;
    BOOST_CHECK_NE(a, b);
    b = a;
    BOOST_CHECK_EQUAL(a, b);
    b = b;
    BOOST_CHECK_EQUAL(a, b);
}

BOOST_AUTO_TEST_CASE(axis_type_streamable) {
    std::vector<double> x;
    x += -1, 0, 1;
    std::vector<axis_type> axes;
    axes.push_back(regular_axis(2, -1, 1));
    axes.push_back(polar_axis(4));
    axes.push_back(variable_axis(x));
    axes.push_back(category_axis("A;B;C"));
    axes.push_back(integer_axis(-1, 1));
    std::ostringstream os;
    BOOST_FOREACH(const axis_type& a, axes)
        os << a;
    BOOST_CHECK(!os.str().empty());
}
