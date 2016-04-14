#include <boost/histogram/histogram.hpp>
#define BOOST_TEST_MODULE histogram_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/assign/std/vector.hpp>
using namespace boost::assign;
using namespace boost::histogram;
namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(init_0)
{
    histogram();
}

BOOST_AUTO_TEST_CASE(init_1)
{
    histogram(regular_axis(3, -1, 1));
}

BOOST_AUTO_TEST_CASE(init_2)
{
    histogram(regular_axis(3, -1, 1),
              integer_axis(-1, 1));    
}

BOOST_AUTO_TEST_CASE(init_3)
{
    histogram(regular_axis(3, -1, 1),
              integer_axis(-1, 1),
              polar_axis(3));    
}

BOOST_AUTO_TEST_CASE(init_4)
{
    std::vector<double> x;
    x += -1, 0, 1;
    histogram(regular_axis(3, -1, 1),
              integer_axis(-1, 1),
              polar_axis(3),
              variable_axis(x));    
}

BOOST_AUTO_TEST_CASE(init_5)
{
    std::vector<double> x;
    x += -1, 0, 1;
    histogram(regular_axis(3, -1, 1),
              integer_axis(-1, 1),
              polar_axis(3),
              variable_axis(x),
              category_axis("A;B;C"));    
}

BOOST_AUTO_TEST_CASE(init_15)
{
    std::vector<double> x;
    x += -1, 0, 1, 2, 3;
    histogram(regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),

              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),

              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1),
              regular_axis(1, -1, 1));
}

BOOST_AUTO_TEST_CASE(d1)
{
    histogram h(regular_axis(2, -1, 1));
    h.fill(-1);
    h.fill(-1.0);
    h.fill(-2.0);
    h.fill(10.0);

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

BOOST_AUTO_TEST_CASE(d1w)
{
    histogram h(regular_axis(2, -1, 1));
    h.fill(0);
    h.wfill(-1.0, 2.0);
    h.fill(-1.0);
    h.fill(-2.0);
    h.wfill(10.0, 5.0);

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
    histogram h(regular_axis(2, -1, 1),
                integer_axis(-1, 1, std::string(), false));
    h.fill(-1, -1);
    h.fill(-1, 0);
    h.fill(-1, -10);
    h.fill(-10, 0);

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
    histogram h(regular_axis(2, -1, 1),
                integer_axis(-1, 1, std::string(), false));
    h.fill(-1, 0);       // -> 0, 1
    h.wfill(-1, -1, 10); // -> 0, 0
    h.wfill(-1, -10, 5); // is ignored
    h.wfill(-10, 0, 7);  // -> -1, 1

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

BOOST_AUTO_TEST_CASE(add_0)
{
    histogram a(integer_axis(-1, 1));
    histogram b(regular_axis(3, -1, 1));
    BOOST_CHECK_THROW(a + b, std::logic_error);
}

BOOST_AUTO_TEST_CASE(add_1)
{
    histogram a(integer_axis(-1, 1));
    histogram b(integer_axis(-1, 1));
    a.fill(0);
    b.fill(-1);
    histogram c = a + b;
    BOOST_CHECK_EQUAL(c.value(-1), 0);
    BOOST_CHECK_EQUAL(c.value(0), 1);
    BOOST_CHECK_EQUAL(c.value(1), 1);
    BOOST_CHECK_EQUAL(c.value(2), 0);
    BOOST_CHECK_EQUAL(c.value(3), 0);
}

BOOST_AUTO_TEST_CASE(add_2)
{
    histogram a(integer_axis(-1, 1));
    histogram b(integer_axis(-1, 1));

    a.fill(0);
    b.wfill(-1, 3);
    histogram c = a + b;
    BOOST_CHECK_EQUAL(c.value(-1), 0);
    BOOST_CHECK_EQUAL(c.value(0), 3);
    BOOST_CHECK_EQUAL(c.value(1), 1);
    BOOST_CHECK_EQUAL(c.value(2), 0);
    BOOST_CHECK_EQUAL(c.value(3), 0);    
}
