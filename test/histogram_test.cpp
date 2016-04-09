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

BOOST_AUTO_TEST_CASE(fill_1)
{
    histogram h(regular_axis(3, -1, 1));
    h.fill(1);
}
