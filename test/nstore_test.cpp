#include <boost/histogram/detail/nstore.hpp>
#include <boost/histogram/detail/wtype.hpp>
#define BOOST_TEST_MODULE nstore_test
#include <boost/test/unit_test.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/move/move.hpp>
#include <sstream>
#include <string>
#include <limits>
using namespace boost::histogram::detail;

BOOST_AUTO_TEST_CASE(wtype_streamer)
{
    std::ostringstream os;
    wtype w;
    w.w = 1.2;
    w.w2 = 2.3;
    os << w;
    BOOST_CHECK_EQUAL(os.str(), std::string("(1.2,2.3)"));
}

BOOST_AUTO_TEST_CASE(nstore_bad_alloc)
{
    BOOST_CHECK_THROW(nstore(nstore::size_type(-1)),
                      std::bad_alloc);
}

BOOST_AUTO_TEST_CASE(nstore_grow)
{
    double x = 1.0;
    nstore n(1);
    n.increase(0);
    for (unsigned i = 0; i < 64; ++i) {
        n += n;
        x += x;
        BOOST_CHECK_EQUAL(n.value(0), x);
        BOOST_CHECK_EQUAL(n.variance(0), x);
    }
    BOOST_CHECK_EQUAL(n.variance(0), x);
}

BOOST_AUTO_TEST_CASE(nstore_bad_add)
{
    nstore a(1), b(2);
    BOOST_CHECK_THROW(a += b, std::logic_error);
}

BOOST_AUTO_TEST_CASE(nstore_add_with_growth)
{
    nstore a(1), b(1), c(1, 0);
    a.increase(0);
    for (unsigned i = 0; i < 10; ++i)
        a += a;
    b.increase(0);
    b += a;
    BOOST_CHECK_EQUAL(b.value(0), a.value(0) + 1.0);
    // c += a;
    // BOOST_CHECK_EQUAL(c.value(0), a.value(0));
}

BOOST_AUTO_TEST_CASE(nstore_equality)
{
    nstore a(1), b(1), c(2);
    BOOST_CHECK(a == b);
    a.increase(0);
    BOOST_CHECK(!(a == b));
    BOOST_CHECK(!(b == c));
}
