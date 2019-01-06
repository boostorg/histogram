#include <boost/config/workaround.hpp>
#if BOOST_WORKAROUND(BOOST_GCC, >= 50000)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#if BOOST_WORKAROUND(BOOST_GCC, >= 60000)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#endif
#if defined(BOOST_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#if defined(BOOST_CLANG)
#pragma clang diagnostic pop
#endif
#if BOOST_WORKAROUND(BOOST_GCC, >= 60000)
#pragma GCC diagnostic pop
#endif
#if BOOST_WORKAROUND(BOOST_GCC, >= 50000)
#pragma GCC diagnostic pop
#endif

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <vector>
#include "utility_histogram.hpp"

namespace ba = boost::accumulators;
using namespace boost::histogram;

int main() {
  {
    using element = ba::accumulator_set<double, ba::stats<ba::tag::mean>>;
    auto h = make_histogram_with(dense_storage<element>(), axis::integer<>(0, 2));
    h(0, sample(1));
    h(0, sample(2));
    h(0, sample(3));
    h(1, sample(2));
    h(1, sample(3));
    BOOST_TEST_EQ(ba::count(h[0]), 3);
    BOOST_TEST_EQ(ba::mean(h[0]), 2);
    BOOST_TEST_EQ(ba::count(h[1]), 2);
    BOOST_TEST_EQ(ba::mean(h[1]), 2.5);
    BOOST_TEST_EQ(ba::count(h[2]), 0);

    auto h2 = h; // copy ok
    BOOST_TEST_EQ(ba::count(h2[0]), 3);
    BOOST_TEST_EQ(ba::mean(h2[0]), 2);
    BOOST_TEST_EQ(ba::count(h2[1]), 2);
    BOOST_TEST_EQ(ba::mean(h2[1]), 2.5);
    BOOST_TEST_EQ(ba::count(h2[2]), 0);
  }

  {
    using element =
        ba::accumulator_set<double, ba::stats<ba::tag::weighted_mean>, double>;
    auto h = make_histogram_with(dense_storage<element>(), axis::integer<>(0, 2));
    h(0, sample(1), weight(3));
    // BOOST_TEST_EQ(ba::count(h[0]), 3);
    // BOOST_TEST_EQ(ba::mean(h[0]), 1);
  }
  return boost::report_errors();
}
