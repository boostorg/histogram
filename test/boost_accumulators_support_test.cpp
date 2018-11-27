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
#include <boost/histogram/storage_adaptor.hpp>
#include <vector>

using namespace boost::histogram;

int main() {
  using namespace boost::accumulators;
  using element = accumulator_set<double, stats<tag::mean>>;
  auto a = storage_adaptor<std::vector<element>>();
  a.reset(3);
  a(0, 1);
  a(0, 2);
  a(0, 3);
  a(1, 2);
  a(1, 3);
  BOOST_TEST_EQ(count(a[0]), 3);
  BOOST_TEST_EQ(mean(a[0]), 2);
  BOOST_TEST_EQ(count(a[1]), 2);
  BOOST_TEST_EQ(mean(a[1]), 2.5);
  BOOST_TEST_EQ(count(a[2]), 0);

  auto b = a; // copy ok
  // b += a; // accumulators do not implement operator+=
  return boost::report_errors();
}
