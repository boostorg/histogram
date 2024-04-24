#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/collector.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/make_histogram.hpp>
#include "throw_exception.hpp"

#include <array>
#include <vector>

using namespace boost::histogram;

int main() {
  using collector_t = accumulators::collector<>;

  auto h = make_histogram_with(dense_storage<collector_t>(), axis::integer<>(0, 5));

  h(0, sample(1.1));
  h(0, sample(2.2));
  h(1, sample(10.10));

  BOOST_TEST_EQ(h.at(0).count(), 2);
  BOOST_TEST_EQ(h.at(1).count(), 1);
  BOOST_TEST_EQ(h.at(2).count(), 0);

  BOOST_TEST_EQ(h.at(0)[0], 1.1);
  BOOST_TEST_EQ(h.at(0)[1], 2.2);
  BOOST_TEST_EQ(h.at(1)[0], 10.10);
  BOOST_TEST_EQ(h.at(0), collector_t({1.1, 2.2}));

  std::vector<int> x = {0, 0, 0, 0, 0};
  std::array<double, 5> data = {-1.1, -1.2, -1.3, -1.4, -1.5};
  h.fill(x, sample(data));

  BOOST_TEST_EQ(h.at(0).count(), 7);
  BOOST_TEST_EQ(h.at(1).count(), 1);
  BOOST_TEST_EQ(h.at(2).count(), 0);
  BOOST_TEST_NE(h.at(0), collector_t({1.1, 2.2}));
  BOOST_TEST_EQ(h.at(0), collector_t({1.1, 2.2, -1.1, -1.2, -1.3, -1.4, -1.5}));
  BOOST_TEST_EQ(h.at(1), collector_t({10.10}));

  return boost::report_errors();
}
