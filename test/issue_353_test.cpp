#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <vector>
#include "throw_exception.hpp"

namespace bh = boost::histogram;

struct my_axis {
  int index(double) { return 0; }
  int size() const { return 0; }
};

int main() {
  auto h =
      bh::make_histogram_with(std::vector<int>(), bh::axis::integer<>(0, 2), my_axis());

  auto ind = bh::indexed(h, bh::coverage::inner);

  BOOST_TEST_EQ(std::distance(ind.begin(), ind.end()), 0);

  return boost::report_errors();
}