// Copyright 2026 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <unordered_map>
#include <vector>
#include "throw_exception.hpp"

namespace bh = boost::histogram;

int main() {
  using cell_t = bh::accumulators::weighted_sum<double>;

  auto h = bh::make_histogram_with(std::unordered_map<std::size_t, cell_t>(),
                                   bh::axis::regular<>(10, 0.0, 1.0));

  const std::vector<double> values = {0.2, 0.3};
  const std::vector<double> weights = {1.0, 1.0};
  h.fill(values, bh::weight(weights));

  const auto s = bh::algorithm::sum(h);
  BOOST_TEST_EQ(s.value(), 2.0);
  BOOST_TEST_EQ(s.variance(), 2.0);

  return boost::report_errors();
}