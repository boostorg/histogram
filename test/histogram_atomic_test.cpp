// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/count.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/ostream.hpp>
#include <vector>
#include "throw_exception.hpp"
#include "utility_histogram.hpp"

using namespace boost::histogram;

using in = axis::integer<int, axis::null_type>;

template <class Tag>
void run_tests() {

  // int
  {
    auto h = make_s(Tag(), std::vector<accumulators::count<int, true>>(), in{0, 2});
    for (int i = -1; i < 4; ++i) h(i);
    // BOOST_TEST_EQ(algorithm::sum(h), 5);
    BOOST_TEST_EQ(h[-1], 1);
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 1);
    BOOST_TEST_EQ(h[2], 2);
  }

  // float
  {
    auto h = make_s(Tag(), std::vector<accumulators::count<float, true>>(), in{0, 2});
    for (int i = -1; i < 4; ++i) h(i);
    // BOOST_TEST_EQ(algorithm::sum(h), 5);
    BOOST_TEST_EQ(h[-1], 1);
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 1);
    BOOST_TEST_EQ(h[2], 2);
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
