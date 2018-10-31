// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_AXIS_HPP
#define BOOST_HISTOGRAM_TEST_UTILITY_AXIS_HPP

#include <boost/core/lightweight_test.hpp>

namespace boost {
namespace histogram {

template <typename Axis>
void test_axis_iterator(const Axis& a, int begin, int end) {
  for (auto bin : a) {
    BOOST_TEST_EQ(bin.idx(), begin);
    BOOST_TEST_EQ(bin, a[begin]);
    ++begin;
  }
  BOOST_TEST_EQ(begin, end);
  auto rit = a.rbegin();
  for (; rit != a.rend(); ++rit) {
    BOOST_TEST_EQ(rit->idx(), --begin);
    BOOST_TEST_EQ(*rit, a[begin]);
  }
}

#define BOOST_TEST_IS_CLOSE(a, b, eps) BOOST_TEST(std::abs(a - b) < eps)

} // namespace histogram
} // namespace boost

#endif
