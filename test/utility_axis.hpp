// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_AXIS_HPP
#define BOOST_HISTOGRAM_TEST_UTILITY_AXIS_HPP

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/interval_bin_view.hpp>
#include <boost/histogram/axis/polymorphic_bin_view.hpp>
#include <boost/histogram/axis/value_bin_view.hpp>

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

namespace axis {
template <typename... Ts, typename... Us>
bool operator==(const polymorphic_bin_view<Ts...>& t, const interval_bin_view<Us...>& u) {
  return t.idx() == u.idx() && t.lower() == u.lower() && t.upper() == u.upper();
}

template <typename... Ts, typename... Us>
bool operator==(const polymorphic_bin_view<Ts...>& t, const value_bin_view<Us...>& u) {
  return t.idx() == u.idx() && t.value() == u.value();
}
} // namespace axis

#define BOOST_TEST_IS_CLOSE(a, b, eps) BOOST_TEST(std::abs(a - b) < eps)

} // namespace histogram
} // namespace boost

#endif
