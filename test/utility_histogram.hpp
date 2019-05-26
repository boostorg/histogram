// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_HISTOGRAM_HPP
#define BOOST_HISTOGRAM_TEST_UTILITY_HISTOGRAM_HPP

#include <boost/histogram/axis.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/make_histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/mp11/algorithm.hpp>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {

template <typename... Ts>
auto make_axis_vector(const Ts&... ts) {
  using Var = boost::mp11::mp_unique<axis::variant<Ts...>>;
  return std::vector<Var>({Var(ts)...});
}

using static_tag = std::false_type;
using dynamic_tag = std::true_type;

template <typename... Axes>
auto make(static_tag, const Axes&... axes) {
  return make_histogram(axes...);
}

template <typename S, typename... Axes>
auto make_s(static_tag, S&& s, const Axes&... axes) {
  return make_histogram_with(s, axes...);
}

template <typename... Axes>
auto make(dynamic_tag, const Axes&... axes) {
  return make_histogram(make_axis_vector(axes...));
}

template <typename S, typename... Axes>
auto make_s(dynamic_tag, S&& s, const Axes&... axes) {
  return make_histogram_with(s, make_axis_vector(axes...));
}

} // namespace histogram
} // namespace boost

#endif
