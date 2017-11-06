// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_HPP_
#define BOOST_HISTOGRAM_UTILITY_HPP_

#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/variant/variant_fwd.hpp>

namespace boost {
namespace histogram {

template <typename A> inline int size(const A &a) { return a.size(); }

template <typename... Axes> inline int size(const boost::variant<Axes...> &a) {
  return apply_visitor(detail::size(), a);
}

template <typename A> inline int shape(const A &a) { return a.shape(); }

template <typename... Axes> inline int shape(const boost::variant<Axes...> &a) {
  return apply_visitor(detail::shape(), a);
}

template <typename A, typename V> inline int index(const A &a, const V v) {
  return a.index(v);
}

template <typename... Axes, typename V>
inline int index(const boost::variant<Axes...> &a, const V v) {
  return apply_visitor(detail::index<V>(v), a);
}

template <typename A> inline typename A::bin_type bin(const A &a, const int i) {
  return a[i];
}

template <typename... Axes>
inline axis::interval<double> bin(const boost::variant<Axes...> &a,
                                  const int i) {
  return apply_visitor(detail::bin(i), a);
}

} // namespace histogram
} // namespace boost

#endif
