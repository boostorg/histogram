// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_HPP_
#define BOOST_HISTOGRAM_UTILITY_HPP_

#include <boost/variant/variant_fwd.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>

namespace boost {
namespace histogram {

template <typename A> inline int bins(const A &a) { return a.bins(); }

template <typename... Axes> inline int bins(const boost::variant<Axes...> &a) {
  return apply_visitor(detail::bins(), a);
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

template <typename A> inline typename A::value_type left(const A &a, const int i) {
  return a[i];
}

template <typename... Axes>
inline double left(const boost::variant<Axes...> &a, const int i) {
  return apply_visitor(detail::left(i), a);
}

template <typename A> inline typename A::value_type right(const A &a, const int i) {
  return left(a, i + 1);
}

template <typename... Axes>
inline double right(const boost::variant<Axes...> &a, const int i) {
  return apply_visitor(detail::right(i), a);
}

template <typename A> inline double center(const A &a, const int i) {
  return 0.5 * (left(a, i) + right(a, i));
}

template <typename... Axes>
inline double center(const boost::variant<Axes...> &a, const int i) {
  return apply_visitor(detail::center(i), a);
}

} // namespace histogram
} // namespace boost

#endif
