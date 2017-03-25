// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_HPP_
#define BOOST_HISTOGRAM_UTILITY_HPP_

#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/dynamic_histogram.hpp>
#include <boost/histogram/static_histogram.hpp>
#include <boost/variant.hpp>

namespace boost {
namespace histogram {

template <typename A> int bins(const A &a) { return a.bins(); }

template <typename... Axes> int bins(const boost::variant<Axes...> &a) {
  return apply_visitor(detail::bins(), a);
}

template <typename A> int shape(const A &a) { return a.shape(); }

template <typename... Axes> int shape(const boost::variant<Axes...> &a) {
  return apply_visitor(detail::shape(), a);
}

template <typename A, typename V> int index(const A &a, const V v) {
  return a.index(v);
}

template <typename... Axes, typename V>
int index(const boost::variant<Axes...> &a, const V v) {
  return apply_visitor(detail::index<V>(v), a);
}

template <typename A> typename A::value_type left(const A &a, const int i) {
  return a[i];
}

template <typename... Axes>
double left(const boost::variant<Axes...> &a, const int i) {
  return apply_visitor(detail::left(i), a);
}

template <typename A> typename A::value_type right(const A &a, const int i) {
  return a[i + 1];
}

template <typename... Axes>
double right(const boost::variant<Axes...> &a, const int i) {
  return apply_visitor(detail::right(i), a);
}

template <typename A> typename A::value_type center(const A &a, const int i) {
  return 0.5 * (a[i] + a[i + 1]);
}

template <typename... Axes>
double center(const boost::variant<Axes...> &a, const int i) {
  return apply_visitor(detail::center(i), a);
}

template <typename Storage, typename Axes, typename Visitor>
void for_each_axis(const static_histogram<Storage, Axes> &h, Visitor &visitor) {
  fusion::for_each(h.axes_, visitor);
}

template <typename Storage, typename Axes, typename Visitor>
void for_each_axis(const dynamic_histogram<Storage, Axes> &h,
                   Visitor &visitor) {
  for (const auto &a : h.axes_)
    apply_visitor(visitor, a);
}

} // namespace histogram
} // namespace boost

#endif
