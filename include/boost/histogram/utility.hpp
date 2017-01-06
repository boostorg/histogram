// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_HPP_
#define BOOST_HISTOGRAM_UTILITY_HPP_

#include <boost/variant.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>

namespace boost {
namespace histogram {

template <typename A>
int bins(const A& a) { return a.bins(); }

template <typename... Axes>
int bins(const boost::variant<Axes...>& a)
{ return apply_visitor(detail::bins(), a); }

template <typename A>
int shape(const A& a) { return a.shape(); }

template <typename... Axes>
int shape(const boost::variant<Axes...>& a)
{ return apply_visitor(detail::shape(), a); }

template <typename A, typename V>
int index(const A& a, const V v) { return a.index(v); }

template <typename... Axes, typename V>
int index(const boost::variant<Axes...>& a, const V v)
{ return apply_visitor(detail::index<V>(v), a); }

template <typename A>
double left(const A& a, const int i) { return a.left(i); }

template <typename... Axes>
double left(const boost::variant<Axes...>& a, const int i)
{ return apply_visitor(detail::left(i), a); }

template <typename A>
double right(const A& a, const int i) { return a.right(i); }

template <typename... Axes>
double right(const boost::variant<Axes...>& a, const int i)
{ return apply_visitor(detail::right(i), a); }

template <typename A>
double center(const A& a, const int i) { return 0.5 * (a.left(i) + a.right(i)); }

template <typename... Axes>
double center(const boost::variant<Axes...>& a, const int i)
{ return apply_visitor(detail::center(i), a); }

}
}

#endif
