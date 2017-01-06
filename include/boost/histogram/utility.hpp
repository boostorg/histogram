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

}
}

#endif
