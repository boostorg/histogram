#ifndef BOOST_HISTOGRAM_UTILITY_HPP_
#define BOOST_HISTOGRAM_UTILITY_HPP_

#include <boost/variant.hpp>
#include <boost/histogram/visitors.hpp>

namespace boost {
namespace histogram {

template <typename A>
int bins(const A& a) { return a.bins(); }

template <typename... Axes>
int bins(const boost::variant<Axes...>& a) { return boost::apply_visitor(visitor::bins(), a); }

template <typename A>
int shape(const A& a) { return a.shape(); }

template <typename... Axes>
int shape(const boost::variant<Axes...>& a) { return boost::apply_visitor(visitor::shape(), a); }

}
}

#endif
