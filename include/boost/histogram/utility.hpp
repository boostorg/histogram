#ifndef BOOST_HISTOGRAM_UTILITY_HPP_
#define BOOST_HISTOGRAM_UTILITY_HPP_

#include <boost/variant.hpp>
#include <boost/histogram/visitors.hpp>

namespace boost {
namespace histogram {

template <typename Storage1,
          typename Storage2,
          typename = typename Storage1::storage_tag,
          typename = typename Storage2::storage_tag>
bool operator==(const Storage1& s1, const Storage2& s2)
{
    if (s1.size() != s2.size())
        return false;
    for (std::size_t i = 0, n = s1.size(); i < n; ++i)
        if (s1.value(i) != s2.value(i) || s1.variance(i) != s2.variance(i))
            return false;
    return true;
}

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
