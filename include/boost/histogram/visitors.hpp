#ifndef _BOOST_HISTOGARM_VISITORS_HPP_
#define _BOOST_HISTOGARM_VISITORS_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/variant/static_visitor.hpp>

namespace boost {
namespace histogram {
namespace detail {

  struct bins_visitor : public static_visitor<unsigned>
  {
    template <typename A>
    unsigned operator()(const A& a) const { return a.bins(); }
  };

  struct fields_visitor : public static_visitor<unsigned> 
  {
    unsigned operator()(const category_axis& a) const { return a.bins(); }

    template <typename A>
    unsigned operator()(const A& a) const { return a.bins() + 2 * a.uoflow(); }
  };

  struct uoflow_visitor : public static_visitor<bool> 
  {
    bool operator()(const category_axis& a) const { return false; }

    template <typename A>
    bool operator()(const A& a) const { return a.uoflow(); }
  };

  struct index_visitor : public static_visitor<int>
  {
    double x;
    index_visitor(double d) : x(d) {}

    int operator()(const category_axis& a) const { return int(x + 0.5); }

    template <typename A>
    int operator()(const A& a) const { return a.index(x); }
  };

  struct cmp_visitor : public static_visitor<bool>
  {
    template <typename T>
    bool operator()(const T& a, const T& b) const
    { return a == b; }

    template <typename T, typename U>
    bool operator()(const T&, const U&) const { return false; }
  };
}
}
}

#endif
