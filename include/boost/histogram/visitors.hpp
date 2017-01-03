// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGARM_VISITORS_HPP_
#define _BOOST_HISTOGARM_VISITORS_HPP_

#include <boost/variant/static_visitor.hpp>

namespace boost {
namespace histogram {
namespace visitor {

  struct bins : public static_visitor<int>
  {
    template <typename A>
    int operator()(const A& a) const
    { return a.bins(); }
  };

  struct shape : public static_visitor<int>
  {
    template <typename A>
    int operator()(const A& a) const
    { return a.shape(); }
  };

  struct uoflow : public static_visitor<bool>
  {
    template <typename A>
    bool operator()(const A& a) const { return a.uoflow(); }
  };

  struct index : public static_visitor<int>
  {
    template <typename A, typename V>
    int operator()(const A& a, const V v) const { return a.index(v); }
  };

  struct cmp : public static_visitor<bool>
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
