// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGARM_AXIS_VISITOR_HPP_
#define _BOOST_HISTOGARM_AXIS_VISITOR_HPP_

#include <boost/variant/static_visitor.hpp>

namespace boost {
namespace histogram {
namespace detail {

  struct bins : public static_visitor<int>
  {
    template <typename A>
    int operator()(const A& a) const { return a.bins(); }
  };

  struct shape : public static_visitor<int>
  {
    template <typename A>
    int operator()(const A& a) const { return a.shape(); }
  };

  struct uoflow : public static_visitor<bool>
  {
    template <typename A>
    bool operator()(const A& a) const { return a.uoflow(); }
  };

  template <typename V>
  struct index : public static_visitor<int>
  {
    const V v;
    index(const V x) : v(x) {}
    template <typename A>
    int operator()(const A& a) const { return a.index(v); }
  };

  struct left : public static_visitor<double>
  {
    const int i;
    left(const int x) : i(x) {}
    template <typename A>
    double operator()(const A& a) const { return a.left(i); }
  };

  struct right : public static_visitor<double>
  {
    const int i;
    right(const int x) : i(x) {}
    template <typename A>
    double operator()(const A& a) const { return a.right(i); }
  };

  struct center : public static_visitor<double>
  {
    const int i;
    center(const int x) : i(x) {}
    template <typename A>
    double operator()(const A& a) const { return 0.5 * (a.left(i) + a.right(i)); }
  };

  struct cmp_axis : public static_visitor<bool>
  {
    template <typename T>
    bool operator()(const T& a, const T& b) const { return a == b; }

    template <typename T, typename U>
    bool operator()(const T&, const U&) const { return false; }
  };

  struct linearize : public static_visitor<void> {
    int j;
    std::size_t out = 0, stride = 1;

    void set(int in) { j = in; }

    template <typename A>
    void operator()(const A& a) {
      // the following is highly optimized code that runs in a hot loop;
      // please measure the performance impact of changes
      const int uoflow = a.uoflow();
      // set stride to zero if 'j' is not in range,
      // this communicates the out-of-range condition to the caller
      stride *= (j >= -uoflow) * (j < (a.bins() + uoflow));
      j += (j < 0) * (a.bins() + 2); // wrap around if in < 0
      out += j * stride;
      #pragma GCC diagnostic ignored "-Wstrict-overflow"
      stride *= a.shape();
    }
  };

  struct linearize_x : public static_visitor<void> {
    double x;
    std::size_t out = 0, stride = 1;

    void set(double in) { x = in; }

    template <typename A>
    void
    operator()(const A& a) {
      // j is guaranteed to be in range [-1, bins]
      int j = a.index(x);
      j += (j < 0) * (a.bins() + 2); // wrap around if j < 0
      out += j * stride;
      #pragma GCC diagnostic ignored "-Wstrict-overflow"
      stride *= (j < a.shape()) * a.shape(); // stride == 0 indicates out-of-range
    }
  };

}
}
}

#endif
