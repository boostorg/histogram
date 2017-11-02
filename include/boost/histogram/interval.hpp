// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_INTERVAL_HPP_
#define _BOOST_HISTOGRAM_INTERVAL_HPP_

#include <utility>

namespace boost { namespace histogram {

template <typename T>
class interval {
public:
  interval() = default;
  interval(const interval&) = default;
  interval& operator=(const interval&) = default;
  interval(interval&&) = default;
  interval& operator=(interval&&) = default;

  template <typename U>
  interval(U&& x, U&& y) : a(std::forward<U>(x)), b(std::forward<U>(y)) {}

  template <typename U>
  interval(const interval<U>& i) : a(i.lower()), b(i.upper()) {}

  template <typename U>
  interval(interval<U>&& i) : a(i.lower()), b(i.upper()) {}

  const T& lower() const noexcept { return a; }
  const T& upper() const noexcept { return b; }

  bool operator==(const interval& i) const noexcept {
    return a == i.a && b == i.b;
  }
  bool operator!=(const interval& i) const noexcept {
    return !operator==(i);
  }
private:
  T a, b;
};

}}

#endif
