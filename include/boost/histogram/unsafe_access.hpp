// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UNSAFE_ACCESS_HPP
#define BOOST_HISTOGRAM_UNSAFE_ACCESS_HPP

#include <boost/histogram/detail/meta.hpp>

namespace boost {
namespace histogram {

/// Unsafe read/write access to classes that potentially break consistency
struct unsafe_access {
  template <typename T>
  static typename T::axes_type& axes(T& t) {
    return t.axes_;
  }

  template <typename T>
  static const typename T::axes_type& axes(const T& t) {
    return t.axes_;
  }

  template <typename T>
  static typename T::storage_type& storage(T& t) {
    return t.storage_;
  }

  template <typename T>
  static const typename T::storage_type& storage(const T& t) {
    return t.storage_;
  }
};

} // namespace histogram
} // namespace boost

#endif
