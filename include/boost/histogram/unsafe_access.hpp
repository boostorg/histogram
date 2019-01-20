// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UNSAFE_ACCESS_HPP
#define BOOST_HISTOGRAM_UNSAFE_ACCESS_HPP

#include <boost/histogram/detail/axes.hpp>
#include <type_traits>

namespace boost {
namespace histogram {

/// Unsafe read/write access to classes that potentially break consistency
struct unsafe_access {
  /// Get axes.
  //@{
  template <class T>
  static auto& axes(T& t) {
    return t.axes_;
  }

  template <class T>
  static const auto& axes(const T& t) {
    return t.axes_;
  }
  //@}

  /// Get mutable axis reference with compile-time or run-time number.
  //@{
  template <class T>
  static decltype(auto) axis(T& t, unsigned i) {
    detail::axis_index_is_valid(t.axes_, i);
    return detail::axis_get(t.axes_, i);
  }

  template <class T, unsigned N = 0>
  static decltype(auto) axis(T& t, std::integral_constant<unsigned, N> = {}) {
    detail::axis_index_is_valid(t.axes_, N);
    return detail::axis_get<N>(t.axes_);
  }
  //@}

  /// Get storage.
  //@{
  template <class T>
  static auto& storage(T& t) {
    return t.storage_;
  }

  template <class T>
  static const auto& storage(const T& t) {
    return t.storage_;
  }
  //@}
};

} // namespace histogram
} // namespace boost

#endif
