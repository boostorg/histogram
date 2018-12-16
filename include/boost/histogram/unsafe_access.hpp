// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UNSAFE_ACCESS_HPP
#define BOOST_HISTOGRAM_UNSAFE_ACCESS_HPP

#include <boost/histogram/detail/axes.hpp>
#include <initializer_list>

namespace boost {
namespace histogram {

/// Unsafe read/write access to classes that potentially break consistency
struct unsafe_access {
  /// Get axes
  template <typename T>
  static typename T::axes_type& axes(T& t) {
    return t.axes_;
  }

  /// Get axes (const version)
  template <typename T>
  static const typename T::axes_type& axes(const T& t) {
    return t.axes_;
  }

  /// Get storage
  template <typename T>
  static typename T::storage_type& storage(T& t) {
    return t.storage_;
  }

  /// Get storage (const version)
  template <typename T>
  static const typename T::storage_type& storage(const T& t) {
    return t.storage_;
  }

  /// Set histogram value at index
  template <typename Histogram, typename Iterable, typename Value,
            typename = detail::requires_iterable<Iterable>>
  static void set_value(Histogram& h, const Iterable& c, Value&& v) {
    const auto idx = detail::at_impl(h.axes_, c);
    if (!idx) BOOST_THROW_EXCEPTION(std::out_of_range("indices out of bounds"));
    h.storage_.set(*idx, std::forward<Value>(v));
  }

  /// Set histogram value at index
  template <typename Histogram, typename T, typename Value>
  static void set_value(Histogram& h, std::initializer_list<T>&& c, Value&& v) {
    set_value(h, c, std::forward<Value>(v));
  }

  /// Add value to histogram cell at index
  template <typename Histogram, typename Iterable, typename Value,
            typename = detail::requires_iterable<Iterable>>
  static void add_value(Histogram& h, const Iterable& c, Value&& v) {
    const auto idx = detail::at_impl(h.axes_, c);
    if (!idx) BOOST_THROW_EXCEPTION(std::out_of_range("indices out of bounds"));
    h.storage_.add(*idx, std::forward<Value>(v));
  }

  /// Add value to histogram cell at index
  template <typename Histogram, typename T, typename Value>
  static void add_value(Histogram& h, std::initializer_list<T>&& c, Value&& v) {
    add_value(h, c, std::forward<Value>(v));
  }
};

} // namespace histogram
} // namespace boost

#endif
