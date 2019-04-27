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

/** Unsafe read/write access to private data that potentially breaks consistency.

  This struct enables access to private data of some classes. It is intended for library
  developers who need this to implement algorithms efficiently, for example,
  serialization. Users should not use this. If you are a user who absolutely needs this to
  get a specific effect, please submit an issue on Github. This means that the public
  interface is insufficient.
*/
struct unsafe_access {
  /**
    Get axes.
    @param hist histogram.
  */
  template <class Histogram>
  static auto& axes(Histogram& hist) {
    return hist.axes_;
  }

  /// @copydoc axes()
  template <class Histogram>
  static const auto& axes(const Histogram& hist) {
    return hist.axes_;
  }

  /**
    Get mutable axis reference with compile-time number.
    @param hist histogram.
    @tparam I axis index (optional, default: 0).
  */
  template <class Histogram, unsigned I = 0>
  static decltype(auto) axis(Histogram& hist, std::integral_constant<unsigned, I> = {}) {
    detail::axis_index_is_valid(hist.axes_, I);
    return detail::axis_get<I>(hist.axes_);
  }

  /**
    Get mutable axis reference with run-time number.
    @param hist histogram.
    @param i axis index.
  */
  template <class Histogram>
  static decltype(auto) axis(Histogram& hist, unsigned i) {
    detail::axis_index_is_valid(hist.axes_, i);
    return detail::axis_get(hist.axes_, i);
  }

  /**
    Get storage.
    @param hist histogram.
  */
  template <class Histogram>
  static auto& storage(Histogram& hist) {
    return hist.storage_and_mutex_.first();
  }

  /// @copydoc storage()
  template <class Histogram>
  static const auto& storage(const Histogram& hist) {
    return hist.storage_and_mutex_.first();
  }

  /**
    Get buffer of unlimited_storage.
    @param storage instance of unlimited_storage.
  */
  template <class Allocator>
  static constexpr auto& unlimited_storage_buffer(unlimited_storage<Allocator>& storage) {
    return storage.buffer_;
  }
};

} // namespace histogram
} // namespace boost

#endif
