// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_HISTOGRAM_HPP
#define BOOST_HISTOGRAM_HISTOGRAM_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/histogram/arithmetic_operators.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/nodiscard.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mp11.hpp>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

template <typename Axes, typename Storage>
class BOOST_HISTOGRAM_DETAIL_NODISCARD histogram {
  static_assert(mp11::mp_size<Axes>::value > 0, "at least one axis required");

public:
  using axes_type = Axes;
  using storage_type = Storage;
  using value_type = typename storage_type::value_type;
  // typedefs for boost::range_iterator
  using iterator = decltype(std::declval<storage_type&>().begin());
  using const_iterator = decltype(std::declval<const storage_type&>().begin());

  histogram() = default;
  histogram(const histogram& rhs) = default;
  histogram(histogram&& rhs) = default;
  histogram& operator=(const histogram& rhs) = default;
  histogram& operator=(histogram&& rhs) = default;

  template <typename A, typename S>
  explicit histogram(const histogram<A, S>& rhs) : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename A, typename S>
  histogram& operator=(const histogram<A, S>& rhs) {
    storage_ = rhs.storage_;
    detail::axes_assign(axes_, rhs.axes_);
    return *this;
  }

  explicit histogram(const axes_type& a, storage_type s = {})
      : axes_(a), storage_(std::move(s)) {
    storage_.reset(detail::bincount(axes_));
  }

  explicit histogram(axes_type&& a, storage_type s = {})
      : axes_(std::move(a)), storage_(std::move(s)) {
    storage_.reset(detail::bincount(axes_));
  }

  template <typename A, typename S>
  bool operator==(const histogram<A, S>& rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename A, typename S>
  bool operator!=(const histogram<A, S>& rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename A, typename S>
  histogram& operator+=(const histogram<A, S>& rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::invalid_argument("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  histogram& operator*=(const double x) {
    storage_ *= x;
    return *this;
  }

  histogram& operator/=(const double x) {
    storage_ *= 1.0 / x;
    return *this;
  }

  /// Number of axes (dimensions) of histogram
  std::size_t rank() const noexcept { return detail::axes_size(axes_); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const noexcept { return storage_.size(); }

  /// Reset bin counters to zero
  void reset() { storage_.reset(storage_.size()); }

  /// Get N-th axis (const version)
  template <std::size_t N>
  decltype(auto) axis(mp11::mp_size_t<N>) const {
    detail::rank_check(axes_, N);
    return detail::axis_get<N>(axes_);
  }

  /// Get N-th axis
  template <std::size_t N>
  decltype(auto) axis(mp11::mp_size_t<N>) {
    detail::rank_check(axes_, N);
    return detail::axis_get<N>(axes_);
  }

  /// Get first axis (convenience for 1-d histograms, const version)
  decltype(auto) axis() const { return axis(mp11::mp_size_t<0>()); }

  /// Get first axis (convenience for 1-d histograms)
  decltype(auto) axis() { return axis(mp11::mp_size_t<0>()); }

  /// Get N-th axis with runtime index (const version)
  decltype(auto) axis(std::size_t i) const {
    detail::rank_check(axes_, i);
    return detail::axis_get(axes_, i);
  }

  /// Get N-th axis with runtime index
  decltype(auto) axis(std::size_t i) {
    detail::rank_check(axes_, i);
    return detail::axis_get(axes_, i);
  }

  /// Apply unary functor/function to each axis
  template <typename Unary>
  void for_each_axis(Unary&& unary) const {
    detail::for_each_axis(axes_, std::forward<Unary>(unary));
  }

  /// Fill histogram with values and optional weight or sample
  template <typename... Ts>
  void operator()(const Ts&... ts) {
    operator()(std::forward_as_tuple(ts...));
  }

  /// Fill histogram with value tuple and optional weight or sample
  template <typename... Ts>
  void operator()(const std::tuple<Ts...>& t) {
    detail::fill_impl(storage_, axes_, t);
  }

  /// Access bin counter at indices
  template <typename... Ts>
  decltype(auto) at(const Ts&... ts) const {
    return at(std::forward_as_tuple(ts...));
  }

  /// Access bin counter at index tuple
  template <typename... Ts>
  decltype(auto) at(const std::tuple<Ts...>& t) const {
    const auto idx = detail::at_impl(axes_, t);
    if (!idx) throw std::out_of_range("indices out of bounds");
    return storage_[*idx];
  }

  /// Access bin counter at index (for 1D histograms and passing index tuple)
  template <typename T>
  decltype(auto) operator[](const T& t) const {
    return at(t);
  }

  const_iterator begin() const noexcept { return storage_.begin(); }
  const_iterator end() const noexcept { return storage_.end(); }

  template <typename Archive>
  void serialize(Archive&, unsigned);

private:
  axes_type axes_;
  storage_type storage_;

  template <typename A, typename S>
  friend class histogram;
  friend struct unsafe_access;
};

} // namespace histogram
} // namespace boost

#endif
