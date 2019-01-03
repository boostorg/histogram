// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_GRID_HPP
#define BOOST_HISTOGRAM_GRID_HPP

#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

template <class Axes, class Storage>
class grid {
  static_assert(mp11::mp_size<Axes>::value > 0, "at least one axis required");

public:
  using axes_type = Axes;
  using storage_type = Storage;
  using value_type = typename storage_type::value_type;
  // typedefs for boost::range_iterator
  using iterator = decltype(std::declval<storage_type&>().begin());
  using const_iterator = decltype(std::declval<const storage_type&>().begin());

  grid() = default;
  grid(const grid& rhs) = default;
  grid(grid&& rhs) = default;
  grid& operator=(const grid& rhs) = default;
  grid& operator=(grid&& rhs) = default;

  template <class A, class S>
  explicit grid(const grid<A, S>& rhs) : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <class A, class S>
  grid& operator=(const grid<A, S>& rhs) {
    storage_ = rhs.storage_;
    detail::axes_assign(axes_, rhs.axes_);
    return *this;
  }

  template <class A, class S>
  explicit grid(A&& a, S&& s) : axes_(std::forward<A>(a)), storage_(std::forward<S>(s)) {
    storage_.reset(detail::bincount(axes_));
  }

  template <class A, class S>
  bool operator==(const grid<A, S>& rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <class A, class S>
  bool operator!=(const grid<A, S>& rhs) const noexcept {
    return !operator==(rhs);
  }

  template <class A, class S>
  grid& operator+=(const grid<A, S>& rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      BOOST_THROW_EXCEPTION(std::invalid_argument("axes of histograms differ"));
    storage_ += rhs.storage_;
    return *this;
  }

  grid& operator*=(const double x) {
    storage_ *= x;
    return *this;
  }

  grid& operator/=(const double x) {
    storage_ *= 1.0 / x;
    return *this;
  }

  /// Number of axes (dimensions)
  unsigned rank() const noexcept { return detail::axes_size(axes_); }

  /// Total number of bins (including underflow/overflow)
  std::size_t size() const noexcept { return storage_.size(); }

  /// Reset values to zero
  void reset() { storage_.reset(storage_.size()); }

  /// Get N-th axis
  template <unsigned N>
  decltype(auto) axis(std::integral_constant<unsigned, N>) {
    detail::rank_check(axes_, N);
    return detail::axis_get<N>(axes_);
  }

  /// Get N-th axis (const version)
  template <unsigned N>
  decltype(auto) axis(std::integral_constant<unsigned, N>) const {
    detail::rank_check(axes_, N);
    return detail::axis_get<N>(axes_);
  }

  /// Get first axis (convenience for 1-d histograms)
  decltype(auto) axis() { return axis(std::integral_constant<unsigned, 0>()); }

  /// Get first axis (convenience for 1-d histograms, const version)
  decltype(auto) axis() const { return axis(std::integral_constant<unsigned, 0>()); }

  /// Get N-th axis with runtime index
  decltype(auto) axis(unsigned i) {
    detail::rank_check(axes_, i);
    return detail::axis_get(axes_, i);
  }

  /// Get N-th axis with runtime index (const version)
  decltype(auto) axis(unsigned i) const {
    detail::rank_check(axes_, i);
    return detail::axis_get(axes_, i);
  }

  /// Apply unary functor/function to each axis
  template <class Unary>
  void for_each_axis(Unary&& unary) const {
    detail::for_each_axis(axes_, std::forward<Unary>(unary));
  }

  /// Access value at indices
  template <typename... Ts>
  decltype(auto) at(const Ts&... ts) {
    return at(std::forward_as_tuple(ts...));
  }

  /// Access value at indices (const version)
  template <typename... Ts>
  decltype(auto) at(const Ts&... ts) const {
    return at(std::forward_as_tuple(ts...));
  }

  /// Access value at index tuple
  template <typename... Ts>
  decltype(auto) at(const std::tuple<Ts...>& t) {
    const auto idx = detail::at_impl(axes_, t);
    if (!idx) BOOST_THROW_EXCEPTION(std::out_of_range("indices out of bounds"));
    return storage_[*idx];
  }

  /// Access value at index tuple (const version)
  template <typename... Ts>
  decltype(auto) at(const std::tuple<Ts...>& t) const {
    const auto idx = detail::at_impl(axes_, t);
    if (!idx) BOOST_THROW_EXCEPTION(std::out_of_range("indices out of bounds"));
    return storage_[*idx];
  }

  /// Access value at index iterable
  template <class Iterable, class = detail::requires_iterable<Iterable>>
  decltype(auto) at(const Iterable& c) {
    const auto idx = detail::at_impl(axes_, c);
    if (!idx) BOOST_THROW_EXCEPTION(std::out_of_range("indices out of bounds"));
    return storage_[*idx];
  }

  /// Access value at index iterable (const version)
  template <class Iterable, class = detail::requires_iterable<Iterable>>
  decltype(auto) at(const Iterable& c) const {
    const auto idx = detail::at_impl(axes_, c);
    if (!idx) BOOST_THROW_EXCEPTION(std::out_of_range("indices out of bounds"));
    return storage_[*idx];
  }

  /// Access value at index (number for rank=1 or index tuple|iterable)
  template <class T>
  decltype(auto) operator[](const T& t) {
    return at(t);
  }

  /// Access value at index (number for rank=1 or index tuple|iterable, const version)
  template <class T>
  decltype(auto) operator[](const T& t) const {
    return at(t);
  }

  // iterator begin() noexcept { return storage_.begin(); }
  // iterator end() noexcept { return storage_.end(); }

  const_iterator begin() const noexcept { return storage_.begin(); }
  const_iterator end() const noexcept { return storage_.end(); }

protected:
  axes_type axes_;
  storage_type storage_;

  template <class A, class S>
  friend class grid;
  friend struct unsafe_access;
};

template <class A, class S>
auto operator*(grid<A, S>&& g, double x) {
  return g *= x;
}

template <class A, class S>
auto operator*(double x, grid<A, S>&& g) {
  return g *= x;
}

template <class A, class S>
auto operator*(const grid<A, S>& g, double x) {
  auto copy = g;
  return copy *= x;
}

template <class A, class S>
auto operator*(double x, const grid<A, S>& g) {
  auto copy = g;
  return copy *= x;
}

template <class A, class S>
auto operator/(grid<A, S>&& g, double x) {
  return g *= 1.0 / x;
}

template <class A, class S>
auto operator/(const grid<A, S>& g, double x) {
  auto copy = g;
  return copy *= 1.0 / x;
}

template <class A1, class S1, class A2, class S2>
auto operator+(const grid<A1, S1>& a, const grid<A2, S2>& b) {
  auto r = grid<detail::common_axes<A1, A2>, detail::common_storage<S1, S2>>(a);
  return r += b;
}

#if __cpp_deduction_guides >= 201606

template <class Axes, class Storage>
grid(Axes&& axes, Storage&& storage)->grid<detail::unqual<Axes>, detail::unqual<Storage>>;

#endif

} // namespace histogram
} // namespace boost

#endif
