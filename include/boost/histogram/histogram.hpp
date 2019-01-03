// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_HISTOGRAM_HPP
#define BOOST_HISTOGRAM_HISTOGRAM_HPP

#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/common_type.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/histogram/grid.hpp>
#include <tuple>
#include <type_traits>

namespace boost {
namespace histogram {

template <class Axes, class Storage>
class histogram : public grid<Axes, Storage> {
  using grid_type = grid<Axes, Storage>;

public:
  histogram() = default;
  histogram(const histogram& rhs) = default;
  histogram(histogram&& rhs) = default;
  histogram& operator=(const histogram& rhs) = default;
  histogram& operator=(histogram&& rhs) = default;

  template <class A, class S>
  explicit histogram(const histogram<A, S>& rhs) : grid_type(rhs) {}

  template <class A, class = detail::requires_axes<A>>
  explicit histogram(A&& a)
      : grid_type(std::forward<A>(a), typename grid_type::storage_type()) {}

  template <class A, class S>
  histogram(A&& a, S&& s) : grid_type(std::forward<A>(a), std::forward<S>(s)) {}

  template <class A, class S>
  histogram& operator=(histogram<A, S>&& rhs) {
    grid_type::operator=(std::move(rhs));
    return *this;
  }

  template <class A, class S>
  histogram& operator=(const histogram<A, S>& rhs) {
    grid_type::operator=(rhs);
    return *this;
  }

  /// Add values of another histogram
  template <class A, class S>
  histogram& operator+=(const histogram<A, S>& rhs) {
    grid_type::operator+=(rhs);
    return *this;
  }

  /// Fill histogram with values and optional weight or sample
  template <class... Ts>
  void operator()(const Ts&... ts) {
    operator()(std::forward_as_tuple(ts...));
  }

  /// Fill histogram with value tuple and optional weight or sample
  template <class... Ts>
  void operator()(const std::tuple<Ts...>& t) {
    detail::fill_impl(grid_type::storage_, grid_type::axes_, t);
  }

  /// Access value at indices (const version)
  template <typename... Ts>
  decltype(auto) at(const Ts&... ts) const {
    return at(std::forward_as_tuple(ts...));
  }

  /// Access value at index tuple (const version)
  template <typename... Ts>
  decltype(auto) at(const std::tuple<Ts...>& t) const {
    return grid_type::at(t);
  }

  /// Access value at index iterable (const version)
  template <class Iterable, class = detail::requires_iterable<Iterable>>
  decltype(auto) at(const Iterable& c) const {
    return grid_type::at(c);
  }

  /// Access value at index (number for rank=1 or index tuple|iterable, const version)
  template <class T>
  decltype(auto) operator[](const T& t) const {
    return at(t);
  }

  auto begin() const noexcept { return grid_type::begin(); }
  auto end() const noexcept { return grid_type::end(); }
};

template <class A1, class S1, class A2, class S2>
auto operator+(const histogram<A1, S1>& a, const histogram<A2, S2>& b) {
  histogram<detail::common_axes<A1, A2>, detail::common_storage<S1, S2>> result = a;
  result += b;
  return result;
}

template <class A, class S>
auto operator+(histogram<A, S>&& a, const histogram<A, S>& b) {
  return a += b;
}

template <class A, class S>
auto operator+(const histogram<A, S>& a, histogram<A, S>&& b) {
  return b += a;
}

#if __cpp_deduction_guides >= 201606

template <class Axes>
histogram(Axes&& axes)->histogram<detail::unqual<Axes>, default_storage>;

template <class Axes, class Storage>
histogram(Axes&& axes, Storage&& storage)
    ->histogram<detail::unqual<Axes>, detail::unqual<Storage>>;

#endif

} // namespace histogram
} // namespace boost

#endif
