// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_STORAGE_ADAPTOR_HPP
#define BOOST_HISTOGRAM_STORAGE_ADAPTOR_HPP

#include <algorithm>
#include <array>
#include <boost/assert.hpp>
#include <boost/histogram/adaptive_storage.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {
template <typename T>
void increment_element_impl(std::true_type, T& t) {
  t();
}

template <typename T>
void increment_element_impl(std::false_type, T& t) {
  ++t;
}

template <typename T>
void increment_element(T& t) {
  increment_element_impl(is_callable<T>(), t);
}

template <typename HasResize, typename HasClear, typename T>
struct storage_reset_impl;

template <typename T>
struct storage_reset_impl<std::true_type, std::true_type, T> : T {
  using T::T;
  storage_reset_impl(const T& t) : T(t) {}
  storage_reset_impl(T&& t) : T(std::move(t)) {}

  void reset(T& t, std::size_t n) {
    t.resize(n);
    std::fill(t.begin(), t.end(), typename T::value_type());
  }
};

template <typename HasClear, typename T>
struct storage_reset_impl<std::false_type, HasClear, T> : T {
  using T::T;
  storage_reset_impl(const T& t) : T(t) {}
  storage_reset_impl(T&& t) : T(std::move(t)) {}

  void reset(T& t, std::size_t n) {
    if (n > this->max_size()) // for std::array
      throw std::runtime_error(
          detail::cat("size ", n, " exceeds maximum capacity ", t.max_size()));
    // map-like has clear(), std::array does not
    static_if<HasClear>(
        [](auto& t) { t.clear(); },
        [n](auto& t) { std::fill_n(t.begin(), n, typename T::value_type()); }, t);
    size_ = n;
  }

  std::size_t size_ = 0;
  std::size_t size() const { return size_; }
};

template <typename T>
using storage_reset = storage_reset_impl<has_method_resize<T>, has_method_clear<T>, T>;
} // namespace detail

/// generic implementation for std::array, vector-like, and map-like containers
template <typename T>
struct storage_adaptor : detail::storage_reset<T> {
  using base_type = detail::storage_reset<T>;
  using base_type::base_type;
  using value_type = typename T::value_type;
  using const_reference = typename T::const_reference;

  storage_adaptor() = default;
  storage_adaptor(const storage_adaptor&) = default;
  storage_adaptor& operator=(const storage_adaptor&) = default;
  storage_adaptor(storage_adaptor&&) = default;
  storage_adaptor& operator=(storage_adaptor&&) = default;

  storage_adaptor(const T& t) : base_type(t) {}
  storage_adaptor(T&& t) : base_type(std::move(t)) {}

  template <typename U>
  storage_adaptor(const storage_adaptor<U>& rhs) {
    reset(rhs.size());
    (*this) = rhs;
  }

  template <typename U>
  storage_adaptor& operator=(const storage_adaptor<U>& rhs) {
    reset(rhs.size());
    (*this) = rhs;
    return *this;
  }

  void reset(std::size_t n) { detail::storage_reset<T>::reset(*this, n); }

  void operator()(std::size_t i) {
    BOOST_ASSERT(i < this->size());
    detail::increment_element((*this)[i]);
  }

  template <typename U>
  void operator()(std::size_t i, U&& u) {
    BOOST_ASSERT(i < this->size());
    (*this)[i] += std::forward<U>(u);
  }

  // precondition: storages have equal size
  template <typename U>
  storage_adaptor& operator+=(const storage_adaptor<U>& u) {
    const auto n = this->size();
    BOOST_ASSERT_MSG(n == u.size(), "sizes must be equal");
    for (std::size_t i = 0; i < n; ++i) (*this)(i, u[i]);
    return *this;
  }

  storage_adaptor& operator*=(const double x) {
    for (std::size_t i = 0, n = this->size(); i < n; ++i) (*this)[i] *= x;
    return *this;
  }

  storage_adaptor& operator/=(const double x) { return operator*=(1.0 / x); }

  template <typename U>
  bool operator==(const storage_adaptor<U>& u) const {
    const auto n = this->size();
    if (n != u.size()) return false;
    for (std::size_t i = 0; i < n; ++i)
      if (!((*this)[i] == u[i])) return false;
    return true;
  }
};

template <typename A>
struct storage_adaptor<adaptive_storage<A>> : adaptive_storage<A> {
  using base_type = adaptive_storage<A>;
  using value_type = typename base_type::value_type;
  using const_reference = typename base_type::const_reference;
  using base_type::base_type;

  storage_adaptor(const adaptive_storage<A>& t) : base_type(t) {}
  storage_adaptor(adaptive_storage<A>&& t) : base_type(std::move(t)) {}
};
} // namespace histogram
} // namespace boost

#endif