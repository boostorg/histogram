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
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/weight.hpp>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

// forward declaration for optional boost.accumulators support
namespace boost {
namespace accumulators {
template <typename Sample, typename Features, typename Weight>
struct accumulator_set;
} // namespace accumulators
} // namespace boost

namespace boost {
namespace histogram {
namespace detail {

template <typename T>
struct is_accumulator_set : std::false_type {};

template <typename... Ts>
struct is_accumulator_set<::boost::accumulators::accumulator_set<Ts...>>
    : std::true_type {};

// specialized form for arithmetic types
struct element_adaptor_arithmetic {
  template <typename T>
  static void forward(T& t) {
    ++t;
  }
  template <typename T, typename U>
  static void forward(T& t, const weight_type<U>& u) {
    t += u.value;
  }
};

// specialized form for accumulator_set
struct element_adaptor_accumulator_set {
  template <typename T>
  static void forward(T& t) {
    t();
  }
  template <typename T, typename W>
  static void forward(T& t, const weight_type<W>& w) {
    t(weight = w.value);
  }
  template <typename T, typename U>
  static void forward(T& t, const U& u) {
    t(u);
  }
  template <typename T, typename U, typename W>
  static void forward(T& t, const weight_type<W>& w, const U& u) {
    t(u, weight = w.value);
  }
};

// generic form for aggregator types
struct element_adaptor_generic {
  template <typename T, typename... Us>
  static void forward(T& t, Us&&... us) {
    t(std::forward<Us>(us)...);
  }
};

template <typename T>
using element_adaptor =
    mp11::mp_if<std::is_arithmetic<T>, element_adaptor_arithmetic,
                mp11::mp_if<is_accumulator_set<T>, element_adaptor_accumulator_set,
                            element_adaptor_generic>>;

template <typename T>
struct ERROR_type_passed_to_storage_adaptor_not_recognized {};

template <typename T>
struct vector_augmentation : T {
  using value_type = typename T::value_type;

  vector_augmentation(T&& t) : T(std::move(t)) {}
  vector_augmentation(const T& t) : T(t) {}

  vector_augmentation() = default;
  vector_augmentation(vector_augmentation&&) = default;
  vector_augmentation(const vector_augmentation&) = default;
  vector_augmentation& operator=(vector_augmentation&&) = default;
  vector_augmentation& operator=(const vector_augmentation&) = default;

  void reset(std::size_t n) {
    this->resize(n);
    std::fill(this->begin(), this->end(), value_type());
  }

  template <typename U>
  void add(std::size_t i, U&& u) {
    T::operator[](i) += std::forward<U>(u);
  }

  template <typename U>
  void set_impl(std::size_t i, U&& u) {
    T::operator[](i) = std::forward<U>(u);
  }

  void mul_impl(std::size_t i, double x) { T::operator[](i) *= x; }
};

template <typename T>
struct array_augmentation : T {
  using value_type = typename T::value_type;

  array_augmentation(T&& t) : T(std::move(t)) {}
  array_augmentation(const T& t) : T(t) {}

  array_augmentation() = default;
  array_augmentation(array_augmentation&&) = default;
  array_augmentation(const array_augmentation&) = default;
  array_augmentation& operator=(array_augmentation&&) = default;
  array_augmentation& operator=(const array_augmentation&) = default;

  void reset(std::size_t n) {
    if (n > this->max_size()) // for std::array
      throw std::runtime_error(
          detail::cat("size ", n, " exceeds maximum capacity ", this->max_size()));
    std::fill_n(this->begin(), n, value_type());
    size_ = n;
  }

  template <typename U>
  void add(std::size_t i, U&& u) {
    T::operator[](i) += std::forward<U>(u);
  }

  std::size_t size() const { return size_; }

  template <typename U>
  void set_impl(std::size_t i, U&& u) {
    T::operator[](i) = std::forward<U>(u);
  }

  void mul_impl(std::size_t i, double x) { T::operator[](i) *= x; }

private:
  std::size_t size_ = 0;
};

template <typename T>
struct map_augmentation : T {
  static_assert(std::is_same<typename T::key_type, std::size_t>::value,
                "map must use std::size_t as key_type");
  using value_type = typename T::mapped_type;

  map_augmentation(T&& t) : T(std::move(t)) {}
  map_augmentation(const T& t) : T(t) {}

  map_augmentation() = default;
  map_augmentation(map_augmentation&&) = default;
  map_augmentation(const map_augmentation&) = default;
  map_augmentation& operator=(map_augmentation&&) = default;
  map_augmentation& operator=(const map_augmentation&) = default;

  void reset(std::size_t n) {
    this->clear();
    size_ = n;
  }

  value_type& operator[](std::size_t i) { return T::operator[](i); }

  value_type operator[](std::size_t i) const {
    auto it = this->find(i);
    return it == this->end() ? value_type() : it->second;
  }

  template <typename U>
  void add(std::size_t i, U&& u) {
    if (u == value_type()) return;
    auto it = this->find(i);
    if (it != this->end())
      it->second += std::forward<U>(u);
    else
      T::operator[](i) = std::forward<U>(u);
  }

  std::size_t size() const { return size_; }

  void mul_impl(std::size_t i, double x) {
    auto it = this->find(i);
    if (it != this->end()) it->second *= x;
  }

  template <typename U>
  void set_impl(std::size_t i, U&& u) {
    auto it = this->find(i);
    if (u == value_type()) {
      if (it != this->end()) this->erase(it);
    } else if (it != this->end())
      it->second = std::forward<U>(u);
    else
      T::operator[](i) = std::forward<U>(u);
  }

private:
  std::size_t size_ = 0;
};

template <typename T>
using storage_augmentation = mp11::mp_if<
    is_vector_like<T>, vector_augmentation<T>,
    mp11::mp_if<is_array_like<T>, array_augmentation<T>,
                mp11::mp_if<is_map_like<T>, map_augmentation<T>,
                            ERROR_type_passed_to_storage_adaptor_not_recognized<T>>>>;

} // namespace detail

/// generic implementation for std::array, vector-like, and map-like containers
template <typename T>
struct storage_adaptor : detail::storage_augmentation<T> {
  struct storage_tag {};
  using base_type = detail::storage_augmentation<T>;
  using value_type = typename base_type::value_type;
  using element_adaptor = detail::element_adaptor<value_type>;
  using const_reference = const value_type&;

  storage_adaptor() = default;
  storage_adaptor(const storage_adaptor&) = default;
  storage_adaptor& operator=(const storage_adaptor&) = default;
  storage_adaptor(storage_adaptor&&) = default;
  storage_adaptor& operator=(storage_adaptor&&) = default;

  storage_adaptor(T&& t) : base_type(std::move(t)) {}
  storage_adaptor(const T& t) : base_type(t) {}

  template <typename U, typename = detail::requires_storage<U>>
  storage_adaptor(const U& rhs) {
    (*this) = rhs;
  }

  template <typename U, typename = detail::requires_storage<U>>
  storage_adaptor& operator=(const U& rhs) {
    this->reset(rhs.size());
    for (std::size_t i = 0, n = this->size(); i < n; ++i) this->set_impl(i, rhs[i]);
    return *this;
  }

  template <typename... Us>
  void operator()(std::size_t i, Us&&... us) {
    BOOST_ASSERT(i < this->size());
    element_adaptor::forward((*this)[i], std::forward<Us>(us)...);
  }

  // precondition: storages have equal size
  template <typename U, typename = detail::requires_storage<U>>
  storage_adaptor& operator+=(const U& rhs) {
    const auto n = this->size();
    BOOST_ASSERT_MSG(n == rhs.size(), "sizes must be equal");
    for (std::size_t i = 0; i < n; ++i) this->add(i, rhs[i]);
    return *this;
  }

  storage_adaptor& operator*=(const double x) {
    for (std::size_t i = 0, n = this->size(); i < n; ++i) this->mul_impl(i, x);
    return *this;
  }

  storage_adaptor& operator/=(const double x) { return operator*=(1.0 / x); }

  template <typename U, typename = detail::requires_storage<U>>
  bool operator==(const U& u) const {
    const auto n = this->size();
    if (n != u.size()) return false;
    for (std::size_t i = 0; i < n; ++i)
      if ((*this)[i] != u[i]) return false;
    return true;
  }
};
} // namespace histogram
} // namespace boost

#endif