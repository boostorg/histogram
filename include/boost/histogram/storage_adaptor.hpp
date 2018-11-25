// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_STORAGE_ADAPTOR_HPP
#define BOOST_HISTOGRAM_STORAGE_ADAPTOR_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/weight.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <stdexcept>
#include <type_traits>

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

struct element_adaptor_incrementable {
  template <typename T>
  static void forward(T& t) {
    ++t;
  }
  template <typename T, typename U>
  static void forward(T& t, const weight_type<U>& u) {
    t += u.value;
  }
};

struct element_adaptor_generic {
  template <typename T, typename... Us>
  static void forward(T& t, Us&&... us) {
    t(std::forward<Us>(us)...);
  }

  template <typename T, typename U, typename... Us>
  static void forward(T& t, const weight_type<U>& u, Us&&... us) {
    t(u.value, std::forward<Us>(us)...);
  }
};

template <typename T>
using element_adaptor =
    mp11::mp_if<is_accumulator_set<T>, element_adaptor_accumulator_set,
                // is_incrementable is used instead of std::is_arithmetic, which also
                // works with wrapped integers like copyable_atomic<int>
                mp11::mp_if<detail::is_incrementable<T>, element_adaptor_incrementable,
                            element_adaptor_generic>>;

template <typename T>
struct ERROR_type_passed_to_storage_adaptor_not_recognized {};

template <typename C>
struct vector_impl {
  using value_type = typename C::value_type;

  vector_impl(C&& c) : container_(std::move(c)) {}
  vector_impl(const C& c) : container_(c) {}

  vector_impl() = default;
  vector_impl(vector_impl&&) = default;
  vector_impl(const vector_impl&) = default;
  vector_impl& operator=(vector_impl&&) = default;
  vector_impl& operator=(const vector_impl&) = default;

  void reset(std::size_t n) {
    container_.resize(n);
    std::fill(container_.begin(), container_.end(), value_type());
  }

  const value_type& operator[](std::size_t i) const noexcept { return container_[i]; }

  template <typename U>
  void add(std::size_t i, U&& u) {
    container_[i] += std::forward<U>(u);
  }
  template <typename U>
  void set(std::size_t i, U&& u) {
    container_[i] = std::forward<U>(u);
  }

  std::size_t size() const noexcept { return container_.size(); }

  decltype(auto) begin() const noexcept { return container_.begin(); }
  decltype(auto) end() const noexcept { return container_.end(); }

  decltype(auto) get_allocator() const noexcept { return container_.get_allocator(); }

  value_type& ref(std::size_t i) noexcept { return container_[i]; }
  void mul(std::size_t i, double x) { container_[i] *= x; }

  C container_;
};

template <typename C>
struct array_impl {
  using value_type = typename C::value_type;

  array_impl(C&& c) : container_(std::move(c)) {}
  array_impl(const C& c) : container_(c) {}

  array_impl() = default;
  array_impl(array_impl&&) = default;
  array_impl(const array_impl&) = default;
  array_impl& operator=(array_impl&&) = default;
  array_impl& operator=(const array_impl&) = default;

  void reset(std::size_t n) {
    if (n > container_.max_size()) // for std::array
      throw std::runtime_error(
          detail::cat("size ", n, " exceeds maximum capacity ", container_.max_size()));
    std::fill_n(container_.begin(), n, value_type());
    size_ = n;
  }

  const value_type& operator[](std::size_t i) const noexcept { return container_[i]; }

  template <typename U>
  void add(std::size_t i, U&& u) {
    container_[i] += std::forward<U>(u);
  }
  template <typename U>
  void set(std::size_t i, U&& u) {
    container_[i] = std::forward<U>(u);
  }

  std::size_t size() const { return size_; }

  decltype(auto) begin() const noexcept { return container_.begin(); }
  decltype(auto) end() const noexcept { return container_.begin() + size_; }

  value_type& ref(std::size_t i) { return container_[i]; }
  void mul(std::size_t i, double x) { container_[i] *= x; }

  C container_;
  std::size_t size_ = 0;
};

template <typename C>
struct map_impl {
  static_assert(std::is_same<typename C::key_type, std::size_t>::value,
                "map must use std::size_t as key_type");

  using value_type = typename C::mapped_type;

  class const_iterator : public boost::iterator_facade<const_iterator, value_type,
                                                       boost::random_access_traversal_tag,
                                                       const value_type&> {
  public:
    const_iterator(const map_impl& parent, std::size_t idx) noexcept
        : parent_(parent), idx_(idx) {}

  protected:
    void increment() noexcept { ++idx_; }
    void decrement() noexcept { --idx_; }
    void advance(std::ptrdiff_t n) noexcept { idx_ += n; }
    std::ptrdiff_t distance_to(const_iterator rhs) const noexcept {
      return rhs.idx_ - idx_;
    }
    bool equal(const_iterator rhs) const noexcept {
      return &parent_ == &rhs.parent_ && idx_ == rhs.idx_;
    }
    const value_type& dereference() const { return parent_[idx_]; }

    friend class ::boost::iterator_core_access;

  private:
    const map_impl& parent_;
    std::size_t idx_;
  };

  map_impl(C&& c) : container_(std::move(c)) {}
  map_impl(const C& c) : container_(c) {}

  map_impl() = default;
  map_impl(map_impl&&) = default;
  map_impl(const map_impl&) = default;
  map_impl& operator=(map_impl&&) = default;
  map_impl& operator=(const map_impl&) = default;

  void reset(std::size_t n) {
    container_.clear();
    size_ = n;
  }

  const value_type& operator[](std::size_t i) const noexcept {
    auto it = container_.find(i);
    static auto null = value_type();
    return it == container_.end() ? null : it->second;
  }

  template <typename U>
  void add(std::size_t i, U&& u) {
    if (u == value_type()) return;
    auto it = container_.find(i);
    if (it != container_.end())
      it->second += std::forward<U>(u);
    else
      container_[i] = std::forward<U>(u);
  }

  template <typename U>
  void set(std::size_t i, U&& u) {
    auto it = container_.find(i);
    if (u == value_type()) {
      if (it != container_.end()) container_.erase(it);
    } else {
      if (it != container_.end())
        it->second = std::forward<U>(u);
      else
        container_[i] = std::forward<U>(u);
    }
  }

  std::size_t size() const noexcept { return size_; }

  const_iterator begin() const noexcept { return {*this, 0}; }
  const_iterator end() const noexcept { return {*this, size_}; }

  decltype(auto) get_allocator() const noexcept { return container_.get_allocator(); }

  value_type& ref(std::size_t i) { return container_[i]; }
  void mul(std::size_t i, double x) {
    auto it = container_.find(i);
    if (it != container_.end()) it->second *= x;
  }

  C container_;
  std::size_t size_ = 0;
};

template <typename T>
using storage_adaptor_impl = mp11::mp_if<
    is_vector_like<T>, vector_impl<T>,
    mp11::mp_if<is_array_like<T>, array_impl<T>,
                mp11::mp_if<is_map_like<T>, map_impl<T>,
                            ERROR_type_passed_to_storage_adaptor_not_recognized<T>>>>;

} // namespace detail

/// generic implementation for std::array, vector-like, and map-like containers
template <typename T>
struct storage_adaptor : detail::storage_adaptor_impl<T> {
  struct storage_tag {};
  using base_type = detail::storage_adaptor_impl<T>;
  using value_type = typename base_type::value_type;
  using element_adaptor = detail::element_adaptor<value_type>;

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
    for (std::size_t i = 0, n = this->size(); i < n; ++i) this->set(i, rhs[i]);
    return *this;
  }

  template <typename... Us>
  void operator()(std::size_t i, Us&&... us) {
    BOOST_ASSERT(i < this->size());
    element_adaptor::forward(this->ref(i), std::forward<Us>(us)...);
  }

  // precondition: storages have equal size
  template <typename U, typename = detail::requires_storage<U>>
  storage_adaptor& operator+=(const U& rhs) {
    const auto n = this->size();
    BOOST_ASSERT(n == rhs.size());
    for (std::size_t i = 0; i < n; ++i) this->add(i, rhs[i]);
    return *this;
  }

  storage_adaptor& operator*=(const double x) {
    for (std::size_t i = 0, n = this->size(); i < n; ++i) this->mul(i, x);
    return *this;
  }

  storage_adaptor& operator/=(const double x) { return operator*=(1.0 / x); }

  template <typename U>
  bool operator==(const U& u) const noexcept {
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