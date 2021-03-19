// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_THREAD_SAFE_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_THREAD_SAFE_HPP

#include <atomic>
#include <boost/core/nvp.hpp>
#include <boost/histogram/detail/priority.hpp>
#include <boost/mp11/utility.hpp>
#include <type_traits>

namespace boost {
namespace histogram {
namespace accumulators {

/** Thread-safe adaptor for integral and floating point numbers.

  This adaptor uses atomic operations to make concurrent increments and additions safe for
  the stored value.

  On common computing platforms, the adapted integer has the same size and
  alignment as underlying type. The atomicity is implemented with a special CPU
  instruction. On exotic platforms the size of the adapted number may be larger and/or the
  type may have different alignment, which means it cannot be tightly packed into arrays.

  This implementation uses a workaround to support atomic operations on floating point
  numbers in C++14. Compiling with C++20 may increase performance for
  operations on floating point numbers. The implementation automatically uses the best
  implementation that is available.

  @tparam T type to adapt; must be arithmetic (integer or floating point).
 */
template <class T>
class thread_safe {
public:
  static_assert(std::is_arithmetic<T>(), "");

  using value_type = T;
  using const_reference = const T&;
  using atomic_value_type = std::atomic<T>;

  thread_safe() noexcept : value_{static_cast<value_type>(0)} {}
  // non-atomic copy and assign is allowed, because storage is locked in this case
  thread_safe(const thread_safe& o) noexcept : thread_safe{o.value()} {}
  thread_safe& operator=(const thread_safe& o) noexcept {
    value_.store(o.value());
    return *this;
  }

  thread_safe(value_type arg) : value_{arg} {}
  thread_safe& operator=(value_type arg) noexcept {
    value_.store(arg);
    return *this;
  }

  /// Increment value by one.
  thread_safe& operator++() {
    increment_impl(detail::priority<1>{});
    return *this;
  }

  /// Increment value by argument.
  thread_safe& operator+=(value_type arg) {
    add_impl(detail::priority<1>{}, arg);
    return *this;
  }

  /// Add another thread_safe.
  thread_safe& operator+=(const thread_safe& arg) {
    add_impl(detail::priority<1>{}, static_cast<value_type>(arg));
    return *this;
  }

  /// Scale by value
  thread_safe& operator*=(const_reference value) noexcept {
    value_ *= value;
    return *this;
  }

  bool operator==(const thread_safe& rhs) const noexcept { return value_ == rhs.value_; }

  bool operator!=(const thread_safe& rhs) const noexcept { return !operator==(rhs); }

  /// Return value.
  value_type value() const noexcept { return value_.load(); }

  // conversion to value_type should be explicit
  explicit operator value_type() const noexcept { return value_.load(); }

  // allow implicit conversion to other thread_safe
  template <class U>
  operator thread_safe<U>() const noexcept {
    return static_cast<U>(value_.value());
  }

  template <class Archive>
  void serialize(Archive& ar, unsigned /* version */) {
    auto value = value_.load();
    ar& make_nvp("value", value);
    value_.store(value);
  }

private:
  template <class U = value_type>
  auto increment_impl(detail::priority<1>) -> decltype(std::declval<std::atomic<U>>()++) {
    return value_.operator++();
  }

  template <class U = value_type>
  auto add_impl(detail::priority<1>, value_type arg)
      -> decltype(std::declval<std::atomic<U>>().fetch_add(arg)) {
    return value_.fetch_add(arg);
  }

  // workaround for floating point numbers in C++14, obsolete in C++20
  template <class U = value_type>
  void increment_impl(detail::priority<0>) {
    operator+=(static_cast<value_type>(1));
  }

  // workaround for floating point numbers in C++14, obsolete in C++20
  template <class U = value_type>
  void add_impl(detail::priority<0>, value_type arg) {
    value_type expected = value_.load();
    value_type desired = expected + arg;
    while (!value_.compare_exchange_weak(expected, desired)) {
      // someone else changed value, adapt desired result
      expected = value_.load();
      desired = expected + arg;
    }
  }

  atomic_value_type value_;
};

template <class T, class U>
std::enable_if_t<std::is_arithmetic<T>::value, bool> operator==(
    const T& t, const thread_safe<U>& rhs) noexcept {
  return rhs == t;
}

template <class T, class U>
std::enable_if_t<std::is_arithmetic<T>::value, bool> operator!=(
    const T& t, const thread_safe<U>& rhs) noexcept {
  return rhs != t;
}

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif
