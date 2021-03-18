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

/** Thread-safe adaptor for builtin integral and floating point numbers.

  This adaptor uses std::atomic to make concurrent increments and additions safe for the
  stored value.

  On common computing platforms, the adapted integer has the same size and
  alignment as underlying type. The atomicity is implemented with a special CPU
  instruction. On exotic platforms the size of the adapted number may be larger and/or the
  type may have different alignment, which means it cannot be tightly packed into arrays.

  @tparam T type to adapt, must be supported by std::atomic.
 */
template <class T>
class thread_safe : public std::atomic<T> {
public:
  using value_type = T;
  using super_t = std::atomic<T>;

  thread_safe() noexcept : super_t(static_cast<T>(0)) {}
  // non-atomic copy and assign is allowed, because storage is locked in this case
  thread_safe(const thread_safe& o) noexcept : super_t(o.load()) {}
  thread_safe& operator=(const thread_safe& o) noexcept {
    super_t::store(o.load());
    return *this;
  }

  thread_safe(value_type arg) : super_t(arg) {}
  thread_safe& operator=(value_type arg) {
    super_t::store(arg);
    return *this;
  }

  thread_safe& operator+=(const thread_safe& arg) {
    operator+=(arg.load());
    return *this;
  }

  thread_safe& operator++() {
    increment_impl(detail::priority<1>{});
    return *this;
  }

  thread_safe& operator+=(value_type arg) {
    add_impl(detail::priority<1>{}, arg);
    return *this;
  }

  template <class Archive>
  void serialize(Archive& ar, unsigned /* version */) {
    auto value = super_t::load();
    ar& make_nvp("value", value);
    super_t::store(value);
  }

private:
  template <class U = value_type>
  auto increment_impl(detail::priority<1>) -> decltype(std::declval<std::atomic<U>>()++) {
    return super_t::operator++();
  }

  template <class U = value_type>
  auto add_impl(detail::priority<1>, value_type arg)
      -> decltype(std::declval<std::atomic<U>>().fetch_add(arg)) {
    return super_t::fetch_add(arg);
  }

  // support for floating point numbers in C++14
  template <class U = value_type>
  void increment_impl(detail::priority<0>) {
    operator+=(static_cast<value_type>(1));
  }

  // support for floating point numbers in C++14
  template <class U = value_type>
  void add_impl(detail::priority<0>, value_type arg) {
    value_type expected = super_t::load();
    value_type desired = expected + arg;
    while (!super_t::compare_exchange_weak(expected, desired)) {
      // someone else changed value, adapt desired result
      expected = super_t::load();
      desired = expected + arg;
    }
  }
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif
