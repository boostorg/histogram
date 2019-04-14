// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_THREAD_SAFE_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_THREAD_SAFE_HPP

#include <atomic>
#include <boost/mp11/utility.hpp>
#include <type_traits>

namespace boost {
namespace histogram {
namespace accumulators {

template <class T>
class thread_safe : public std::atomic<T> {
public:
  using super_t = std::atomic<T>;

  thread_safe() noexcept : super_t(static_cast<T>(0)) {}
  // non-atomic copy and assign is allowed, because storage is locked in this case
  thread_safe(const thread_safe& o) noexcept : super_t(o.load()) {}
  thread_safe& operator=(const thread_safe& o) noexcept {
    super_t::store(o.load());
    return *this;
  }

  thread_safe(T arg) : super_t(arg) {}
  thread_safe& operator=(T arg) {
    super_t::store(arg);
    return *this;
  }

  void operator+=(T arg) { super_t::fetch_add(arg, std::memory_order_relaxed); }
  void operator++() { operator+=(static_cast<T>(1)); }
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif