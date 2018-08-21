// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_HPP_
#define BOOST_HISTOGRAM_TEST_UTILITY_HPP_

#include <boost/core/typeinfo.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/tuple.hpp>
#include <iostream>
#include <numeric>
#include <ostream>
#include <tuple>
#include <vector>

using i0 = boost::mp11::mp_size_t<0>;
using i1 = boost::mp11::mp_size_t<1>;
using i2 = boost::mp11::mp_size_t<2>;
using i3 = boost::mp11::mp_size_t<3>;

namespace std { // never add to std, we only do it to get ADL working
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  os << "[ ";
  for (const auto& x : v) os << x << " ";
  os << "]";
  return os;
}

struct ostreamer {
  ostream& os;
  template <typename T>
  void operator()(const T& t) const {
    os << t << " ";
  }
};

template <typename... Ts>
ostream& operator<<(ostream& os, const tuple<Ts...>& t) {
  os << "[ ";
  ::boost::mp11::tuple_for_each(t, ostreamer{os});
  os << "]";
  return os;
}
} // namespace std

namespace boost {
namespace histogram {
template <typename Histogram>
typename Histogram::element_type sum(const Histogram& h) {
  return std::accumulate(h.begin(), h.end(), typename Histogram::element_type(0));
}

struct static_tag {};
struct dynamic_tag {};

template <typename... Axes>
auto make(static_tag, Axes&&... axes)
    -> decltype(make_static_histogram(std::forward<Axes>(axes)...)) {
  return make_static_histogram(std::forward<Axes>(axes)...);
}

template <typename S, typename... Axes>
auto make_s(static_tag, S&& s, Axes&&... axes)
    -> decltype(make_static_histogram_with(s, std::forward<Axes>(axes)...)) {
  return make_static_histogram_with(s, std::forward<Axes>(axes)...);
}

template <typename... Axes>
auto make(dynamic_tag, Axes&&... axes)
    -> decltype(make_dynamic_histogram<axis::any<detail::rm_cv_ref<Axes>...>>(std::forward<Axes>(axes)...)) {
  return make_dynamic_histogram<axis::any<detail::rm_cv_ref<Axes>...>>(std::forward<Axes>(axes)...);
}

template <typename S, typename... Axes>
auto make_s(dynamic_tag, S&& s, Axes&&... axes)
    -> decltype(make_dynamic_histogram_with<axis::any<detail::rm_cv_ref<Axes>...>>(s, std::forward<Axes>(axes)...)) {
  return make_dynamic_histogram_with<axis::any<detail::rm_cv_ref<Axes>...>>(s, std::forward<Axes>(axes)...);
}

template <class T>
struct tracing_allocator {
  using value_type = T;

  std::size_t* allocated_bytes = nullptr;
  std::size_t* deallocated_bytes = nullptr;

  tracing_allocator() noexcept {}
  tracing_allocator(std::size_t& b, std::size_t& d) noexcept : allocated_bytes(&b),
                                                               deallocated_bytes(&d) {}
  template <class U>
  tracing_allocator(const tracing_allocator<U>& a) noexcept
      : allocated_bytes(a.allocated_bytes),
        deallocated_bytes(a.deallocated_bytes) {}
  ~tracing_allocator() noexcept {}

  T* allocate(std::size_t n) {
    const auto& ti = BOOST_CORE_TYPEID(T);
    std::cerr << "alloc " << n << " x " << boost::core::demangled_name(ti) << " = "
              << (n * sizeof(T)) << std::endl;
    if (allocated_bytes) *allocated_bytes += n * sizeof(T);
    return static_cast<T*>(::operator new(n * sizeof(T)));
  }
  void deallocate(T*& p, std::size_t n) {
    const auto& ti = BOOST_CORE_TYPEID(T);
    std::cerr << "dealloc " << n << " x " << boost::core::demangled_name(ti) << " = "
              << (n * sizeof(T)) << std::endl;
    if (deallocated_bytes) *deallocated_bytes += n * sizeof(T);
    ::operator delete((void*)p);
  }
};

template <class T, class U>
constexpr bool operator==(const tracing_allocator<T>&,
                          const tracing_allocator<U>&) noexcept {
  return true;
}

template <class T, class U>
constexpr bool operator!=(const tracing_allocator<T>& t,
                          const tracing_allocator<U>& u) noexcept {
  return !operator==(t, u);
}

} // namespace histogram
} // namespace boost

#endif
