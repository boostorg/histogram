// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_SAFE_COMPARISON_HPP
#define BOOST_HISTOGRAM_DETAIL_SAFE_COMPARISON_HPP

#include <boost/histogram/detail/large_int.hpp>
#include <boost/mp11/utility.hpp>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <class T>
auto make_unsigned(const T t) noexcept {
  static_assert(std::is_integral<T>::value, "");
  return static_cast<typename std::make_unsigned<T>::type>(t);
}

// clang-format off
template <class T>
using number_category = mp11::mp_cond<
  is_large_int<T>, unsigned,
  std::is_floating_point<T>, float,
  std::is_integral<T>, mp11::mp_if<std::is_signed<T>, int, unsigned>,
  std::true_type, char>; // do not handle this type
// clang-format on

// version of std::equal_to<> which handles signed, unsigned, large_int, floating
struct equal {
  template <class T, class U>
  bool operator()(const T& t, const U& u) const noexcept {
    return impl(number_category<T>{}, number_category<U>{}, t, u);
  }

  template <class C1, class C2, class T, class U>
  bool impl(C1, C2, const T& t, const U& u) const noexcept {
    return t == u;
  }

  template <class C, class T, class U>
  bool impl(float, C, const T& t, const U& u) const noexcept {
    return t == static_cast<T>(u);
  }

  template <class C, class T, class U>
  bool impl(C, float, const T& t, const U& u) const noexcept {
    return impl(float{}, C{}, u, t);
  }

  template <class T, class U>
  bool impl(float, float, const T& t, const U& u) const noexcept {
    return t == u;
  }

  template <class T, class U>
  bool impl(int, unsigned, const T& t, const U& u) const noexcept {
    return t >= 0 && make_unsigned(t) == u;
  }

  template <class T, class U>
  bool impl(unsigned, int, const T& t, const U& u) const noexcept {
    return impl(int{}, unsigned{}, u, t);
  }
};

// version of std::less<> which handles comparison of signed and unsigned
struct less {
  template <class T, class U>
  bool operator()(const T& t, const U& u) const noexcept {
    return impl(number_category<T>{}, number_category<U>{}, t, u);
  }

  template <class C1, class C2, class T, class U>
  bool impl(C1, C2, const T& t, const U& u) const noexcept {
    return t < u;
  }

  template <class C, class T, class U>
  bool impl(float, C, const T& t, const U& u) const noexcept {
    return t < static_cast<T>(u);
  }

  template <class C, class T, class U>
  bool impl(C, float, const T& t, const U& u) const noexcept {
    return static_cast<U>(t) < u;
  }

  template <class T, class U>
  bool impl(float, float, const T& t, const U& u) const noexcept {
    return t < u;
  }

  template <class T, class U>
  bool impl(int, unsigned, const T& t, const U& u) const noexcept {
    return t < 0 || make_unsigned(t) < u;
  }

  template <class T, class U>
  bool impl(unsigned, int, const T& t, const U& u) const noexcept {
    return 0 < u && t < make_unsigned(u);
  }
};

// version of std::greater<> which handles comparison of signed and unsigned
struct greater {
  template <class T, class U>
  bool operator()(const T& t, const U& u) const noexcept {
    return less()(u, t);
  }
};

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
