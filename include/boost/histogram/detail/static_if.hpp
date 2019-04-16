// Copyright 2018-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_STATIC_IF_HPP
#define BOOST_HISTOGRAM_DETAIL_STATIC_IF_HPP

#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <class T, class F, class... Args>
constexpr decltype(auto) static_if_impl(std::true_type, T&& t, F&&, Args&&... args) {
  return std::forward<T>(t)(std::forward<Args>(args)...);
}

template <class T, class F, class... Args>
constexpr decltype(auto) static_if_impl(std::false_type, T&&, F&& f, Args&&... args) {
  return std::forward<F>(f)(std::forward<Args>(args)...);
}

template <bool B, class... Ts>
constexpr decltype(auto) static_if_c(Ts&&... ts) {
  return static_if_impl(std::integral_constant<bool, B>{}, std::forward<Ts>(ts)...);
}

template <class Bool, class... Ts>
constexpr decltype(auto) static_if(Ts&&... ts) {
  return static_if_impl(Bool{}, std::forward<Ts>(ts)...);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
