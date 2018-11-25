// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_TRAITS_HPP
#define BOOST_HISTOGRAM_AXIS_TRAITS_HPP

#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {
template <typename T>
constexpr axis::option_type options_impl(const T&, std::false_type) {
  return axis::option_type::none;
}

template <typename T>
axis::option_type options_impl(const T& t, std::true_type) {
  return t.options();
}
} // namespace detail

namespace axis {
namespace traits {
template <typename T>
decltype(auto) metadata(T&& t) noexcept {
  return detail::static_if<detail::has_method_metadata<T>>(
      [](auto&& x) -> decltype(auto) { return x.metadata(); },
      [](T &&) -> detail::copy_qualifiers<T, axis::null_type> {
        static axis::null_type m;
        return m;
      },
      std::forward<T>(t));
}

template <typename T>
option_type options(const T& t) noexcept {
  return detail::options_impl(t, detail::has_method_options<T>());
}

template <typename T>
int underflow_index(const T& t) noexcept {
  const auto opt = options(t);
  return opt & option_type::underflow ? t.size() + (opt & option_type::overflow) : -1;
}

template <typename T>
int overflow_index(const T& t) noexcept {
  const auto opt = options(t);
  return opt & option_type::overflow ? t.size() : -1;
}

template <typename T>
unsigned extend(const T& t) noexcept {
  const auto opt = options(t);
  return t.size() + (opt & option_type::underflow) + (opt & option_type::overflow);
}

} // namespace traits
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
