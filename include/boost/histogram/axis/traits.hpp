// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_TRAITS_HPP
#define BOOST_HISTOGRAM_AXIS_TRAITS_HPP

#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/detail/meta.hpp>

namespace boost {
namespace histogram {
namespace detail {
  template <typename B, typename T>
  struct metadata_type_impl;
  template <typename T>
  struct metadata_type_impl<std::true_type, T> {
    using type = detail::rm_cvref<decltype(std::declval<const T&>().metadata())>;
  };
  template <typename T>
  struct metadata_type_impl<std::false_type, T> {
    using type = axis::missing_metadata_type;
  };
} // namespace detail
namespace axis {
namespace traits {
  template <typename T>
  using args = detail::args_type<decltype(&T::operator())>;

  template <typename T>
  using metadata_type = typename detail::metadata_type_impl<detail::has_method_metadata<T>, T>::type;

  template <typename T>
  option_type options(const T& t) {
    return detail::overload(
      [](std::true_type, const auto& x) { return x.options(); },
      [](std::false_type, const auto&) { return axis::option_type::none; }
    )(detail::has_method_options<T>(), t);
  }

  template <typename T>
  unsigned extend(const T& t) {
    return t.size() + static_cast<unsigned>(options(t));
  }
} // namespace traits
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
