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
namespace axis {
namespace traits {
  template <typename T>
  using args = detail::args_type<decltype(&T::operator())>;

  template <typename T>
  decltype(auto) metadata(T&& t) {
    return detail::static_if<detail::has_method_metadata<T>>(
      [](auto&& x) -> decltype(auto) { return x.metadata(); },
      [](auto&&) -> detail::copy_qualifiers<T, axis::empty_metadata_type> {
        static axis::empty_metadata_type m; return m;
      },
      t);
  }

  template <typename T>
  option_type options(const T& t) {
    return detail::static_if<detail::has_method_options<T>>(
      [](const auto& x) { return x.options(); },
      [](const T&) { return axis::option_type::none; },
      t);
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
