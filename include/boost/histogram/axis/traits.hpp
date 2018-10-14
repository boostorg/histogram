// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_TRAITS_HPP
#define BOOST_HISTOGRAM_AXIS_TRAITS_HPP

#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/callable_traits/args.hpp>
#include <type_traits>

namespace boost {
namespace histogram {

namespace detail {
template <typename T>
axis::option_type axis_traits_options_impl(std::true_type, const T& t) {
  return t.options();
}

template <typename T>
axis::option_type axis_traits_options_impl(std::false_type, const T&) {
  return axis::option_type::none;
}
} // namespace detail

namespace axis {

template <typename T>
struct traits {
  using call_args = boost::callable_traits::args_t<decltype(&T::operator())>;

  static option_type options(const T& t) {
    return detail::axis_traits_options_impl(detail::has_method_options<T>(), t);
  }

  static unsigned extend(const T& t) {
    return t.size() + static_cast<unsigned>(options(t));
  }
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
