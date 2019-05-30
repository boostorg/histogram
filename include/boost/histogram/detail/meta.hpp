// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_META_HPP
#define BOOST_HISTOGRAM_DETAIL_META_HPP

#include <boost/config/workaround.hpp>
#if BOOST_WORKAROUND(BOOST_GCC, >= 60000)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif
#include <boost/callable_traits/args.hpp>
#if BOOST_WORKAROUND(BOOST_GCC, >= 60000)
#pragma GCC diagnostic pop
#endif
#include <boost/mp11/algorithm.hpp> // mp_at
#include <boost/mp11/function.hpp>  // mp_and
#include <boost/mp11/list.hpp>      // mp_pop_front
#include <boost/mp11/utility.hpp>   // mp_if, mp_eval_or
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <class T, class U>
using convert_integer = mp11::mp_if<std::is_integral<std::decay_t<T>>, U, T>;

template <class T, class Args = boost::callable_traits::args_t<T>>
using args_type =
    mp11::mp_if<std::is_member_function_pointer<T>, mp11::mp_pop_front<Args>, Args>;

template <class T, std::size_t N = 0>
using arg_type = typename mp11::mp_at_c<args_type<T>, N>;

template <class T>
using get_scale_type_helper = typename T::value_type;

template <class T>
using get_scale_type = mp11::mp_eval_or<T, detail::get_scale_type_helper, T>;

struct one_unit {};

template <class T>
T operator*(T&& t, const one_unit&) {
  return std::forward<T>(t);
}

template <class T>
T operator/(T&& t, const one_unit&) {
  return std::forward<T>(t);
}

template <class T>
using get_unit_type_helper = typename T::unit_type;

template <class T>
using get_unit_type = mp11::mp_eval_or<one_unit, detail::get_unit_type_helper, T>;

template <class T, class R = get_scale_type<T>>
R get_scale(const T& t) {
  return t / get_unit_type<T>();
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
