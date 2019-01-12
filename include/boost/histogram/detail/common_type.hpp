// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_COMMON_TYPE_HPP
#define BOOST_HISTOGRAM_DETAIL_COMMON_TYPE_HPP

#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>
#include <tuple>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {
template <class T, class U>
// clang-format off
using common_axes = mp11::mp_cond<
  std::is_same<T, U>, T,
  is_tuple<T>, T,
  is_tuple<U>, U,
  is_sequence_of_axis<T>, T,
  is_sequence_of_axis<U>, U,
  std::true_type, T
>;
// clang-format on

template <class T, class U>
struct common_storage_impl;

template <class T, class U>
struct common_storage_impl<storage_adaptor<T>, storage_adaptor<U>> {
  using V = std::common_type_t<mp11::mp_front<T>, mp11::mp_front<U>>;
  using type = storage_adaptor<mp11::mp_replace_front<T, V>>;
};

template <class T, class A>
struct common_storage_impl<storage_adaptor<T>, adaptive_storage<A>> {
  using U = typename adaptive_storage<A>::value_type;
  using V = std::common_type_t<mp11::mp_front<T>, U>;
  using type = storage_adaptor<mp11::mp_replace_front<T, V>>;
};

template <class T, class A>
struct common_storage_impl<adaptive_storage<A>, storage_adaptor<T>>
    : common_storage_impl<storage_adaptor<T>, adaptive_storage<A>> {};

template <class A1, class A2>
struct common_storage_impl<adaptive_storage<A1>, adaptive_storage<A2>> {
  using type = adaptive_storage<A1>;
};

template <class A, class B>
using common_storage = typename common_storage_impl<A, B>::type;
} // namespace detail
} // namespace histogram
} // namespace boost

#endif
