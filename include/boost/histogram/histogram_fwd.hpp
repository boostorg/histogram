// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_

#include <boost/mp11.hpp>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {

namespace axis {

namespace transform {
struct identity;
struct log;
struct sqrt;
struct pow;
} // namespace transform

template <typename Transform = transform::identity, typename T = double,
          typename Allocator = std::allocator<char>>
class regular;
template <typename T = double, typename Allocator = std::allocator<char>>
class circular;
template <typename T = double, typename Allocator = std::allocator<char>>
class variable;
template <typename T = int, typename Allocator = std::allocator<char>>
class integer;
template <typename T = int, typename Allocator = std::allocator<char>>
class category;

using types = mp11::mp_list<
    regular<transform::identity>,
    regular<transform::log>, regular<transform::sqrt>,
    regular<transform::pow>, circular<>, variable<>,
    integer<>, category<>, category<std::string>>;

template <typename... Ts>
class any;
using any_std = mp11::mp_rename<types, any>;

} // namespace axis

template <typename Allocator = std::allocator<char>>
class adaptive_storage;
template <typename T, typename Allocator = std::allocator<T>>
class array_storage;

namespace {
template <typename... Ts>
using rebind =
    typename std::allocator_traits<typename mp11::mp_front<mp11::mp_list<
        Ts...>>::allocator_type>::template rebind_alloc<axis::any<Ts...>>;
}

template <typename... Ts>
using dynamic_axes = std::vector<axis::any<Ts...>, rebind<Ts...>>;
template <typename... Ts>
using static_axes = std::tuple<Ts...>;

template <class Axes = std::vector<axis::any_std>,
          class Storage = adaptive_storage<>>
class histogram;

} // namespace histogram
} // namespace boost

#endif
