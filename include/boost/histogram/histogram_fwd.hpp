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

template <typename... Ts>
class any;
using any_std =
    any<regular<transform::identity, double, std::allocator<char>>,
        regular<transform::log, double, std::allocator<char>>,
        regular<transform::sqrt, double, std::allocator<char>>,
        regular<transform::pow, double, std::allocator<char>>,
        circular<double, std::allocator<char>>, variable<double, std::allocator<char>>,
        integer<int, std::allocator<char>>, category<int, std::allocator<char>>,
        category<std::string, std::allocator<char>>>;

} // namespace axis

template <typename Allocator = std::allocator<char>>
class adaptive_storage;
template <typename T, typename Allocator = std::allocator<T>>
class array_storage;

namespace {
template <typename... Ts>
using rebind = typename std::allocator_traits<typename mp11::mp_front<
    mp11::mp_list<Ts...>>::allocator_type>::template rebind_alloc<axis::any<Ts...>>;
}

template <typename... Ts>
using dynamic_axes = std::vector<axis::any<Ts...>, rebind<Ts...>>;
template <typename... Ts>
using static_axes = std::tuple<Ts...>;

template <class Axes = std::vector<axis::any_std>, class Storage = adaptive_storage<>>
class histogram;

} // namespace histogram
} // namespace boost

#endif
