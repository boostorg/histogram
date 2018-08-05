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

template <typename T = double, typename Transform = transform::identity,
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

using types =
    mp11::mp_list<axis::regular<axis::transform::identity>,
                  axis::regular<axis::transform::log>,
                  axis::regular<axis::transform::sqrt>,
                  axis::regular<axis::transform::pow>, axis::circular<>,
                  axis::variable<>, axis::integer<>, axis::category<>>;

template <typename... Ts>
class any;
using any_std = mp11::mp_rename<types, any>;

} // namespace axis

template <typename Allocator = std::allocator<char>>
class adaptive_storage;
template <typename T, typename Allocator = std::allocator<T>>
class array_storage;

template <typename... Ts>
using dynamic_axes = std::vector<axis::any<Ts...>>;
template <typename... Ts>
using static_axes = std::tuple<Ts...>;

template <class Axes, class Storage>
class histogram;

} // namespace histogram
} // namespace boost

#endif
