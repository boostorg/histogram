// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_

#include <boost/mp11.hpp>
#include <string>
#include <type_traits>

namespace boost {
namespace histogram {

class adaptive_storage;
template <typename T>
class array_storage;

namespace axis {

namespace transform {
struct identity;
struct log;
struct sqrt;
struct pow;
} // namespace transform

template <typename RealType = double,
          typename Transform = transform::identity>
class regular;
template <typename RealType = double>
class circular;
template <typename RealType = double>
class variable;
template <typename IntType = int>
class integer;
template <typename T = int>
class category;

using types = mp11::mp_list<axis::regular<double, axis::transform::identity>,
                            axis::regular<double, axis::transform::log>,
                            axis::regular<double, axis::transform::sqrt>,
                            axis::regular<double, axis::transform::pow>,
                            axis::circular<double>, axis::variable<double>,
                            axis::integer<int>, axis::category<int>,
                            axis::category<std::string>>;

template <typename... Ts>
class any;
using any_std = mp11::mp_rename<types, any>;

} // namespace axis

struct dynamic_tag {};
struct static_tag {};
template <class Type, class Axes, class Storage = adaptive_storage>
class histogram;

template <class Axes = axis::types, class Storage = adaptive_storage>
using dynamic_histogram = histogram<dynamic_tag, Axes, Storage>;

template <class Axes, class Storage = adaptive_storage>
using static_histogram = histogram<static_tag, Axes, Storage>;

namespace detail {
template <typename T>
struct weight {
  T value;
};
// template <typename T> struct is_weight : std::false_type {};
// template <typename T> struct is_weight<weight_t<T>> : std::true_type {};

template <typename T>
struct sample {
  T value;
};
// template <typename T> struct is_sample : std::false_type {};
// template <typename T> struct is_sample<sample_t<T>> : std::true_type {};
} // namespace detail

template <typename T>
detail::weight<T> weight(T&& t) {
  return {t};
}

template <typename T>
detail::sample<T> sample(T&& t) {
  return {t};
}

} // namespace histogram
} // namespace boost

#endif
