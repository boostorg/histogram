// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_FWD_HPP_

#include <boost/mpl/bool.hpp>
#include <boost/mpl/vector.hpp>
#include <string>

namespace boost {
namespace histogram {

class adaptive_storage;
template <typename T> class array_storage;

namespace axis {

namespace transform {
struct identity;
struct log;
struct sqrt;
struct cos;
struct pow;
} // namespace transform

template <typename RealType = double, typename Transform = transform::identity>
class regular;
template <typename RealType = double> class circular;
template <typename RealType = double> class variable;
template <typename IntType = int> class integer;
template <typename T = int> class category;

using builtins =
    mpl::vector<axis::regular<>, axis::regular<double, axis::transform::log>,
                axis::regular<double, axis::transform::sqrt>,
                axis::regular<double, axis::transform::cos>,
                axis::regular<double, axis::transform::pow>, axis::circular<>,
                axis::variable<>, axis::integer<>, axis::category<>,
                axis::category<std::string>>;

template <typename Axes = builtins> class any;

} // namespace axis

struct dynamic_tag {};
struct static_tag {};
template <class Type, class Axes, class Storage = adaptive_storage>
class histogram;

template <class Axes = axis::builtins, class Storage = adaptive_storage>
using dynamic_histogram = histogram<dynamic_tag, Axes, Storage>;

template <class Axes, class Storage = adaptive_storage>
using static_histogram = histogram<static_tag, Axes, Storage>;

namespace detail {
template <typename T> struct weight_t { T value; };
template <typename T> struct is_weight : mpl::false_ {};
template <typename T> struct is_weight<weight_t<T>> : mpl::true_ {};

template <typename T> struct sample_t { T value; };
template <typename T> struct is_sample : mpl::false_ {};
template <typename T> struct is_sample<sample_t<T>> : mpl::true_ {};
} // namespace detail

template <typename T> detail::weight_t<T> weight(T &&t) { return {t}; }

template <typename T> detail::sample_t<T> sample(T &&t) { return {t}; }

} // namespace histogram
} // namespace boost

#endif
