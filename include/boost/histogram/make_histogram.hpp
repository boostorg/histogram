// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_MAKE_HISTOGRAM_HPP
#define BOOST_HISTOGRAM_MAKE_HISTOGRAM_HPP

#include <boost/container/vector.hpp>
#include <boost/histogram/accumulators/weighted_sum.hpp>
#include <boost/histogram/adaptive_storage.hpp> // implements default_storage
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <tuple>

namespace boost {
namespace histogram {

/// histogram factory from compile-time axis configuration with custom storage
template <typename C, typename T, typename... Ts, typename = detail::requires_axis<T>>
auto make_histogram_with(C&& c, T&& axis0, Ts&&... axis) {
  using CU = detail::unqual<C>;
  using S = mp11::mp_if<detail::is_storage<CU>, CU, storage_adaptor<CU>>;
  auto axes = std::make_tuple(std::forward<T>(axis0), std::forward<Ts>(axis)...);
  return histogram<decltype(axes), S>(std::move(axes), std::forward<C>(c));
}

/// histogram factory from compile-time axis configuration with default storage
template <typename T, typename... Ts, typename = detail::requires_axis<T>>
auto make_histogram(T&& axis0, Ts&&... axis) {
  return make_histogram_with(default_storage(), std::forward<T>(axis0),
                             std::forward<Ts>(axis)...);
}

/// histogram factory from compile-time axis configuration with weight storage
template <typename T, typename... Ts, typename = detail::requires_axis<T>>
auto make_weighted_histogram(T&& axis0, Ts&&... axis) {
  return make_histogram_with(weight_storage(), std::forward<T>(axis0),
                             std::forward<Ts>(axis)...);
}

/// histogram factory from vector-like with custom storage
template <typename C, typename T, typename = detail::requires_axis_vector<T>>
auto make_histogram_with(C&& c, T&& t) {
  using CU = detail::unqual<C>;
  using S = mp11::mp_if<detail::is_storage<CU>, CU, storage_adaptor<CU>>;
  return histogram<detail::unqual<T>, S>(std::forward<T>(t), std::forward<C>(c));
}

/// histogram factory from vector-like with default storage
template <typename T, typename = detail::requires_axis_vector<T>>
auto make_histogram(T&& t) {
  return make_histogram_with(default_storage(), std::forward<T>(t));
}

/// histogram factory from vector-like with default storage
template <typename T, typename = detail::requires_axis_vector<T>>
auto make_weighted_histogram(T&& t) {
  return make_histogram_with(weight_storage(), std::forward<T>(t));
}

/// histogram factory from iterator range with custom storage
template <typename C, typename Iterator, typename = detail::requires_iterator<Iterator>>
auto make_histogram_with(C&& c, Iterator begin, Iterator end) {
  using T = detail::iterator_value_type<Iterator>;
  auto axes = std::vector<T>(begin, end);
  return make_histogram_with(std::forward<C>(c), std::move(axes));
}

/// dynamic type factory from iterator range with default storage
template <typename Iterator, typename = detail::requires_iterator<Iterator>>
auto make_histogram(Iterator begin, Iterator end) {
  return make_histogram_with(default_storage(), begin, end);
}

/// dynamic type factory from iterator range with weight storage
template <typename Iterator, typename = detail::requires_iterator<Iterator>>
auto make_weighted_histogram(Iterator begin, Iterator end) {
  return make_histogram_with(weight_storage(), begin, end);
}

} // namespace histogram
} // namespace boost

#endif
