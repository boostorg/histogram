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
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/mp11.hpp>
#include <tuple>

namespace boost {
namespace histogram {

/// histogram factory from compile-time axis configuration with custom storage
template <typename StorageOrContainer, typename T, typename... Ts,
          typename = detail::requires_axis<T>>
auto make_histogram_with(StorageOrContainer&& s, T&& axis0, Ts&&... axis) {
  auto axes = std::make_tuple(std::forward<T>(axis0), std::forward<Ts>(axis)...);
  using U = detail::unqual<StorageOrContainer>;
  using S = mp11::mp_if<detail::is_storage<U>, U, storage_adaptor<U>>;
  return histogram<decltype(axes), S>(std::move(axes),
                                      S(std::forward<StorageOrContainer>(s)));
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
template <typename StorageOrContainer, typename Iterable,
          typename = detail::requires_sequence_of_any_axis<Iterable>>
auto make_histogram_with(StorageOrContainer&& s, Iterable&& c) {
  using U = detail::unqual<StorageOrContainer>;
  using S = mp11::mp_if<detail::is_storage<U>, U, storage_adaptor<U>>;
  using It = detail::unqual<Iterable>;
  using A = mp11::mp_if<detail::is_indexable_container<It>, It,
                        boost::container::vector<mp11::mp_first<It>>>;
  return histogram<A, S>(std::forward<Iterable>(c),
                         S(std::forward<StorageOrContainer>(s)));
}

/// histogram factory from vector-like with default storage
template <typename Iterable, typename = detail::requires_sequence_of_any_axis<Iterable>>
auto make_histogram(Iterable&& c) {
  return make_histogram_with(default_storage(), std::forward<Iterable>(c));
}

/// histogram factory from vector-like with default storage
template <typename Iterable, typename = detail::requires_sequence_of_any_axis<Iterable>>
auto make_weighted_histogram(Iterable&& c) {
  return make_histogram_with(weight_storage(), std::forward<Iterable>(c));
}

/// histogram factory from iterator range with custom storage
template <typename StorageOrContainer, typename Iterator,
          typename = detail::requires_iterator<Iterator>>
auto make_histogram_with(StorageOrContainer&& s, Iterator begin, Iterator end) {
  using T = detail::unqual<decltype(*begin)>;
  return make_histogram_with(std::forward<StorageOrContainer>(s),
                             boost::container::vector<T>(begin, end));
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
