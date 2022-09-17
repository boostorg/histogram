// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_SPAN_HPP
#define BOOST_HISTOGRAM_DETAIL_SPAN_HPP

#include <boost/core/span.hpp>
#include <boost/histogram/detail/nonmember_container_access.hpp>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {

namespace dtl = boost::histogram::detail;

template <class T>
auto make_span(T* begin, T* end) {
  return span<T>{begin, end};
}

template <class T>
auto make_span(T* begin, std::size_t size) {
  return span<T>{begin, size};
}

template <class Container, class = decltype(dtl::size(std::declval<Container>()),
                                            dtl::data(std::declval<Container>()))>
auto make_span(const Container& cont) {
  return make_span(dtl::data(cont), dtl::size(cont));
}

template <class T, std::size_t N>
auto make_span(T (&arr)[N]) {
  return span<T, N>(arr, N);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
