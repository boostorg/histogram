// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_BUFFER_HPP
#define BOOST_HISTOGRAM_DETAIL_BUFFER_HPP

#include <boost/histogram/detail/meta.hpp>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <typename AT, typename... Ts>
void create_buffer_impl(std::true_type, typename AT::allocator_type& a, std::size_t n,
                        typename AT::pointer it, Ts&&... ts) {
  // never throws
  for (const auto end = it + n; it != end; ++it)
    AT::construct(a, it, std::forward<Ts>(ts)...);
}

template <typename AT, typename... Ts>
void create_buffer_impl(std::false_type, typename AT::allocator_type& a, std::size_t n,
                        typename AT::pointer it, Ts&&... ts) {
  const auto ptr = it;
  try {
    for (const auto end = it + n; it != end; ++it)
      AT::construct(a, it, std::forward<Ts>(ts)...);
  } catch (...) {
    // release resources that were already acquired before rethrowing
    while (it != ptr) AT::destroy(a, it--);
    AT::deallocate(a, ptr, n);
    throw;
  }
}

template <typename AT, typename Iterator, typename... Ts>
void create_buffer_from_iter_impl(std::true_type, typename AT::allocator_type& a,
                                  std::size_t n, typename AT::pointer it,
                                  Iterator iter, Ts&&... ts) {
  // never throws
  for (const auto end = it + n; it != end; ++it)
    AT::construct(a, it, *iter++, std::forward<Ts>(ts)...);
}

template <typename AT, typename Iterator, typename... Ts>
void create_buffer_from_iter_impl(std::false_type, typename AT::allocator_type& a,
                                  std::size_t n, typename AT::pointer it,
                                  Iterator iter, Ts&&... ts) {
  const auto ptr = it;
  try {
    for (const auto end = it + n; it != end; ++it)
      AT::construct(a, it, *iter++, std::forward<Ts>(ts)...);
  } catch (...) {
    // release resources that were already acquired before rethrowing
    while (it != ptr) AT::destroy(a, it--);
    AT::deallocate(a, ptr, n);
    throw;
  }
}

template <typename Allocator, typename... Ts>
typename std::allocator_traits<Allocator>::pointer create_buffer(Allocator& a,
                                                                 std::size_t n,
                                                                 Ts&&... ts) {
  using AT = std::allocator_traits<Allocator>;
  using T = typename AT::value_type;
  auto ptr = AT::allocate(a, n); // may throw
  create_buffer_impl<AT>(std::is_nothrow_constructible<T, Ts...>(), a, n, ptr,
                         std::forward<Ts>(ts)...);
  return ptr;
}

template <typename Allocator, typename Iterator, typename... Ts>
typename std::allocator_traits<Allocator>::pointer create_buffer_from_iter(
    Allocator& a, std::size_t n, Iterator iter, Ts&&... ts) {
  using AT = std::allocator_traits<Allocator>;
  using T = typename AT::value_type;
  using U = decltype(*iter);
  auto ptr = AT::allocate(a, n); // may throw
  create_buffer_from_iter_impl<AT>(std::is_nothrow_constructible<T, U>(), a, n, ptr,
                                   iter, std::forward<Ts>(ts)...);
  return ptr;
}

template <typename Allocator>
void destroy_buffer(Allocator& a, typename std::allocator_traits<Allocator>::pointer p,
                    std::size_t n) {
  using AT = std::allocator_traits<Allocator>;
  auto it = p + n;
  const auto end = p;
  while (it != end) {
    --it;
    AT::destroy(a, it);
  }
  AT::deallocate(a, p, n);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
