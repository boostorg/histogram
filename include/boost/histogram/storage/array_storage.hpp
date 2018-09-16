// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_STORAGE_ARRAY_HPP
#define BOOST_HISTOGRAM_STORAGE_ARRAY_HPP

#include <algorithm>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/assert.hpp>
#include <cstddef>
#include <memory>
#include <vector>

namespace boost {
namespace histogram {

template <typename T, typename ScaleType, typename Allocator>
struct array_storage {
  using allocator_type = Allocator;
  using element_type = T;
  using scale_type = ScaleType;
  using const_reference = const T&;

  using buffer_type = std::vector<element_type, allocator_type>;

  array_storage(const array_storage&) = default;
  array_storage& operator=(const array_storage&) = default;
  array_storage(array_storage&&) = default;
  array_storage& operator=(array_storage&&) = default;

  template <typename S, typename = detail::requires_storage<S>>
  explicit array_storage(const S& o) : buffer(o.get_allocator()) {
    buffer.reserve(o.size());
    for (std::size_t i = 0; i < o.size(); ++i)
      buffer.emplace_back(static_cast<element_type>(o[i]));
  }

  template <typename S, typename = detail::requires_storage<S>>
  array_storage& operator=(const S& o) {
    buffer = buffer_type(o.get_allocator());
    buffer.reserve(o.size());
    for (std::size_t i = 0; i < o.size(); ++i)
      buffer.emplace_back(static_cast<element_type>(o[i]));
    return *this;
  }

  explicit array_storage(const allocator_type& a = allocator_type()) : buffer(a) {}

  allocator_type get_allocator() const { return buffer.get_allocator(); }

  void reset(std::size_t s) {
    if (s == size()) {
      std::fill(buffer.begin(), buffer.end(), element_type(0));
    } else {
      buffer = buffer_type(s, element_type(0), buffer.get_allocator());
    }
  }

  std::size_t size() const noexcept { return buffer.size(); }

  void increase(std::size_t i) noexcept {
    BOOST_ASSERT(i < size());
    ++buffer[i];
  }

  template <typename U>
  void add(std::size_t i, const U& x) noexcept {
    BOOST_ASSERT(i < size());
    buffer[i] += x;
  }

  const_reference operator[](std::size_t i) const noexcept {
    BOOST_ASSERT(i < size());
    return buffer[i];
  }

  template <typename... Ts>
  bool operator==(const array_storage<Ts...>& rhs) const noexcept {
    if (size() != rhs.size()) return false;
    return std::equal(buffer.begin(), buffer.end(), rhs.buffer.begin());
  }

  template <typename S>
  array_storage& operator+=(const S& rhs) noexcept {
    for (std::size_t i = 0; i < size(); ++i) add(i, rhs[i]);
    return *this;
  }

  array_storage& operator*=(const scale_type& x) noexcept {
    for (std::size_t i = 0; i < size(); ++i) buffer[i] *= x;
    return *this;
  }

  buffer_type buffer;
};

} // namespace histogram
} // namespace boost

#endif
