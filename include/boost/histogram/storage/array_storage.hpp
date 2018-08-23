// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_ARRAY_HPP_
#define _BOOST_HISTOGRAM_STORAGE_ARRAY_HPP_

#include <algorithm>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <cstddef>
#include <memory>
#include <vector>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
}
} // namespace boost

namespace boost {
namespace histogram {

template <typename T, typename Allocator>
class array_storage {
public:
  using allocator_type = Allocator;
  using element_type = T;
  using const_reference = const T&;

private:
  using array_type = std::vector<element_type, allocator_type>;

public:
  array_storage(const array_storage&) = default;
  array_storage& operator=(const array_storage&) = default;
  array_storage(array_storage&&) = default;
  array_storage& operator=(array_storage&&) = default;

  template <typename S, typename = detail::requires_storage<S>>
  explicit array_storage(const S& o) : array_(o.get_allocator()) {
    array_.reserve(o.size());
    for (std::size_t i = 0; i < o.size(); ++i)
      array_.emplace_back(static_cast<element_type>(o[i]));
  }

  template <typename S, typename = detail::requires_storage<S>>
  array_storage& operator=(const S& o) {
    array_ = array_type(o.get_allocator());
    array_.reserve(o.size());
    for (std::size_t i = 0; i < o.size(); ++i)
      array_.emplace_back(static_cast<element_type>(o[i]));
    return *this;
  }

  explicit array_storage(const allocator_type& a = allocator_type()) : array_(a) {}

  allocator_type get_allocator() const { return array_.get_allocator(); }

  void reset(std::size_t s) {
    if (s == size()) {
      std::fill(array_.begin(), array_.end(), element_type(0));
    } else {
      array_ = array_type(s, element_type(0), array_.get_allocator());
    }
  }

  std::size_t size() const noexcept { return array_.size(); }

  void increase(std::size_t i) noexcept {
    BOOST_ASSERT(i < size());
    ++array_[i];
  }

  template <typename U>
  void add(std::size_t i, const U& x) noexcept {
    BOOST_ASSERT(i < size());
    array_[i] += x;
  }

  const_reference operator[](std::size_t i) const noexcept {
    BOOST_ASSERT(i < size());
    return array_[i];
  }

  template <typename U, typename A>
  bool operator==(const array_storage<U, A>& rhs) const noexcept {
    if (size() != rhs.size()) return false;
    return std::equal(array_.begin(), array_.end(), rhs.array_.begin());
  }

  template <typename S>
  array_storage& operator+=(const S& rhs) noexcept {
    for (std::size_t i = 0; i < size(); ++i) add(i, rhs[i]);
    return *this;
  }

  array_storage& operator*=(const element_type& x) noexcept {
    for (std::size_t i = 0; i < size(); ++i) array_[i] *= x;
    return *this;
  }

private:
  array_type array_;

  template <typename U, typename A>
  friend class array_storage;

  friend class ::boost::serialization::access;
  template <typename Archive>
  void serialize(Archive&, unsigned);
};

} // namespace histogram
} // namespace boost

#endif
