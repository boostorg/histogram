// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_ARRAY_HPP_
#define _BOOST_HISTOGRAM_STORAGE_ARRAY_HPP_

#include <algorithm>
#include <boost/histogram/storage/weight_counter.hpp>
#include <cstddef>
#include <memory>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
}
} // namespace boost

namespace boost {
namespace histogram {

template <typename T> class array_storage {
public:
  using element_type = T;
  using const_reference = const T &;

  explicit array_storage(std::size_t s)
      : size_(s), array_(new element_type[s]) {
    std::fill(array_.get(), array_.get() + s, element_type(0));
  }

  array_storage() = default;
  array_storage(const array_storage &other)
      : size_(other.size()), array_(new element_type[other.size()]) {
    std::copy(other.array_.get(), other.array_.get() + size_, array_.get());
  }
  array_storage &operator=(const array_storage &other) {
    if (this != &other) {
      reset(other.size());
      std::copy(other.array_.get(), other.array_.get() + size_, array_.get());
    }
    return *this;
  }
  array_storage(array_storage &&other) {
    std::swap(size_, other.size_);
    std::swap(array_, other.array_);
  }
  array_storage &operator=(array_storage &&other) {
    if (this != &other) {
      std::swap(size_, other.size_);
      std::swap(array_, other.array_);
    }
    return *this;
  }

  template <typename S, typename = detail::requires_storage<S>>
  explicit array_storage(const S &other) {
    reset(other.size());
    for (std::size_t i = 0; i < size_; ++i)
      array_[i] = static_cast<element_type>(other[i]);
  }

  template <typename S, typename = detail::requires_storage<S>>
  array_storage &operator=(const S &other) {
    reset(other.size());
    for (std::size_t i = 0; i < size_; ++i)
      array_[i] = static_cast<element_type>(other[i]);
    return *this;
  }

  std::size_t size() const noexcept { return size_; }

  void increase(std::size_t i) noexcept { ++array_[i]; }

  template <typename U> void add(std::size_t i, const U &x) noexcept {
    array_[i] += x;
  }

  const_reference operator[](std::size_t i) const noexcept { return array_[i]; }

  template <typename U>
  bool operator==(const array_storage<U> &rhs) const noexcept {
    if (size_ != rhs.size_)
      return false;
    return std::equal(array_.get(), array_.get() + size_, rhs.array_.get());
  }

  template <typename S> array_storage &operator+=(const S &rhs) noexcept {
    for (std::size_t i = 0; i < size_; ++i)
      add(i, rhs[i]);
    return *this;
  }

  template <typename U> array_storage &operator*=(const U &x) noexcept {
    for (std::size_t i = 0; i < size_; ++i)
      array_[i] *= x;
    return *this;
  }

private:
  std::size_t size_ = 0;
  std::unique_ptr<element_type[]> array_;

  void reset(std::size_t size) {
    size_ = size;
    array_.reset(new element_type[size]);
  }

  template <typename U> friend class array_storage;

  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

} // namespace histogram
} // namespace boost

#endif
