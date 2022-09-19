// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_MULTI_STORAGE_HPP
#define BOOST_HISTOGRAM_MULTI_STORAGE_HPP

#include <algorithm>
#include <boost/core/span.hpp>
#include <boost/histogram/detail/iterator_adaptor.hpp>
#include <boost/histogram/fwd.hpp>
#include <memory>
#include <vector>

namespace boost {
namespace histogram {

template <class ElementType = double>
class multi_storage {
public:
  using element_type = ElementType;

  struct accumulator : public span<element_type> {
    using span<element_type>::span;

    void operator()(span<const element_type> values) {
      if (values.size() != this->size()) throw std::runtime_error("size does not match");

      auto it = this->begin();
      for (const element_type& v : values) {
        // TODO make this more flexible, support operator++ and operator()
        *it++ += v;
      }
    }
  };

  using value_type = std::vector<element_type>;
  using reference = span<element_type>;
  using const_reference = span<const element_type>;

  template <class T>
  struct iterator_base
      : public detail::iterator_adaptor<iterator_base<T>, T*, span<T>, value_type> {
    using reference = typename iterator_base::reference;
    using difference_type = typename iterator_base::difference_type;

    iterator_base(T* ptr, std::size_t nelem)
        : iterator_base::iterator_adaptor_{ptr}, nelem_{nelem} {}
    iterator_base(const iterator_base& other)
        : iterator_base::iterator_adaptor_(other), nelem_{other.nelem_} {}
    iterator_base& operator=(const iterator_base& other) {
      if (this != &other) {
        iterator_base::iterator_adaptor_::operator=(other);
        nelem_ = other.nelem_;
      }
      return *this;
    }

    iterator_base& operator+=(difference_type n) {
      iterator_base::iterator_adaptor_::operator+=(n* nelem_);
      return *this;
    }

    reference operator*() { return {this->base(), nelem_}; }

    std::size_t nelem_;
  };

  using iterator = iterator_base<element_type>;
  using const_iterator = iterator_base<const element_type>;

  static constexpr bool has_threading_support() { return false; }

  multi_storage(const std::size_t nelem) : nelem_{nelem} {}

  std::size_t size() const { return size_; }
  std::size_t width() const { return nelem_; }

  void reset(std::size_t n) {
    size_ = n;
    buffer_.reset(new element_type[n * nelem_]);
  }

  iterator begin() { return {buffer_.get(), nelem_}; }
  iterator end() { return {buffer_.get() + size_ * nelem_, nelem_}; }

  const_iterator begin() const { return {buffer_.get(), nelem_}; }
  const_iterator end() const { return {buffer_.get() + size_ * nelem_, nelem_}; }

  reference operator[](std::size_t i) { return {buffer_.get() + i * nelem_, nelem_}; }
  const_reference operator[](std::size_t i) const {
    return {buffer_.get() + i * nelem_, nelem_};
  }

  template <class T>
  bool operator==(const multi_storage<T>& other) const {
    if (size() != other.size()) return false;
    return std::equal(buffer_._get(), buffer_.get() + size_ * nelem_, other.ptr_.get());
  }

public:
  std::size_t size_ = 0;
  std::size_t nelem_;
  std::unique_ptr<element_type[]> buffer_;
};

} // namespace histogram
} // namespace boost

#endif