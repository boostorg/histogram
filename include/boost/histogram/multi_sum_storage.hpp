#ifndef BOOST_HISTOGRAM_MULTI_SUM_STORAGE_HPP
#define BOOST_HISTOGRAM_MULTI_SUM_STORAGE_HPP

#include <algorithm>
#include <boost/core/span.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/detail/iterator_adaptor.hpp>
#include <iostream>
#include <memory>

namespace boost {
namespace histogram {

namespace accumulators {
template <class T>
struct multi_sum : public boost::span<T> {
  using boost::span<T>::span;

  void operator()(boost::span<T> values) {
    if (values.size() != this->size()) throw std::runtime_error("size does not match");

    auto it = this->begin();
    for (const T& x : values) *it++ += x;
  }
};
} // namespace accumulators

template <class ElementType = double>
class multi_sum_storage {
public:
  template <class T>
  using multi_sum = accumulators::multi_sum<T>;
  using element_type = ElementType;

  using value_type = multi_sum<element_type>;
  using reference = value_type;
  using const_reference = multi_sum<const element_type>;

  template <class T>
  struct iterator_base
      : public detail::iterator_adaptor<iterator_base<T>, std::size_t, multi_sum<T>> {
    using base_type =
        detail::iterator_adaptor<iterator_base<T>, std::size_t, multi_sum<T>>;

    iterator_base(T* ptr, std::size_t idx, std::size_t nelem)
        : base_type{idx}, ptr_{ptr}, nelem_{nelem} {}
    iterator_base(const iterator_base& other)
        : base_type(other), ptr_{other.ptr_}, nelem_{other.nelem_} {}
    iterator_base& operator=(const iterator_base& other) {
      if (this != &other) {
        base_type::operator=(other);
        ptr_ = other.ptr_;
        nelem_ = other.nelem_;
      }
      return *this;
    }

    decltype(auto) operator*() {
      return multi_sum<T>{ptr_ + this->base() * nelem_, nelem_};
    }

    T* ptr_;
    std::size_t nelem_;
  };

  using iterator = iterator_base<element_type>;
  using const_iterator = iterator_base<const element_type>;

  static constexpr bool has_threading_support() { return false; }

  multi_sum_storage(const std::size_t nelem) : nelem_{nelem} {}

  std::size_t size() const { return size_; }

  void reset(std::size_t n) {
    size_ = n;
    buffer_.reset(new element_type[n * nelem_]);
  }

  iterator begin() { return {buffer_.get(), 0, nelem_}; }
  iterator end() { return {buffer_.get(), size_, nelem_}; }

  const_iterator begin() const { return {buffer_.get(), 0, nelem_}; }
  const_iterator end() const { return {buffer_.get(), size_, nelem_}; }

  reference operator[](std::size_t i) {
    return reference{buffer_.get() + i * nelem_, nelem_};
  }
  const_reference operator[](std::size_t i) const {
    return const_reference{buffer_.get() + i * nelem_, nelem_};
  }

  template <class T>
  bool operator==(const multi_sum_storage<T>& other) const {
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