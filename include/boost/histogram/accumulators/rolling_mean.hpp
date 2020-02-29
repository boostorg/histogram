// Copyright 2020 John Buonagurio
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_ROLLING_MEAN_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_ROLLING_MEAN_HPP

#include <boost/assert.hpp>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {
namespace accumulators {
namespace detail {

template <class T, class Allocator = std::allocator<T>>
struct circular_buffer
{
  using value_type = typename std::allocator_traits<Allocator>::value_type;
  using pointer = typename std::allocator_traits<Allocator>::pointer;
  using const_pointer = typename std::allocator_traits<Allocator>::const_pointer;
  using reference = value_type&;
  using const_reference = const value_type&;
  using difference_type = typename std::allocator_traits<Allocator>::difference_type;
  using size_type = typename std::allocator_traits<Allocator>::size_type;
  using allocator_type = Allocator;
  
  static_assert(std::is_trivial<value_type>::value,
                "circular_buffer requires trivial value_type");
  
  explicit circular_buffer(const allocator_type& a = {}) : alloc_(a) {}
  
  explicit circular_buffer(size_type buffer_capacity, const allocator_type& a = {}) : alloc_(a) {
    ptr_ = alloc_.allocate(buffer_capacity);
    end_ = ptr_ + buffer_capacity;
    first_ = ptr_;
    last_ = ptr_;
  }
  
  circular_buffer(const circular_buffer& rhs) : alloc_(rhs.alloc_), size_(rhs.size_) {
    ptr_ = alloc_.allocate(rhs.capacity());
    end_ = ptr_ + rhs.capacity();
    first_ = ptr_;
    last_ = std::uninitialized_copy_n(rhs.first_, size_, ptr_);
    if (last_ == end_)
      last_ = ptr_;
  }
  
  circular_buffer& operator=(const circular_buffer& rhs) {
    *this = circular_buffer(rhs);
    return *this;
  }
  
  circular_buffer(circular_buffer&& rhs) noexcept
    : alloc_(std::move(rhs.alloc_))
    , ptr_(std::exchange(rhs.ptr_, nullptr))
    , end_(std::exchange(rhs.end_, nullptr))
    , first_(std::exchange(rhs.first_, nullptr))
    , last_(std::exchange(rhs.last_, nullptr))
    , size_(std::exchange(rhs.size_, 0)) {}
  
  circular_buffer& operator=(circular_buffer&& rhs) noexcept {
    std::swap(alloc_, rhs.alloc_);
    std::swap(ptr_, rhs.ptr_);
    std::swap(end_, rhs.end_);
    std::swap(first_, rhs.first_);
    std::swap(last_, rhs.last_);
    std::swap(size_, rhs.size_);
    return *this;
  }
  
  ~circular_buffer() noexcept { destroy(); }
  
  struct iterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = circular_buffer::value_type;
    using pointer = circular_buffer::pointer;
    using reference = circular_buffer::reference;
    using size_type = circular_buffer::size_type;
    using difference_type = circular_buffer::difference_type;
    
    iterator(const circular_buffer* cb, const pointer p) : cb_(cb), it_(p) {}
    
    iterator& operator=(const iterator& rhs) {
      cb_ = rhs.cb_;
      it_ = rhs.it_;
      return *this;
    }
    
    bool operator==(const iterator& rhs) const noexcept {
      return cb_ == rhs.cb_ && it_ == rhs.it_;
    }

    bool operator!=(const iterator& rhs) const noexcept { return !operator==(rhs); }

    reference operator*() const {
      BOOST_ASSERT(cb_ != nullptr);
      BOOST_ASSERT(it_ != nullptr);
      return *it_;
    }

    pointer operator->() const { return it_; }

    iterator& operator++() {
      BOOST_ASSERT(cb_ != nullptr);
      BOOST_ASSERT(it_ != nullptr);
      cb_->increment(it_);
      if (it_ == cb_->last_)
        it_ = nullptr;
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++*this;
      return tmp;
    }
    
  private:
    const circular_buffer *cb_ = nullptr;
    pointer it_ = nullptr;
  };
  
  iterator begin() noexcept { return iterator(this, empty() ? nullptr : first_); }
  
  iterator end() noexcept { return iterator(this, nullptr); }
  
  size_type size() const noexcept { return size_; }
  
  size_type capacity() const noexcept { return end_ - ptr_; }
  
  bool empty() const noexcept { return size() == 0; }
  
  bool full() const noexcept { return capacity() == size(); }

  void push_back(const value_type& item) {
    push_back_impl<const value_type&>(item);
  }
  
  void push_back(value_type&& item) {
    push_back_impl<value_type&&>(std::move(item));
  }
  
  reference front() {
    BOOST_ASSERT(!empty());
    return *first_;
  }
  
private:
  void destroy() noexcept {
    BOOST_ASSERT((ptr_ == nullptr) == (capacity() == 0));
    if (ptr_ == nullptr) return;
    alloc_.deallocate(ptr_, capacity());
    ptr_ = nullptr;
    end_ = nullptr;
    first_ = nullptr;
    last_ = nullptr;
    size_ = 0;
  }

  void increment(pointer& p) const noexcept {
    if (++p == end_)
      p = ptr_;
  }
  
  void replace(pointer p, const value_type& item) noexcept {
    *p = item;
  }
  
  void replace(pointer p, value_type&& item) noexcept {
    *p = std::move(item);
  }

  template <class U>
  void push_back_impl(U item) {
    if (full()) {
      if (empty())
        return;
      replace(last_, static_cast<U>(item));
      increment(last_);
      first_ = last_;
    } else {
      std::allocator_traits<Allocator>::construct(alloc_, last_, static_cast<U>(item));
      increment(last_);
      ++size_;
    }
  }
  
  allocator_type alloc_;
  pointer ptr_ = nullptr;
  pointer end_ = nullptr;
  pointer first_ = nullptr;
  pointer last_ = nullptr;
  size_type size_ = 0;
  
  friend iterator;
};

} // namespace detail

template <class ValueType>
struct rolling_mean
{
  using value_type = ValueType;
  using const_reference = const value_type&;
  using buffer_type = detail::circular_buffer<value_type>;
  
  rolling_mean(std::size_t window_size) : buffer_(window_size) {}
  
  /// Allow implicit conversion from rolling_mean<T>
  template <class T>
  rolling_mean(const rolling_mean<T>& o) noexcept
    : value_{o.value_}, buffer_{o.buffer_} {}
  
  /// Insert sample x
  void operator()(const_reference x) {
    if (buffer_.full()) {
      if (buffer_.front() > x)
        value_ -= (buffer_.front() - x) / buffer_.size();
      else if (buffer_.front() < x)
        value_ += (x - buffer_.front()) / buffer_.size();
      buffer_.push_back(x);
    }
    else {
      buffer_.push_back(x);
      const auto prev_value = value_;
      if (prev_value > x)
        value_ -= (prev_value - x) / buffer_.size();
      else if (prev_value < x)
        value_ += (x - prev_value) / buffer_.size();
    }
  }
  
  /// Return how many samples were accumulated
  const_reference count() const noexcept { return buffer_.size(); }
  
  /// Return rolling mean of accumulated samples
  const_reference value() const noexcept { return value_; }
  
private:
  value_type value_{};
  buffer_type buffer_;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED
namespace std {
template <class T, class U>
struct common_type<boost::histogram::accumulators::rolling_mean<T>,
                   boost::histogram::accumulators::rolling_mean<U>> {
  using type = boost::histogram::accumulators::rolling_mean<common_type_t<T, U>>;
};
} // namespace std
#endif

#endif