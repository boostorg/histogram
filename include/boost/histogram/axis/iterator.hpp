// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_ITERATOR_HPP
#define BOOST_HISTOGRAM_AXIS_ITERATOR_HPP

#include <boost/iterator/iterator_facade.hpp>

namespace boost {
namespace histogram {
namespace axis {

template <typename Axis>
class iterator_over
    : public iterator_facade<iterator_over<Axis>, typename Axis::bin_type,
                             random_access_traversal_tag,
                             typename Axis::bin_type> {
public:
  explicit iterator_over(const Axis& axis, int idx)
      : axis_(axis), idx_(idx) {}

  iterator_over(const iterator_over&) = default;
  iterator_over& operator=(const iterator_over&) = default;

protected:
  void increment() noexcept { ++idx_; }
  void decrement() noexcept { --idx_; }
  void advance(int n) noexcept { idx_ += n; }
  int distance_to(const iterator_over& other) const noexcept {
    return other.idx_ - idx_;
  }
  bool equal(const iterator_over& other) const noexcept {
    return &axis_ == &other.axis_ && idx_ == other.idx_;
  }
  typename Axis::bin_type dereference() const { return axis_[idx_]; }
  friend class ::boost::iterator_core_access;

  const Axis& axis_;
  int idx_;
};

template <typename Axis>
class reverse_iterator_over
    : public iterator_facade<
          reverse_iterator_over<Axis>, typename Axis::bin_type,
          random_access_traversal_tag, typename Axis::bin_type> {
public:
  explicit reverse_iterator_over(const Axis& axis, int idx)
      : axis_(axis), idx_(idx) {}

  reverse_iterator_over(const reverse_iterator_over&) = default;
  reverse_iterator_over& operator=(const reverse_iterator_over&) = default;

protected:
  void increment() noexcept { --idx_; }
  void decrement() noexcept { ++idx_; }
  void advance(int n) noexcept { idx_ -= n; }
  int distance_to(const reverse_iterator_over& other) const noexcept {
    return other.idx_ - idx_;
  }
  bool equal(const reverse_iterator_over& other) const noexcept {
    return &axis_ == &other.axis_ && idx_ == other.idx_;
  }
  typename Axis::bin_type dereference() const { return axis_[idx_ - 1]; }
  friend class ::boost::iterator_core_access;

  const Axis& axis_;
  int idx_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
