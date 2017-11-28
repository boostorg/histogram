// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_ITERATOR_HPP_
#define _BOOST_HISTOGRAM_AXIS_ITERATOR_HPP_

#include <boost/iterator/iterator_facade.hpp>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

template <typename Axis>
class iterator_over
    : public iterator_facade<iterator_over<Axis>,
                             std::pair<int, typename Axis::bin_type>,
                             random_access_traversal_tag,
                             std::pair<int, typename Axis::bin_type>> {
public:
  explicit iterator_over(const Axis &axis, int idx) : axis_(axis), idx_(idx) {}

  iterator_over(const iterator_over &) = default;
  iterator_over &operator=(const iterator_over &) = default;

private:
  void increment() noexcept { ++idx_; }
  void decrement() noexcept { --idx_; }
  void advance(int n) noexcept { idx_ += n; }
  int distance_to(const iterator_over &other) const noexcept {
    return other.idx_ - idx_;
  }
  bool equal(const iterator_over &other) const noexcept {
    return idx_ == other.idx_;
  }
  std::pair<int, typename Axis::bin_type> dereference() const {
    return std::make_pair(idx_, axis_[idx_]);
  }
  const Axis &axis_;
  int idx_;
  friend class ::boost::iterator_core_access;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
