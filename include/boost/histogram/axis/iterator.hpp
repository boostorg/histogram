// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_ITERATOR_HPP
#define BOOST_HISTOGRAM_AXIS_ITERATOR_HPP

#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/reverse_iterator.hpp>

namespace boost {
namespace histogram {
namespace axis {

template <typename Axis>
class iterator
    : public boost::iterator_facade<iterator<Axis>, decltype(std::declval<Axis&>()[0]),
                                    boost::random_access_traversal_tag,
                                    decltype(std::declval<Axis&>()[0]), int> {
public:
  explicit iterator(const Axis& axis, int idx) : axis_(axis), idx_(idx) {}

protected:
  void increment() noexcept { ++idx_; }
  void decrement() noexcept { --idx_; }
  void advance(int n) noexcept { idx_ += n; }
  int distance_to(const iterator& other) const noexcept { return other.idx_ - idx_; }
  bool equal(const iterator& other) const noexcept {
    return &axis_ == &other.axis_ && idx_ == other.idx_;
  }
  decltype(std::declval<Axis&>()[0]) dereference() const { return axis_[idx_]; }

  friend class ::boost::iterator_core_access;

private:
  const Axis& axis_;
  int idx_;
};

/// Uses CRTP to inject iterator logic into Derived.
template <typename Derived>
class iterator_mixin {
public:
  using const_iterator = iterator<Derived>;
  using const_reverse_iterator = boost::reverse_iterator<const_iterator>;

  const_iterator begin() const noexcept {
    return const_iterator(*static_cast<const Derived*>(this), 0);
  }
  const_iterator end() const noexcept {
    return const_iterator(*static_cast<const Derived*>(this),
                          static_cast<const Derived*>(this)->size());
  }
  const_reverse_iterator rbegin() const noexcept {
    return boost::make_reverse_iterator(end());
  }
  const_reverse_iterator rend() const noexcept {
    return boost::make_reverse_iterator(begin());
  }
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
