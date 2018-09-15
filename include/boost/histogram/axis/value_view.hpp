// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_VALUE_VIEW_HPP
#define BOOST_HISTOGRAM_AXIS_VALUE_VIEW_HPP

#include <functional>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

template <typename Axis>
class value_view {
public:
  value_view(int idx, const Axis& axis) : idx_(idx), axis_(axis) {}

  value_view(const value_view&) = default;
  value_view& operator=(const value_view&) = default;
  value_view(value_view&&) = default;
  value_view& operator=(value_view&&) = default;

  int idx() const noexcept { return idx_; }

  auto value() const -> decltype(std::declval<Axis&>().value(0)) {
    return axis_.value(idx_);
  }

  bool operator==(const value_view& rhs) const noexcept {
    return idx_ == rhs.idx_ && axis_ == rhs.axis_;
  }
  bool operator!=(const value_view& rhs) const noexcept {
    return !operator==(rhs);
  }

  explicit operator int() const noexcept { return idx_; }

private:
  const int idx_;
  const Axis& axis_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
