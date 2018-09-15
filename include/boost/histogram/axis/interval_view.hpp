// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_INTERVAL_VIEW_HPP
#define BOOST_HISTOGRAM_AXIS_INTERVAL_VIEW_HPP

#include <functional>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

template <typename Axis>
class interval_view {
public:
  interval_view(int idx, const Axis& axis) : idx_(idx), axis_(axis) {}

  interval_view(const interval_view&) = default;
  interval_view& operator=(const interval_view&) = default;
  interval_view(interval_view&&) = default;
  interval_view& operator=(interval_view&&) = default;

  int idx() const noexcept { return idx_; }

  auto lower() const noexcept -> decltype(std::declval<Axis&>().lower(0)) {
    return axis_.lower(idx_);
  }
  auto upper() const noexcept -> decltype(std::declval<Axis&>().lower(0)) {
    return axis_.lower(idx_ + 1);
  }
  typename Axis::value_type width() const noexcept {
    return upper() - lower();
  }

  bool operator==(const interval_view& rhs) const noexcept {
    return idx_ == rhs.idx_ && axis_ == rhs.axis_;
  }
  bool operator!=(const interval_view& rhs) const noexcept {
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
