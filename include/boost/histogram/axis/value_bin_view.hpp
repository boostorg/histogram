// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_VALUE_BIN_VIEW_HPP
#define BOOST_HISTOGRAM_AXIS_VALUE_BIN_VIEW_HPP

#include <utility>

namespace boost {
namespace histogram {
namespace axis {

template <typename Axis>
class value_bin_view {
public:
  value_bin_view(int idx, const Axis& axis) : idx_(idx), axis_(axis) {}

  int idx() const noexcept { return idx_; }

  decltype(auto) value() const { return axis_.value(idx_); }

  bool operator==(const value_bin_view& rhs) const noexcept {
    return idx_ == rhs.idx_ && axis_ == rhs.axis_;
  }
  bool operator!=(const value_bin_view& rhs) const noexcept { return !operator==(rhs); }

  explicit operator int() const noexcept { return idx_; }

private:
  const int idx_;
  const Axis& axis_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
