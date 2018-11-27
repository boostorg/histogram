// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_INTERVAL_BIN_VIEW_HPP
#define BOOST_HISTOGRAM_AXIS_INTERVAL_BIN_VIEW_HPP

namespace boost {
namespace histogram {
namespace axis {

template <typename Axis>
class interval_bin_view {
public:
  interval_bin_view(int idx, const Axis& axis) : idx_(idx), axis_(axis) {}
  // avoid viewing a temporary that goes out of scope
  interval_bin_view(int idx, Axis&& axis) = delete;

  int idx() const noexcept { return idx_; }

  decltype(auto) lower() const noexcept { return axis_.value(idx_); }
  decltype(auto) upper() const noexcept { return axis_.value(idx_ + 1); }
  decltype(auto) center() const noexcept { return axis_.value(idx_ + 0.5); }
  decltype(auto) width() const noexcept { return upper() - lower(); }

  template <typename BinType>
  bool operator==(const BinType& rhs) const noexcept {
    return idx() == rhs.idx() && lower() == rhs.lower() && upper() == rhs.upper();
  }

  bool operator==(const interval_bin_view& rhs) const noexcept {
    return idx() == rhs.idx() && (&rhs.axis_ == &rhs.axis_ ||
                                  (lower() == rhs.lower() && upper() == rhs.upper()));
  }

  template <typename BinType>
  bool operator!=(const BinType& rhs) const noexcept {
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
