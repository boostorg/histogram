// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_POLYMORPHIC_BIN_VIEW_HPP
#define BOOST_HISTOGRAM_AXIS_POLYMORPHIC_BIN_VIEW_HPP

#include <boost/histogram/detail/meta.hpp>

namespace boost {
namespace histogram {
namespace axis {

template <typename Axis>
class polymorphic_bin_view {
  using value_type = detail::return_type<decltype(&Axis::value)>;

public:
  polymorphic_bin_view(int idx, const Axis& axis, bool is_continuous)
      : idx_(idx), axis_(axis), is_continuous_(is_continuous) {}

  int idx() const noexcept { return idx_; }

  value_type value() const {
    if (is_continuous_)
      throw std::runtime_error("calling value() for continuous axis is ambiguous");
    return axis_.value(idx_);
  }
  value_type lower() const {
    if (!is_continuous_)
      throw std::runtime_error("cannot call lower() for discontinuous axis");
    return axis_.value(idx_);
  }
  value_type upper() const {
    if (!is_continuous_)
      throw std::runtime_error("cannot call upper() for discontinuous axis");
    return axis_.value(idx_ + 1);
  }
  value_type center() const {
    if (!is_continuous_)
      throw std::runtime_error("cannot call center() for discontinuous axis");
    return axis_.value(idx_ + 0.5);
  }
  template <typename U = value_type,
            typename = decltype(std::declval<U&>() - std::declval<U&>())>
  value_type width() const {
    return upper() - lower();
  }

  bool operator==(const polymorphic_bin_view& rhs) const noexcept {
    return idx_ == rhs.idx_ && axis_ == rhs.axis_;
  }
  bool operator!=(const polymorphic_bin_view& rhs) const noexcept {
    return !operator==(rhs);
  }

  explicit operator int() const noexcept { return idx_; }

  bool is_continuous() const noexcept { return is_continuous_; }

private:
  const int idx_;
  const Axis& axis_;
  const bool is_continuous_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
