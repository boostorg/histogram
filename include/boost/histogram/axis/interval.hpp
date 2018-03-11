// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_INTERVAL_HPP_
#define _BOOST_HISTOGRAM_AXIS_INTERVAL_HPP_

#include <functional>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

template <typename T> class interval_view {
public:
  interval_view(int idx, std::function<T(int)> eval) : idx_(idx), eval_(eval) {}

  interval_view(const interval_view &) = default;
  interval_view &operator=(const interval_view &) = default;
  interval_view(interval_view &&) = default;
  interval_view &operator=(interval_view &&) = default;

  T lower() const noexcept { return eval_(idx_); }
  T upper() const noexcept { return eval_(idx_ + 1); }
  T width() const noexcept { return upper() - lower(); }

  bool operator==(const interval_view &rhs) const noexcept {
    return lower() == rhs.lower() && upper() == rhs.upper();
  }
  bool operator!=(const interval_view &rhs) const noexcept {
    return !operator==(rhs);
  }

private:
  const int idx_;
  const std::function<T(int)> eval_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
