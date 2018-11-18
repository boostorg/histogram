// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_POLYMORPHIC_BIN_HPP
#define BOOST_HISTOGRAM_AXIS_POLYMORPHIC_BIN_HPP

#include <cmath>
#include <stdexcept>
#include <tuple>

namespace boost {
namespace histogram {
namespace axis {

/**
  Holds the bin data of a axis::variant.

  The interface is a superset of the `value_bin_view` and `interval_bin_view`
  classes. Calling the wrong set of methods throws `std::runtime_error`.

  This is not a view like interval_bin_view or value_bin_view for two reasons.
  - Sequential calls to lower() and upper() would have to each loop through
    the variant types. This is likely to be slower than filling all the data in
    one loop.
  - polymorphic_bin may be created from a temporary instance of axis::variant,
    like in the call histogram::axis(0). Storing a reference to the axis would
    result in a dangling reference. Rather than specialing the code to handle
    this, it seems easier to just use a value instead of a view here.
*/
template <typename T>
class polymorphic_bin {
  using value_type = T;

public:
  polymorphic_bin(int idx, std::tuple<value_type, value_type, value_type> data)
      : idx_(idx), data_(data) {}

  int idx() const noexcept { return idx_; }

  value_type value() const {
    if (!is_discrete())
      throw std::runtime_error("cannot call value() for continuous axis");
    return std::get<0>(data_);
  }
  value_type lower() const {
    if (is_discrete()) throw std::runtime_error("cannot call lower() for discrete axis");
    return std::get<0>(data_);
  }
  value_type upper() const {
    if (is_discrete()) throw std::runtime_error("cannot call upper() for discrete axis");
    return std::get<1>(data_);
  }
  value_type center() const {
    if (is_discrete()) throw std::runtime_error("cannot call center() for discrete axis");
    return std::get<2>(data_);
  }
  value_type width() const { return upper() - lower(); }

  bool operator==(const polymorphic_bin& rhs) const noexcept {
    return idx_ == rhs.idx_ && data_ == rhs.data_;
  }
  bool operator!=(const polymorphic_bin& rhs) const noexcept { return !operator==(rhs); }

  explicit operator int() const noexcept { return idx_; }

  bool is_discrete() const noexcept { return std::isnan(std::get<2>(data_)); }

private:
  const int idx_;
  const std::tuple<value_type, value_type, value_type> data_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
