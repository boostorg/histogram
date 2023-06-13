// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_INT_RESOLVER_HPP
#define BOOST_HISTOGRAM_AXIS_INT_RESOLVER_HPP

#include <algorithm>
#include <boost/core/nvp.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/metadata_base.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/piece.hpp>
#include <boost/histogram/axis/piecewise.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/detail/convert_integer.hpp>
#include <boost/histogram/detail/limits.hpp>
#include <boost/histogram/detail/relaxed_equal.hpp>
#include <boost/histogram/detail/replace_type.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11/utility.hpp>
#include <boost/throw_exception.hpp>
#include <boost/variant2/variant.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

/** Bin shift integrator

  Accumulates bin shifts to support growable axes.
*/
template <class Value, class Payload>
class bin_shift_integrator {
  using value_type = Value;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /// Constructor for bin_shift_integrator
  explicit bin_shift_integrator(const Payload& payload) : payload_(payload) {}

  /// Shifts the axis
  void shift_axis(index_type n) {
    if (n < 0) {
      bins_under_ += std::abs(n);
    } else if (0 < n) {
      bins_over_ += n;
    }
  }

  /// The mapping from bin space Y to input space X
  template <class T>
  T inverse(T y) const noexcept {
    return payload_.inverse(y - bins_under_);
  }

  /// The mapping from input space X to bin space Y
  template <class T>
  T forward(T x) const noexcept {
    return payload_.forward(x) + bins_under_;
  }

  /// The number of bins in the axis
  index_type size() const noexcept { return payload_.size() + bins_under_ + bins_over_; }

private:
  Payload payload_;
  index_type bins_under_{0};
  index_type bins_over_{0};
};

/** Int resolver

  Resolves float bin numbers to integer bin numbers.
*/
template <class Value, class Payload>
class int_resolver_linear {
  using value_type = Value;
  using unit_type = detail::get_unit_type<value_type>;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  /// Constructor for int_resolver_linear
  explicit int_resolver_linear(const Payload& payload) : payload_(payload) {}

  /// The mapping from input space X to integer bin space Y
  index_type index(value_type x) const noexcept {
    // Runs in hot loop, please measure impact of changes

    const value_type y = payload_.forward(x);

    if (y < size()) {
      if (0 <= y)
        return static_cast<index_type>(y); // 0 <= i < size
      else
        return -1; // i < 0
    }

    // upper edge of last bin is inclusive if overflow bin is not present
    if (std::is_floating_point<value_type>::value) {
      // TODO: enable support for this feature
      // if (!options_type::test(option::overflow) && y == size()) return size() - 1;
    }

    return size(); // also returned if x is NaN
  }

  /// TODO: move this to a new class.
  std::pair<index_type, index_type> update(value_type x) noexcept {
    const value_type y = forward(x);

    if (y < size()) {
      if (0 <= y) {
        const auto i_int = static_cast<axis::index_type>(y);
        return {i_int, 0};
      } else if (y != -std::numeric_limits<internal_value_type>::infinity()) {
        const auto i_int = static_cast<axis::index_type>(std::floor(y));
        payload_.shift_axis(i_int);
        return {0, -i_int};
      } else {
        return {-1, 0}; // i is -infinity
      }
    }
    // i either beyond range, infinite, or NaN
    if (y < std::numeric_limits<internal_value_type>::infinity()) {
      const auto i_int = static_cast<axis::index_type>(y);
      const auto n = i_int - size() + 1;
      payload_.shift_axis(n);
      return {y, -n};
    }
    // z either infinite or NaN
    return {size(), 0};
  }

  /// The mapping from bin space Y to input space X
  value_type value(real_index_type i) const noexcept {
    if (i < 0) {
      return -std::numeric_limits<value_type>::infinity();
    } else if (i <= size()) {
      return static_cast<value_type>(payload_.inverse(i) * unit_type{});
    } else {
      return std::numeric_limits<value_type>::infinity();
    }
  }

  index_type size() const noexcept { return payload_.size(); }

private:
  internal_value_type forward(internal_value_type x) const noexcept {
    return payload_.forward(x);
  }

  Payload payload_;
};

/** Circular int resolver

  Resolves float bin numbers to integer bin numbers.
*/
template <class Value, class Payload>
class int_resolver_circular {
  using value_type = Value;

public:
  /// Constructor for int_resolver_circular
  explicit int_resolver_circular(const Payload& payload, Value x_low, Value x_high)
      : payload_(payload)
      , x_low_(x_low)
      , x_high_(x_high)
      , x_low_plus_delta_x_(x_high - x_low) {
    if (x_high <= x_low) BOOST_THROW_EXCEPTION(std::invalid_argument("x_high <= x_low"));
    if (payload.x0() < x_low)
      BOOST_THROW_EXCEPTION(std::invalid_argument("payload.x0() < x_low"));
    if (x_high < payload.xN())
      BOOST_THROW_EXCEPTION(std::invalid_argument("x_high < payload.xN()"));
  }

  index_type index(value_type x) const noexcept {
    // Take the mod of the input
    x -= x_low_plus_delta_x_ * std::floor((x - x_low_) / x_low_plus_delta_x_);

    if (x < x_low_ || x_high_ < x) {
      return payload_.size();
    } else {
      const value_type y = payload_.forward(x);

      if (std::isfinite(y)) { return static_cast<index_type>(y); }

      return payload_.size();
    }
  }

private:
  Payload payload_;
  Value x_low_;
  Value x_high_;
  Value x_low_plus_delta_x_;
};

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
