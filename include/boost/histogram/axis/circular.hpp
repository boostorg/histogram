// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_CIRCULAR_HPP
#define BOOST_HISTOGRAM_AXIS_CIRCULAR_HPP

#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_bin_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace axis {

// two_pi can be found in boost/math, but it is defined here to reduce deps
constexpr double two_pi = 6.283185307179586;

/** Axis for real values on a circle.
 *
 * The axis is circular and wraps around reaching the perimeter value.
 * It has no underflow bin and the overflow bin merely counts special
 * values like NaN and infinity. Binning is a O(1) operation.
 */
template <typename RealType, typename MetaData>
class circular : public base<MetaData>,
                 public iterator_mixin<circular<RealType, MetaData>> {
  using base_type = base<MetaData>;
  using value_type = RealType;
  using metadata_type = MetaData;

public:
  /** Construct n bins with an optional offset.
   *
   * \param n         number of bins.
   * \param phase     starting phase.
   * \param perimeter range after which value wraps around.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   */
  circular(unsigned n, RealType phase = 0, RealType perimeter = two_pi,
           MetaData m = MetaData(), option_type o = option_type::overflow)
      : base_type(n, std::move(m),
                  o == option_type::underflow_and_overflow ? option_type::overflow : o)
      , phase_(phase)
      , delta_(perimeter / n) {
    if (!std::isfinite(phase) || !(perimeter > 0))
      throw std::invalid_argument("invalid phase or perimeter");
  }

  circular() = default;

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    const auto z = std::floor((x - phase_) / delta_);
    if (std::isfinite(z)) {
      const auto i = static_cast<int>(z) % base_type::size();
      return i + (i < 0) * base_type::size();
    }
    return base_type::size();
  }

  /// Returns axis value for fractional index.
  value_type value(value_type i) const noexcept { return phase_ + i * delta_; }

  auto operator[](int idx) const noexcept { return interval_bin_view<circular>(idx, *this); }

  bool operator==(const circular& o) const noexcept {
    return base_type::operator==(o) && phase_ == o.phase_ && delta_ == o.delta_;
  }

  bool operator!=(const circular<>& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  value_type phase_ = 0.0, delta_ = 1.0;
};
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
