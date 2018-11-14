// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_INTEGER_HPP
#define BOOST_HISTOGRAM_AXIS_INTEGER_HPP

#include <boost/container/string.hpp> // default meta data
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_bin_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/value_bin_view.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace axis {
/** Axis for an interval of integer values with unit steps.
 *
 * Binning is a O(1) operation. This axis operates
 * faster than a regular axis.
 */
template <typename IntType, typename MetaData>
class integer : public base<MetaData>, public iterator_mixin<integer<IntType, MetaData>> {
  using base_type = base<MetaData>;
  using value_type = IntType;
  using metadata_type = MetaData;
  using bin_view =
      std::conditional_t<std::is_integral<value_type>::value, value_bin_view<integer>,
                         interval_bin_view<integer>>;

public:
  /** Construct over semi-open integer interval [start, stop).
   *
   * \param start    first integer of covered range.
   * \param stop      one past last integer of covered range.
   * \param metadata description of the axis.
   * \param options  extra bin options.
   */
  integer(value_type start, value_type stop, metadata_type m = metadata_type(),
          option_type o = option_type::underflow_and_overflow)
      : base_type(stop - start, std::move(m), o), min_(start) {
    if (start >= stop) { throw std::invalid_argument("start < stop required"); }
  }

  integer() = default;

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    x = std::floor(x - min_);
    return x >= 0 ? (x > static_cast<value_type>(base_type::size()) ? base_type::size()
                                                                    : static_cast<int>(x))
                  : -1;
  }

  /// Returns axis value for index.
  value_type value(value_type i) const noexcept {
    if (i < 0) {
      return detail::static_if<std::is_integral<value_type>>(
          [](auto) { return std::numeric_limits<value_type>::min(); },
          [](auto) { return -std::numeric_limits<value_type>::infinity(); }, 0);
    }
    if (i > static_cast<int>(base_type::size())) {
      return detail::static_if<std::is_integral<value_type>>(
          [](auto) { return std::numeric_limits<value_type>::max(); },
          [](auto) { return std::numeric_limits<value_type>::infinity(); }, 0);
    }
    return min_ + i;
  }

  decltype(auto) operator[](int idx) const noexcept { return bin_view(idx, *this); }

  bool operator==(const integer& o) const noexcept {
    return base_type::operator==(o) && min_ == o.min_;
  }

  bool operator!=(const integer& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  value_type min_ = 0;
};
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
