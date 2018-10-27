// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_REGULAR_HPP
#define BOOST_HISTOGRAM_AXIS_REGULAR_HPP

#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_bin_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mp11.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace axis {
namespace transform {
template <typename T>
struct identity {
  static T forward(T x) { return x; }
  static T inverse(T x) { return x; }

  bool operator==(const identity&) const noexcept { return true; }
  template <class Archive>
  void serialize(Archive&, unsigned) {} // noop
};

template <typename T>
struct log : identity<T> {
  static T forward(T x) { return std::log(x); }
  static T inverse(T x) { return std::exp(x); }
};

template <typename T>
struct sqrt : identity<T> {
  static T forward(T x) { return std::sqrt(x); }
  static T inverse(T x) { return x * x; }
};

template <typename T>
struct pow {
  using U = mp11::mp_if<std::is_integral<T>, double, T>;
  U power = 1.0;

  pow(U p) : power(p) {}
  pow() = default;

  U forward(U v) const { return std::pow(v, power); }
  U inverse(U v) const { return std::pow(v, 1.0 / power); }

  bool operator==(const pow& o) const noexcept { return power == o.power; }
  template <class Archive>
  void serialize(Archive&, unsigned);
};

template <typename Q>
struct unit {
  using T = typename Q::value_type;
  using U = typename Q::unit_type;
  T forward(Q x) const { return x / U(); }
  Q inverse(T x) const { return x * U(); }
};
} // namespace transform

/** Axis for equidistant intervals on the real line.
 *
 * The most common binning strategy.
 * Very fast. Binning is a O(1) operation.
 */
template <typename Transform, typename MetaData>
class regular : public base<MetaData>,
                public iterator_mixin<regular<Transform, MetaData>>,
                protected Transform {
  using base_type = base<MetaData>;
  using transform_type = Transform;
  using external_type = detail::return_type<decltype(&Transform::inverse)>;
  using internal_type = detail::return_type<decltype(&Transform::forward)>;
  static_assert(!std::is_integral<internal_type>::value,
                "type returned by forward transform cannot be integral");
  using metadata_type = MetaData;

public:
  /** Construct n bins over real transformed range [begin, end).
   *
   * \param trans    transform instance to use.
   * \param n        number of bins.
   * \param start    low edge of first bin.
   * \param stop     high edge of last bin.
   * \param metadata description of the axis.
   * \param options  extra bin options.
   */
  regular(transform_type trans, unsigned n, external_type start, external_type stop,
          metadata_type m = {}, option_type o = option_type::underflow_and_overflow)
      : base_type(n, std::move(m), o)
      , transform_type(std::move(trans))
      , min_(this->forward(start))
      , delta_((this->forward(stop) - this->forward(start)) / n) {
    if (!std::isfinite(min_) || !std::isfinite(delta_))
      throw std::invalid_argument("forward transform of start or stop invalid");
    if (delta_ == 0)
      throw std::invalid_argument("range of forward transformed axis is zero");
  }

  /** Construct n bins over real range [begin, end).
   *
   * \param n        number of bins.
   * \param start    low edge of first bin.
   * \param stop     high edge of last bin.
   * \param metadata description of the axis.
   * \param options  extra bin options.
   */
  regular(unsigned n, external_type start, external_type stop, metadata_type m = {},
          option_type o = option_type::underflow_and_overflow)
      : regular({}, n, start, stop, std::move(m), o) {}

  regular() = default;

  /// Returns instance of the transform type
  const transform_type& transform() const noexcept { return *this; }

  /// Returns the bin index for the passed argument.
  int operator()(external_type x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    const auto z = (this->forward(x) - min_) / delta_;
    if (z < base_type::size()) {
      if (z >= 0)
        return static_cast<int>(z);
      else
        return -1;
    }
    return base_type::size(); // also returned if z is NaN

    // const auto lt_max = z < base_type::size();
    // const auto ge_zero = z >= 0;
    // return lt_max * (ge_zero * static_cast<int>(z) - !ge_zero) + !lt_max *
    // base_type::size();
  }

  /// Returns axis value for fractional index.
  external_type value(internal_type i) const noexcept {
    i /= base_type::size();
    if (i < 0)
      i = std::copysign(std::numeric_limits<internal_type>::infinity(), -delta_);
    else if (i > 1)
      i = std::copysign(std::numeric_limits<internal_type>::infinity(), delta_);
    else {
      i = (1 - i) * min_ + i * (min_ + delta_ * base_type::size());
    }
    return this->inverse(i);
  }

  /// Access bin at index
  auto operator[](int idx) const noexcept { return interval_bin_view<regular>(idx, *this); }

  bool operator==(const regular& o) const noexcept {
    return base_type::operator==(o) && transform_type::operator==(o) && min_ == o.min_ &&
           delta_ == o.delta_;
  }

  bool operator!=(const regular& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  internal_type min_, delta_;
};
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
