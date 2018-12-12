// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_REGULAR_HPP
#define BOOST_HISTOGRAM_AXIS_REGULAR_HPP

#include <boost/container/string.hpp> // default meta data
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_bin_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/throw_exception.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace axis {

// two_pi can be found in boost/math, but it is defined here to reduce deps
constexpr double two_pi = 6.283185307179586;

namespace transform {

template <typename T>
struct identity {
  static T forward(T x) { return x; }
  static T inverse(T x) { return x; }
};

template <typename T>
struct log {
  static T forward(T x) { return std::log(x); }
  static T inverse(T x) { return std::exp(x); }
};

template <typename T>
struct sqrt {
  static T forward(T x) { return std::sqrt(x); }
  static T inverse(T x) { return x * x; }
};

template <typename T>
struct pow {
  T power = 1;

  explicit pow(T p) : power(p) {}
  pow() = default;

  auto forward(T v) const { return std::pow(v, power); }
  auto inverse(T v) const { return std::pow(v, 1.0 / power); }

  bool operator==(const pow& o) const noexcept { return power == o.power; }
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
template <typename TransformOrT, typename MetaData, option_type Options>
class regular
    : public base<MetaData, Options>,
      public iterator_mixin<regular<TransformOrT, MetaData, Options>>,
      protected std::conditional_t<detail::is_transform<TransformOrT>::value,
                                   TransformOrT, transform::identity<TransformOrT>> {
  using base_type = base<MetaData, Options>;
  using metadata_type = MetaData;
  using transform_type =
      std::conditional_t<detail::is_transform<TransformOrT>::value, TransformOrT,
                         transform::identity<TransformOrT>>;
  using value_type = detail::return_type<decltype(&transform_type::inverse)>;

  static_assert(!(Options & option_type::circular) || !(Options & option_type::underflow),
                "circular axis cannot have underflow");

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
  regular(transform_type trans, unsigned n, value_type start, value_type stop,
          metadata_type m = {})
      : base_type(n, std::move(m))
      , transform_type(std::move(trans))
      , min_(transform().forward(start))
      , delta_((transform().forward(stop) - min_) / base_type::size()) {
    if (!std::isfinite(min_) || !std::isfinite(delta_))
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("forward transform of start or stop invalid"));
    if (delta_ == 0)
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("range of forward transformed axis is zero"));
  }

  /** Construct n bins over real range [begin, end).
   *
   * \param n        number of bins.
   * \param start    low edge of first bin.
   * \param stop     high edge of last bin.
   * \param metadata description of the axis.
   * \param options  extra bin options.
   */
  regular(unsigned n, value_type start, value_type stop, metadata_type m = {})
      : regular({}, n, start, stop, std::move(m)) {}

  /// Constructor used by algorithm::reduce to shrink and rebin (not for users).
  regular(const regular& src, int begin, int end, unsigned merge)
      : base_type((end - begin) / merge, src.metadata())
      , transform_type(src.transform())
      , min_(transform().forward(src.value(begin)))
      , delta_(src.delta_ * merge) {
    BOOST_ASSERT((end - begin) % merge == 0);
    if (Options & option_type::circular && !(begin == 0 && end == src.size()))
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot shrink circular axis"));
  }

  regular() = default;

  /// Returns instance of the transform type
  const transform_type& transform() const noexcept { return *this; }

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    auto z = (this->forward(x) - min_) / delta_;
    if (Options & option_type::circular) {
      if (std::isfinite(z)) {
        z -= std::floor(z / base_type::size()) * base_type::size();
        return static_cast<int>(z);
      }
    } else {
      if (z < base_type::size()) {
        if (z >= 0)
          return static_cast<int>(z);
        else
          return -1;
      }
    }
    return base_type::size(); // also returned if x is NaN
  }

  /// Returns axis value for fractional index.
  value_type value(double i) const noexcept {
    if (Options & option_type::circular) {
      i = min_ + i * delta_;
    } else {
      if (i < 0)
        i = std::copysign(std::numeric_limits<double>::infinity(), -delta_);
      else if (i > base_type::size())
        i = std::copysign(std::numeric_limits<double>::infinity(), delta_);
      else {
        i /= base_type::size();
        i = (1 - i) * min_ + i * (min_ + delta_ * base_type::size());
      }
    }
    return transform().inverse(i);
  }

  /// Access bin at index
  auto operator[](int idx) const noexcept {
    return interval_bin_view<regular>(idx, *this);
  }

  bool operator==(const regular& o) const noexcept {
    return base_type::operator==(o) &&
           detail::static_if<detail::is_equal_comparable<transform_type>>(
               [&o](auto t) { return t == o.transform(); }, [](auto) { return true; },
               transform()) &&
           min_ == o.min_ && delta_ == o.delta_;
  }

  bool operator!=(const regular& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  detail::return_type<decltype(&transform_type::forward)> min_, delta_;
}; // namespace axis

#ifdef __cpp_deduction_guides

template <class T>
regular(unsigned, T, T)->regular<detail::convert_integer<T, double>>;

template <class T>
regular(unsigned, T, T, const char*)->regular<detail::convert_integer<T, double>>;

template <class T, class M>
regular(unsigned, T, T, M)->regular<detail::convert_integer<T, double>, M>;

template <class Tr, class T>
regular(Tr, unsigned, T, T)->regular<Tr>;

template <class Tr, class T>
regular(Tr, unsigned, T, T, const char*)->regular<Tr>;

template <class Tr, class T, class M>
regular(Tr, unsigned, T, T, M)->regular<Tr, M>;

#endif

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
