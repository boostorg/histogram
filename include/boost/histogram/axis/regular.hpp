// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_REGULAR_HPP
#define BOOST_HISTOGRAM_AXIS_REGULAR_HPP

#include <boost/container/string.hpp> // default meta data
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace boost {
namespace histogram {
namespace axis {

// two_pi can be found in boost/math, but it is defined here to reduce deps
constexpr double two_pi = 6.283185307179586;

namespace transform {

struct id {
  template <typename T>
  static T forward(T&& x) noexcept {
    return std::forward<T>(x);
  }
  template <typename T>
  static T inverse(T&& x) noexcept {
    return std::forward<T>(x);
  }
};

struct log {
  template <typename T>
  static T forward(T x) {
    return std::log(x);
  }
  template <typename T>
  static T inverse(T x) {
    return std::exp(x);
  }
};

struct sqrt {
  template <typename T>
  static T forward(T x) {
    return std::sqrt(x);
  }
  template <typename T>
  static T inverse(T x) {
    return x * x;
  }
};

struct pow {
  double power = 1;

  explicit pow(double p) : power(p) {}
  pow() = default;

  template <typename T>
  auto forward(T v) const {
    return std::pow(v, power);
  }
  template <typename T>
  auto inverse(T v) const {
    return std::pow(v, 1.0 / power);
  }

  bool operator==(const pow& o) const noexcept { return power == o.power; }
};

} // namespace transform

template <class, class, bool>
class optional_regular_mixin {};

/** Axis for equidistant intervals on the real line.
 *
 * The most common binning strategy.
 * Very fast. Binning is a O(1) operation.
 */
template <class Value, class Transform, class MetaData, option Options>
class regular
    : public iterator_mixin<regular<Value, Transform, MetaData, Options>>,
      public optional_regular_mixin<regular<Value, Transform, MetaData, Options>, Value,
                                    test(Options, option::growth)>,
      protected Transform {
  static_assert(!test(Options, option::circular) || !test(Options, option::underflow),
                "circular axis cannot have underflow");

public:
  using value_type = Value;
  using transform_type = Transform;
  using metadata_type = MetaData;

private:
  using unit_type = detail::get_unit_type<value_type>;
  using internal_value_type = detail::get_scale_type<value_type>;

public:
  regular() = default;

  /** Construct n bins over real transformed range [begin, end).
   *
   * \param trans    transform instance to use.
   * \param n        number of bins.
   * \param start    low edge of first bin.
   * \param stop     high edge of last bin.
   * \param metadata description of the axis.
   */
  regular(transform_type trans, unsigned n, value_type start, value_type stop,
          metadata_type m = {})
      : transform_type(std::move(trans))
      , size_meta_(static_cast<index_type>(n), std::move(m))
      , min_(this->forward(detail::get_scale(start)))
      , delta_((this->forward(detail::get_scale(stop)) - min_)) {
    if (size() == 0) BOOST_THROW_EXCEPTION(std::invalid_argument("bins > 0 required"));
    if (!std::isfinite(min_) || !std::isfinite(delta_))
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("forward transform of start or stop invalid"));
    if (delta_ == 0)
      BOOST_THROW_EXCEPTION(std::invalid_argument("range of axis is zero"));
    delta_ /= size();
  }

  /** Construct n bins over real range [begin, end).
   *
   * \param n        number of bins.
   * \param start    low edge of first bin.
   * \param stop     high edge of last bin.
   * \param metadata description of the axis.
   */
  regular(unsigned n, value_type start, value_type stop, metadata_type m = {})
      : regular({}, n, start, stop, std::move(m)) {}

  /// Constructor used by algorithm::reduce to shrink and rebin (not for users).
  regular(const regular& src, index_type begin, index_type end, unsigned merge)
      : regular(src.transform(), (end - begin) / merge, src.value(begin), src.value(end),
                src.metadata()) {
    BOOST_ASSERT((end - begin) % merge == 0);
    if (test(Options, option::circular) && !(begin == 0 && end == src.size()))
      BOOST_THROW_EXCEPTION(std::invalid_argument("cannot shrink circular axis"));
  }

  /// Returns instance of the transform type
  const transform_type& transform() const noexcept { return *this; }

  /// Returns the bin index for the passed argument.
  index_type operator()(value_type x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    auto z = (this->forward(x / unit_type()) - min_) / delta_;
    if (test(Options, option::circular)) {
      if (std::isfinite(z)) {
        z -= std::floor(z / size()) * size();
        return static_cast<index_type>(z);
      }
    } else {
      if (z < size()) {
        if (z >= 0)
          return static_cast<index_type>(z);
        else
          return -1;
      }
    }
    return size(); // also returned if x is NaN
  }

  /// Returns axis value for fractional index.
  value_type value(real_index_type i) const noexcept {
    auto z = i / size();
    if (!test(Options, option::circular) && z < 0.0)
      z = -std::numeric_limits<internal_value_type>::infinity() * delta_;
    else if (test(Options, option::circular) || z <= 1.0)
      z = (1.0 - z) * min_ + z * (min_ + size() * delta_);
    else {
      z = std::numeric_limits<internal_value_type>::infinity() * delta_;
    }
    return this->inverse(z) * unit_type();
  }

  /// Access bin at index
  decltype(auto) operator[](index_type idx) const noexcept {
    return interval_view<regular>(*this, idx);
  }

  /// Returns the number of bins, without extra bins.
  index_type size() const noexcept { return size_meta_.first(); }
  /// Returns the options.
  static constexpr option options() noexcept { return Options; }
  /// Returns the metadata.
  metadata_type& metadata() noexcept { return size_meta_.second(); }
  /// Returns the metadata (const version).
  const metadata_type& metadata() const noexcept { return size_meta_.second(); }

  bool operator==(const regular& o) const noexcept {
    return detail::relaxed_equal(transform(), o.transform()) && size() == o.size() &&
           detail::relaxed_equal(metadata(), o.metadata()) && min_ == o.min_ &&
           delta_ == o.delta_;
  }

  bool operator!=(const regular& o) const noexcept { return !operator==(o); }

  template <class Archive>
  void serialize(Archive&, unsigned);

private:
  detail::compressed_pair<index_type, metadata_type> size_meta_{0};
  internal_value_type min_{0}, delta_{1};

  template <class, class, bool>
  friend class optional_regular_mixin;
};

template <class Axis, class Value>
class optional_regular_mixin<Axis, Value, true> {
  using value_type = Value;

public:
  /// Returns index and shift (if axis has grown) for the passed argument.
  auto update(value_type x) {
    auto& der = static_cast<Axis&>(*this);
    auto z = (der.forward(x / typename Axis::unit_type{}) - der.min_) / der.delta_;
    if (std::isfinite(z)) {
      auto i = static_cast<index_type>(z);
      if (0 <= z) { // don't use i here!
        if (i < der.size()) return std::make_pair(i, 0);
        const auto n = i - der.size() + 1;
        der.size_meta_.first() += n;
        return std::make_pair(i, -n);
      } else {
        i -= 1; // correct after integral cast which rounds negative number towards zero
        der.min_ += der.delta_ * i;
        der.size_meta_.first() -= i;
        return std::make_pair(0, -i);
      }
    }
    BOOST_THROW_EXCEPTION(std::invalid_argument("argument is not finite"));
    return std::make_pair(0, 0);
  }
};

#if __cpp_deduction_guides >= 201606

template <class T>
regular(unsigned, T, T)->regular<detail::convert_integer<T, double>>;

template <class T>
regular(unsigned, T, T, const char*)->regular<detail::convert_integer<T, double>>;

template <class T, class M>
regular(unsigned, T, T, M)->regular<detail::convert_integer<T, double>, transform::id, M>;

template <class Tr, class T>
regular(Tr, unsigned, T, T)->regular<detail::convert_integer<T, double>, Tr>;

template <class Tr, class T>
regular(Tr, unsigned, T, T, const char*)->regular<detail::convert_integer<T, double>, Tr>;

template <class Tr, class T, class M>
regular(Tr, unsigned, T, T, M)->regular<detail::convert_integer<T, double>, Tr, M>;

#endif

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
