// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_TYPES_HPP_
#define _BOOST_HISTOGRAM_AXIS_TYPES_HPP_

#include <algorithm>
#include <boost/bimap.hpp>
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/value_view.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
} // namespace serialization
} // namespace boost

namespace boost {
namespace histogram {
namespace axis {

namespace transform {
namespace detail {
struct stateless {
  bool operator==(const stateless&) const noexcept { return true; }
  template <class Archive>
  void serialize(Archive&, unsigned) {}
};
} // namespace detail

struct identity : public detail::stateless {
  template <typename T>
  static T&& forward(T&& v) {
    return std::forward<T>(v);
  }
  template <typename T>
  static T&& inverse(T&& v) {
    return std::forward<T>(v);
  }
};

struct log : public detail::stateless {
  template <typename T>
  static T forward(T v) {
    return std::log(v);
  }
  template <typename T>
  static T inverse(T v) {
    return std::exp(v);
  }
};

struct sqrt : public detail::stateless {
  template <typename T>
  static T forward(T v) {
    return std::sqrt(v);
  }
  template <typename T>
  static T inverse(T v) {
    return v * v;
  }
};

// struct cos : public detail::stateless {
//   template <typename T> static T forward(T v) { return std::cos(v); }
//   template <typename T> static T inverse(T v) { return std::acos(v); }
// };

struct pow {
  double power = 1.0;

  pow() = default;
  pow(double p) : power(p) {}
  template <typename T>
  T forward(T v) const {
    return std::pow(v, power);
  }
  template <typename T>
  T inverse(T v) const {
    return std::pow(v, 1.0 / power);
  }
  bool operator==(const pow& other) const noexcept {
    return power == other.power;
  }

private:
  friend ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};
} // namespace transform

/** Axis for equidistant intervals on the real line.
 *
 * The most common binning strategy.
 * Very fast. Binning is a O(1) operation.
 */
// private inheritance from Transform wastes no space if it is stateless
template <typename RealType, typename Transform>
class regular : public base,
                public iterator_mixin<regular<RealType, Transform>>,
                Transform {
public:
  using value_type = RealType;
  using bin_type = interval_view<regular>;

  /** Construct axis with n bins over real range [lower, upper).
   *
   * \param n number of bins.
   * \param lower low edge of first bin.
   * \param upper high edge of last bin.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   * \param trans arguments passed to the transform.
   */
  regular(unsigned n, value_type lower, value_type upper,
          string_view label = {}, axis::uoflow uo = axis::uoflow::on,
          Transform trans = Transform())
      : base(n, label, uo),
        Transform(trans),
        min_(trans.forward(lower)),
        delta_((trans.forward(upper) - trans.forward(lower)) / n) {
    if (lower < upper) {
      BOOST_ASSERT(!std::isnan(min_));
      BOOST_ASSERT(!std::isnan(delta_));
    } else {
      throw std::invalid_argument("lower < upper required");
    }
  }

  regular() = default;
  regular(const regular&) = default;
  regular& operator=(const regular&) = default;
  regular(regular&&) = default;
  regular& operator=(regular&&) = default;

  /// Returns the bin index for the passed argument.
  int index(value_type x) const noexcept {
    // Optimized code, measure impact of changes
    const value_type z = (Transform::forward(x) - min_) / delta_;
    return z < base::size() ? (z >= 0.0 ? static_cast<int>(z) : -1)
                            : base::size();
  }

  /// Returns lower edge of bin.
  value_type lower(int i) const noexcept {
    const auto n = base::size();
    value_type x;
    if (i < 0)
      x = -std::numeric_limits<value_type>::infinity();
    else if (i > n)
      x = std::numeric_limits<value_type>::infinity();
    else {
      const auto z = value_type(i) / n;
      x = (1.0 - z) * min_ + z * (min_ + delta_ * n);
    }
    return Transform::inverse(x);
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const regular& o) const noexcept {
    return base::operator==(o) && Transform::operator==(o) &&
           min_ == o.min_ && delta_ == o.delta_;
  }

  /// Access properties of the transform.
  const Transform& transform() const noexcept {
    return static_cast<const Transform&>(*this);
  }

private:
  value_type min_ = 0.0, delta_ = 1.0;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

/** Axis for real values on a circle.
 *
 * The axis is circular and wraps around reaching the
 * perimeter value. Therefore, there are no overflow/underflow
 * bins for this axis. Binning is a O(1) operation.
 */
template <typename RealType>
class circular : public base, public iterator_mixin<circular<RealType>> {
public:
  using value_type = RealType;
  using bin_type = interval_view<circular>;

  /** Constructor for n bins with an optional offset.
   *
   * \param n         number of bins.
   * \param phase     starting phase.
   * \param perimeter range after which value wraps around.
   * \param label     description of the axis.
   */
  explicit circular(unsigned n, value_type phase = 0.0,
                    value_type perimeter = boost::histogram::detail::two_pi,
                    string_view label = {})
      : base(n, label, axis::uoflow::off),
        phase_(phase),
        perimeter_(perimeter) {}

  circular() = default;
  circular(const circular&) = default;
  circular& operator=(const circular&) = default;
  circular(circular&&) = default;
  circular& operator=(circular&&) = default;

  /// Returns the bin index for the passed argument.
  int index(value_type x) const noexcept {
    const value_type z = (x - phase_) / perimeter_;
    const int i =
        static_cast<int>(std::floor(z * base::size())) % base::size();
    return i + (i < 0) * base::size();
  }

  /// Returns lower edge of bin.
  value_type lower(int i) const noexcept {
    const value_type z = value_type(i) / base::size();
    return z * perimeter_ + phase_;
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const circular& o) const noexcept {
    return base::operator==(o) && phase_ == o.phase_ &&
           perimeter_ == o.perimeter_;
  }

  value_type perimeter() const { return perimeter_; }
  value_type phase() const { return phase_; }

private:
  value_type phase_ = 0.0, perimeter_ = 1.0;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

/** Axis for non-equidistant bins on the real line.
 *
 * Binning is a O(log(N)) operation. If speed matters and the problem
 * domain allows it, prefer a regular axis, possibly with a transform.
 */
template <typename RealType>
class variable : public base, public iterator_mixin<variable<RealType>> {
public:
  using value_type = RealType;
  using bin_type = interval_view<variable>;

  /** Construct an axis from bin edges.
   *
   * \param x sequence of bin edges.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   */
  variable(std::initializer_list<value_type> x, string_view label = {},
           axis::uoflow uo = axis::uoflow::on)
      : base(x.size() - 1, label, uo), x_(new value_type[x.size()]) {
    if (x.size() >= 2) {
      std::copy(x.begin(), x.end(), x_.get());
      std::sort(x_.get(), x_.get() + base::size() + 1);
    } else {
      throw std::invalid_argument("at least two values required");
    }
  }

  template <typename Iterator>
  variable(Iterator begin, Iterator end, string_view label = {},
           axis::uoflow uo = axis::uoflow::on)
      : base(std::distance(begin, end) - 1, label, uo),
        x_(new value_type[std::distance(begin, end)]) {
    std::copy(begin, end, x_.get());
    std::sort(x_.get(), x_.get() + base::size() + 1);
  }

  variable() = default;
  variable(const variable& o)
      : base(o), x_(new value_type[base::size() + 1]) {
    std::copy(o.x_.get(), o.x_.get() + base::size() + 1, x_.get());
  }
  variable& operator=(const variable& o) {
    if (this != &o) {
      base::operator=(o);
      x_.reset(new value_type[base::size() + 1]);
      std::copy(o.x_.get(), o.x_.get() + base::size() + 1, x_.get());
    }
    return *this;
  }
  variable(variable&&) = default;
  variable& operator=(variable&&) = default;

  /// Returns the bin index for the passed argument.
  int index(value_type x) const noexcept {
    return std::upper_bound(x_.get(), x_.get() + base::size() + 1, x) -
           x_.get() - 1;
  }

  /// Returns the starting edge of the bin.
  value_type lower(int i) const noexcept {
    if (i < 0) { return -std::numeric_limits<value_type>::infinity(); }
    if (i > base::size()) {
      return std::numeric_limits<value_type>::infinity();
    }
    return x_[i];
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const variable& o) const noexcept {
    if (!base::operator==(o)) { return false; }
    return std::equal(x_.get(), x_.get() + base::size() + 1, o.x_.get());
  }

private:
  std::unique_ptr<value_type[]> x_; // smaller size compared to std::vector

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

/** Axis for an interval of integral values with unit steps.
 *
 * Binning is a O(1) operation. This axis operates
 * faster than a regular.
 */
template <typename IntType>
class integer : public base, public iterator_mixin<integer<IntType>> {
public:
  using value_type = IntType;
  using bin_type = interval_view<integer>;

  /** Construct axis over a semi-open integer interval [lower, upper).
   *
   * \param lower smallest integer of the covered range.
   * \param upper largest integer of the covered range.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   */
  integer(value_type lower, value_type upper, string_view label = {},
          axis::uoflow uo = axis::uoflow::on)
      : base(upper - lower, label, uo), min_(lower) {
    if (!(lower < upper)) {
      throw std::invalid_argument("lower < upper required");
    }
  }

  integer() = default;
  integer(const integer&) = default;
  integer& operator=(const integer&) = default;
  integer(integer&&) = default;
  integer& operator=(integer&&) = default;

  /// Returns the bin index for the passed argument.
  int index(value_type x) const noexcept {
    const int z = x - min_;
    return z >= 0 ? (z > base::size() ? base::size() : z) : -1;
  }

  /// Returns lower edge of the integral bin.
  value_type lower(int i) const noexcept {
    if (i < 0) { return -std::numeric_limits<value_type>::max(); }
    if (i > base::size()) { return std::numeric_limits<value_type>::max(); }
    return min_ + i;
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const integer& o) const noexcept {
    return base::operator==(o) && min_ == o.min_;
  }

private:
  value_type min_ = 0;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

/** Axis which maps unique single values to bins (one on one).
 *
 * The axis maps a set of values to bins, following the order of
 * arguments in the constructor. There is an optional overflow bin
 * for this axis, which counts values that are not part of the set.
 * Binning is a O(1) operation. The value type must be hashable.
 */
template <typename T>
class category : public base, public iterator_mixin<category<T>> {
  using map_type = bimap<T, int>;

public:
  using value_type = T;
  using bin_type = value_view<category>;

  category() = default;
  category(const category& rhs) : base(rhs), map_(new map_type(*rhs.map_)) {}
  category& operator=(const category& rhs) {
    if (this != &rhs) {
      base::operator=(rhs);
      map_.reset(new map_type(*rhs.map_));
    }
    return *this;
  }
  category(category&& rhs) = default;
  category& operator=(category&& rhs) = default;

  /** Construct from an initializer list of strings.
   *
   * \param seq sequence of unique values.
   */
  category(std::initializer_list<value_type> seq, string_view label = {},
           axis::uoflow uo = axis::uoflow::oflow)
      : base(seq.size(), label,
             uo == axis::uoflow::on ? axis::uoflow::oflow : uo),
        map_(new map_type()) {
    int index = 0;
    for (const auto& x : seq) map_->insert({x, index++});
    if (index == 0) throw std::invalid_argument("sequence is empty");
  }

  template <typename Iterator,
            typename = boost::histogram::detail::requires_iterator<Iterator>>
  category(Iterator begin, Iterator end, string_view label = {},
           axis::uoflow uo = axis::uoflow::oflow)
      : base(std::distance(begin, end), label,
             uo == axis::uoflow::on ? axis::uoflow::oflow : uo),
        map_(new map_type()) {
    int index = 0;
    while (begin != end) map_->insert({*begin++, index++});
    if (index == 0) throw std::invalid_argument("iterator range is empty");
  }

  /// Returns the bin index for the passed argument.
  int index(const value_type& x) const noexcept {
    auto it = map_->left.find(x);
    if (it == map_->left.end()) return base::size();
    return it->second;
  }

  /// Returns the value for the bin index (performs a range check).
  const value_type& value(int idx) const {
    auto it = map_->right.find(idx);
    if (it == map_->right.end())
      throw std::out_of_range("category index out of range");
    return it->second;
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const category& o) const noexcept {
    return base::operator==(o) &&
           std::equal(map_->begin(), map_->end(), o.map_->begin());
  }

private:
  std::unique_ptr<map_type> map_;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
