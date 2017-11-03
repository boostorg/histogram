// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_HPP_
#define _BOOST_HISTOGRAM_AXIS_HPP_

#include <algorithm>
#include <boost/bimap.hpp>
#include <boost/histogram/interval.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/operators.hpp>
#include <boost/utility/string_view.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
} // namespace serialization
} // namespace boost

namespace boost {
namespace histogram {

namespace axis {

enum { with_uoflow = true, without_uoflow = false };

namespace detail {
// similar to boost::reference_wrapper, but with default ctor
template <typename T> class cref {
public:
  cref() = default;
  cref(const T &t) : ptr_(&t) {}
  operator const T &() const { return *ptr_; }

private:
  const T *ptr_ = nullptr;
};

template <typename Axis>
using axis_iterator_value = std::pair<
    int, typename conditional<
             is_reference<typename Axis::bin_type>::value,
             cref<typename remove_reference<typename Axis::bin_type>::type>,
             typename Axis::bin_type>::type>;
} // namespace detail

template <typename Axis>
class axis_iterator : public iterator_facade<axis_iterator<Axis>,
                                             detail::axis_iterator_value<Axis>,
                                             random_access_traversal_tag> {
  using value_type = detail::axis_iterator_value<Axis>;

public:
  explicit axis_iterator(const Axis &axis, int idx) : axis_(axis) {
    value_.first = idx;
  }

private:
  void increment() noexcept { ++value_.first; }
  void decrement() noexcept { --value_.first; }
  void advance(int n) noexcept { value_.first += n; }
  int distance_to(const axis_iterator &other) const noexcept {
    return other.value_.first - value_.first;
  }
  bool equal(const axis_iterator &other) const noexcept {
    return value_.first == other.value_.first;
  }
  value_type &dereference() const {
    value_.second = axis_[value_.first];
    return value_;
  }
  const Axis &axis_;
  mutable value_type value_;
  friend class boost::iterator_core_access;
};

/// Common base class for all axes.
template <bool UOFlow> class axis_base;

/// Specialization with overflow/underflow bins.
template <> class axis_base<with_uoflow> {
public:
  /// Returns the number of bins, excluding overflow/underflow.
  inline int size() const { return size_; }
  /// Returns the number of bins, including overflow/underflow.
  inline int shape() const { return shape_; }
  /// Returns whether axis has extra overflow and underflow bins.
  inline bool uoflow() const { return shape_ > size_; }
  /// Returns the axis label, which is a name or description.
  string_view label() const { return label_; }
  /// Change the label of an axis.
  void label(string_view label) { label_.assign(label.begin(), label.end()); }

protected:
  axis_base(unsigned n, string_view label, bool uoflow)
      : size_(n), shape_(size_ + 2 * static_cast<int>(uoflow)),
        label_(label.begin(), label.end()) {
    if (n == 0) {
      throw std::logic_error("bins > 0 required");
    }
  }

  axis_base() = default;
  axis_base(const axis_base &) = default;
  axis_base &operator=(const axis_base &) = default;
  axis_base(axis_base &&other)
      : size_(other.size_), shape_(other.shape_),
        label_(std::move(other.label_)) {
    other.size_ = 0;
    other.shape_ = 0;
  }
  axis_base &operator=(axis_base &&other) {
    if (this != &other) {
      size_ = other.size_;
      shape_ = other.shape_;
      label_ = std::move(other.label_);
      other.size_ = 0;
      other.shape_ = 0;
    }
    return *this;
  }

  bool operator==(const axis_base &o) const {
    return size_ == o.size_ && shape_ == o.shape_ && label_ == o.label_;
  }

private:
  int size_ = 0;
  int shape_ = 0;
  std::string label_;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/// Specialization without overflow/underflow bins.
template <> class axis_base<without_uoflow> {
public:
  /// Returns the number of bins, excluding overflow/underflow.
  inline int size() const { return size_; }
  /// Returns the number of bins, including overflow/underflow.
  inline int shape() const { return size_; }
  /// Returns whether axis has extra overflow and underflow bins.
  inline bool uoflow() const { return false; }
  /// Returns the axis label, which is a name or description.
  string_view label() const { return label_; }
  /// Change the label of an axis.
  void label(string_view label) { label_.assign(label.begin(), label.end()); }

protected:
  axis_base(unsigned n, string_view label)
      : size_(n), label_(label.begin(), label.end()) {
    if (n == 0) {
      throw std::logic_error("bins > 0 required");
    }
    std::copy(label.begin(), label.end(), label_.begin());
  }

  axis_base() = default;
  axis_base(const axis_base &) = default;
  axis_base &operator=(const axis_base &) = default;
  axis_base(axis_base &&other)
      : size_(other.size_), label_(std::move(other.label_)) {
    other.size_ = 0;
  }
  axis_base &operator=(axis_base &&other) {
    if (this != &other) {
      size_ = other.size_;
      label_ = std::move(other.label_);
      other.size_ = 0;
    }
    return *this;
  }

  bool operator==(const axis_base &other) const {
    return size_ == other.size_ && label_ == other.label_;
  }

private:
  int size_ = 0;
  std::string label_;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

namespace transform {
struct identity {
  template <typename T> static T forward(T v) { return v; }
  template <typename T> static T inverse(T v) { return v; }
  bool operator==(const identity &) const { return true; }
  template <class Archive> void serialize(Archive &, unsigned) {}
};

struct log {
  template <typename T> static T forward(T v) { return std::log(v); }
  template <typename T> static T inverse(T v) { return std::exp(v); }
  bool operator==(const log &) const { return true; }
  template <class Archive> void serialize(Archive &, unsigned) {}
};

struct sqrt {
  template <typename T> static T forward(T v) { return std::sqrt(v); }
  template <typename T> static T inverse(T v) { return v * v; }
  bool operator==(const sqrt &) const { return true; }
  template <class Archive> void serialize(Archive &, unsigned) {}
};

struct cos {
  template <typename T> static T forward(T v) { return std::cos(v); }
  template <typename T> static T inverse(T v) { return std::acos(v); }
  bool operator==(const cos &) const { return true; }
  template <class Archive> void serialize(Archive &, unsigned) {}
};

struct pow {
  pow() = default;
  pow(double exponent) : value(exponent) {}
  template <typename T> T forward(T v) const { return std::pow(v, value); }
  template <typename T> T inverse(T v) const {
    return std::pow(v, 1.0 / value);
  }
  double value = 1.0;
  bool operator==(const pow &other) const { return value == other.value; }
  template <class Archive> void serialize(Archive &, unsigned);
};
} // namespace transform

/** Axis for binning real-valued data into equidistant bins.
 *
 * The simplest and common binning strategy.
 * Very fast. Binning is a O(1) operation.
 */
template <typename RealType = double, typename Transform = transform::identity>
class regular : public axis_base<with_uoflow>,
                boost::operators<regular<RealType, Transform>> {
public:
  using value_type = RealType;
  using bin_type = interval<value_type>;
  using transform_type = Transform;
  using const_iterator = axis_iterator<regular>;

  /** Construct axis with n bins over range [min, max).
   *
   * \param n number of bins.
   * \param min low edge of first bin.
   * \param max high edge of last bin.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   * \param trans arguments passed to the transform.
   */
  regular(unsigned n, value_type min, value_type max,
          string_view label = string_view(), bool uoflow = true,
          transform_type trans = transform_type())
      : axis_base<with_uoflow>(n, label, uoflow), min_(trans.forward(min)),
        delta_((trans.forward(max) - trans.forward(min)) / n), trans_(trans) {
    if (!(min < max)) {
      throw std::logic_error("min < max required");
    }
    BOOST_ASSERT(!std::isnan(min_));
    BOOST_ASSERT(!std::isnan(delta_));
  }

  regular() = default;
  regular(const regular &) = default;
  regular &operator=(const regular &) = default;
  regular(regular &&) = default;
  regular &operator=(regular &&) = default;

  /// Returns the bin index for the passed argument.
  inline int index(value_type x) const noexcept {
    // Optimized code
    const value_type z = (trans_.forward(x) - min_) / delta_;
    return z >= 0.0 ? (z > size() ? size() : static_cast<int>(z)) : -1;
  }

  /// Returns the starting edge of the bin.
  bin_type operator[](int idx) const noexcept {
    auto eval = [this](int i) {
      const auto n = size();
      if (i < 0)
        return trans_.inverse(-std::numeric_limits<value_type>::infinity());
      if (i > n)
        return trans_.inverse(std::numeric_limits<value_type>::infinity());
      const auto z = value_type(i) / n;
      return trans_.inverse((1.0 - z) * min_ + z * (min_ + delta_ * n));
    };
    return {eval(idx), eval(idx + 1)};
  }

  bool operator==(const regular &o) const noexcept {
    return axis_base<with_uoflow>::operator==(o) && min_ == o.min_ &&
           delta_ == o.delta_ && trans_ == o.trans_;
  }

  const_iterator begin() const noexcept {
    return const_iterator(*this, uoflow() ? -1 : 0);
  }

  const_iterator end() const noexcept {
    return const_iterator(*this, uoflow() ? size() + 1 : size());
  }

  const transform_type &transform() const noexcept { return trans_; }

private:
  value_type min_ = 0.0, delta_ = 1.0;
  transform_type trans_;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/** Axis for real-valued angles.
 *
 * The axis is circular and wraps around reaching the
 * perimeter value. Therefore, there are no overflow/underflow
 * bins for this axis. Binning is a O(1) operation.
 */
template <typename RealType = double>
class circular : public axis_base<without_uoflow>,
                 boost::operators<regular<RealType>> {
public:
  using value_type = RealType;
  using bin_type = interval<value_type>;
  using const_iterator = axis_iterator<circular>;

  /** Constructor for n bins with an optional offset.
   *
   * \param n         number of bins.
   * \param phase     starting phase.
   * \param perimeter range after which value wraps around.
   * \param label     description of the axis.
   */
  explicit circular(unsigned n, value_type phase = 0.0,
                    value_type perimeter = math::double_constants::two_pi,
                    string_view label = string_view())
      : axis_base<without_uoflow>(n, label), phase_(phase),
        perimeter_(perimeter) {}

  circular() = default;
  circular(const circular &) = default;
  circular &operator=(const circular &) = default;
  circular(circular &&) = default;
  circular &operator=(circular &&) = default;

  /// Returns the bin index for the passed argument.
  inline int index(value_type x) const noexcept {
    const value_type z = (x - phase_) / perimeter_;
    const int i = static_cast<int>(std::floor(z * size())) % size();
    return i + (i < 0) * size();
  }

  /// Returns the starting edge of the bin.
  bin_type operator[](int idx) const {
    auto eval = [this](int i) {
      const value_type z = value_type(i) / size();
      return z * perimeter_ + phase_;
    };
    return {eval(idx), eval(idx + 1)};
  }

  bool operator==(const circular &o) const {
    return axis_base<without_uoflow>::operator==(o) && phase_ == o.phase_ &&
           perimeter_ == o.perimeter_;
  }

  value_type perimeter() const { return perimeter_; }
  value_type phase() const { return phase_; }

  const_iterator begin() const { return const_iterator(*this, 0); }

  const_iterator end() const { return const_iterator(*this, size()); }

private:
  value_type phase_ = 0.0, perimeter_ = 1.0;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/** An axis for real-valued data and bins of varying width.
 *
 * Binning is a O(log(N)) operation. If speed matters
 * and the problem domain allows it, prefer a regular.
 */
template <typename RealType = double>
class variable : public axis_base<with_uoflow>,
                 boost::operators<variable<RealType>> {
public:
  using value_type = RealType;
  using bin_type = interval<value_type>;
  using const_iterator = axis_iterator<variable>;

  /** Construct an axis from bin edges.
   *
   * \param x sequence of bin edges.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   */
  variable(std::initializer_list<value_type> x,
           string_view label = string_view(), bool uoflow = true)
      : axis_base<with_uoflow>(x.size() - 1, label, uoflow),
        x_(new value_type[x.size()]) {
    if (x.size() < 2) {
      throw std::logic_error("at least two values required");
    }
    std::copy(x.begin(), x.end(), x_.get());
    std::sort(x_.get(), x_.get() + size() + 1);
  }

  template <typename Iterator>
  variable(Iterator begin, Iterator end, string_view label = string_view(),
           bool uoflow = true)
      : axis_base<with_uoflow>(std::distance(begin, end) - 1, label, uoflow),
        x_(new value_type[std::distance(begin, end)]) {
    std::copy(begin, end, x_.get());
    std::sort(x_.get(), x_.get() + size() + 1);
  }

  variable() = default;
  variable(const variable &o)
      : axis_base<with_uoflow>(o), x_(new value_type[size() + 1]) {
    std::copy(o.x_.get(), o.x_.get() + size() + 1, x_.get());
  }
  variable &operator=(const variable &o) {
    if (this != &o) {
      axis_base<with_uoflow>::operator=(o);
      x_.reset(new value_type[size() + 1]);
      std::copy(o.x_.get(), o.x_.get() + size() + 1, x_.get());
    }
    return *this;
  }
  variable(variable &&) = default;
  variable &operator=(variable &&) = default;

  /// Returns the bin index for the passed argument.
  inline int index(value_type x) const noexcept {
    return std::upper_bound(x_.get(), x_.get() + size() + 1, x) - x_.get() - 1;
  }

  /// Returns the starting edge of the bin.
  bin_type operator[](int idx) const {
    auto eval = [this](int i) {
      if (i < 0) {
        return -std::numeric_limits<value_type>::infinity();
      }
      if (i > size()) {
        return std::numeric_limits<value_type>::infinity();
      }
      return x_[i];
    };
    return {eval(idx), eval(idx + 1)};
  }

  bool operator==(const variable &o) const {
    if (!axis_base<with_uoflow>::operator==(o)) {
      return false;
    }
    return std::equal(x_.get(), x_.get() + size() + 1, o.x_.get());
  }

  const_iterator begin() const {
    return const_iterator(*this, uoflow() ? -1 : 0);
  }

  const_iterator end() const {
    return const_iterator(*this, uoflow() ? size() + 1 : size());
  }

private:
  std::unique_ptr<value_type[]> x_; // smaller size compared to std::vector

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/** An axis for a contiguous range of integers.
 *
 * Binning is a O(1) operation. This axis operates
 * faster than a regular.
 */
template <typename IntType = int>
class integer : public axis_base<with_uoflow>,
                boost::operators<integer<IntType>> {
public:
  using value_type = IntType;
  using bin_type = interval<value_type>;
  using const_iterator = axis_iterator<integer>;

  /** Construct axis over integer range [min, max].
   *
   * \param min smallest integer of the covered range.
   * \param max largest integer of the covered range.
   */
  integer(value_type min, value_type max, string_view label = string_view(),
          bool uoflow = true)
      : axis_base<with_uoflow>(max - min, label, uoflow), min_(min) {
    if (min > max) {
      throw std::logic_error("min <= max required");
    }
  }

  integer() = default;
  integer(const integer &) = default;
  integer &operator=(const integer &) = default;
  integer(integer &&) = default;
  integer &operator=(integer &&) = default;

  /// Returns the bin index for the passed argument.
  inline int index(value_type x) const noexcept {
    const int z = x - min_;
    return z >= 0 ? (z > size() ? size() : z) : -1;
  }

  /// Returns the integer that is mapped to the bin index.
  bin_type operator[](int idx) const { return {min_ + idx, min_ + idx + 1}; }

  bool operator==(const integer &o) const {
    return axis_base<with_uoflow>::operator==(o) && min_ == o.min_;
  }

  const_iterator begin() const {
    return const_iterator(*this, uoflow() ? -1 : 0);
  }

  const_iterator end() const {
    return const_iterator(*this, uoflow() ? size() + 1 : size());
  }

private:
  value_type min_ = 0;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/** An axis for a set of unique values.
 *
 * The axis maps a set of values to bins, following the order of
 * arguments in the constructor. There is an optional overflow bin
 * for this axis, which counts values that are not part of the set.
 * Binning is a O(1) operation. The value type must be hashable.
 */
template <typename T = int>
class category : public axis_base<without_uoflow>,
                 boost::operators<category<T>> {
  using map_type = bimap<T, int>;

public:
  using value_type = T;
  using bin_type = const value_type &;
  using const_iterator = axis_iterator<category<T>>;

  category() = default;

  /** Construct from an initializer list of strings.
   *
   * \param seq sequence of unique values.
   */
  category(std::initializer_list<T> seq, string_view label = string_view())
      : axis_base<without_uoflow>(seq.size(), label) {
    int index = 0;
    for (const auto &x : seq)
      map_.insert({x, index++});
    if (index == 0)
      throw std::logic_error("sequence is empty");
  }

  template <typename Iterator>
  category(Iterator begin, Iterator end, string_view label = string_view())
      : axis_base<without_uoflow>(std::distance(begin, end), label) {
    int index = 0;
    while (begin != end)
      map_.insert({*begin++, index++});
    if (index == 0)
      throw std::logic_error("iterator range is empty");
  }

  /// Returns the bin index for the passed argument.
  /// Performs a range check.
  inline int index(const value_type &x) const noexcept {
    auto it = map_.left.find(x);
    if (it == map_.left.end())
      return size();
    return it->second;
  }

  /// Returns the value for the bin index.
  bin_type operator[](int idx) const {
    auto it = map_.right.find(idx);
    BOOST_ASSERT_MSG(it != map_.right.end(), "category index out of range");
    return it->second;
  }

  bool operator==(const category &other) const {
    return axis_base<without_uoflow>::operator==(other) &&
           std::equal(map_.begin(), map_.end(), other.map_.begin());
  }

  const_iterator begin() const { return const_iterator(*this, 0); }

  const_iterator end() const { return const_iterator(*this, size()); }

private:
  map_type map_;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

} // namespace axis

using builtin_axes =
    mpl::vector<axis::regular<>, axis::regular<double, axis::transform::log>,
                axis::regular<double, axis::transform::sqrt>,
                axis::regular<double, axis::transform::cos>,
                axis::regular<double, axis::transform::pow>, axis::circular<>,
                axis::variable<>, axis::integer<>, axis::category<>>;

} // namespace histogram
} // namespace boost

#endif
