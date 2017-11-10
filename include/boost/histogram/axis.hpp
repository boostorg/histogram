// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_HPP_
#define _BOOST_HISTOGRAM_AXIS_HPP_

#include <algorithm>
#include <boost/bimap.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/interval.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/utility/string_view.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/mpl/contains.hpp>
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

enum class uoflow { off = false, on = true };

template <typename Axis>
class axis_iterator
    : public iterator_facade<axis_iterator<Axis>,
                             std::pair<int, typename Axis::bin_type>,
                             random_access_traversal_tag,
                             std::pair<int, typename Axis::bin_type>> {
public:
  explicit axis_iterator(const Axis &axis, int idx) : axis_(axis), idx_(idx) {}

  axis_iterator(const axis_iterator &o) = default;
  axis_iterator &operator=(const axis_iterator &o) = default;

private:
  void increment() noexcept { ++idx_; }
  void decrement() noexcept { --idx_; }
  void advance(int n) noexcept { idx_ += n; }
  int distance_to(const axis_iterator &other) const noexcept {
    return other.idx_ - idx_;
  }
  bool equal(const axis_iterator &other) const noexcept {
    return idx_ == other.idx_;
  }
  std::pair<int, typename Axis::bin_type> dereference() const {
    return std::make_pair(idx_, axis_[idx_]);
  }
  const Axis& axis_;
  int idx_;
  friend class boost::iterator_core_access;
};

/// Base class for all axes.
class axis_base {
public:
  /// Returns the number of bins, excluding overflow/underflow.
  inline int size() const noexcept { return size_; }
  /// Returns the number of bins, including overflow/underflow.
  inline int shape() const noexcept { return size_; }
  /// Returns true if axis has extra overflow and underflow bins.
  inline bool uoflow() const noexcept { return false; }
  /// Returns the axis label, which is a name or description.
  string_view label() const noexcept { return label_; }
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
  axis_base(axis_base &&rhs) : size_(rhs.size_), label_(std::move(rhs.label_)) {
    rhs.size_ = 0;
  }
  axis_base &operator=(axis_base &&rhs) {
    if (this != &rhs) {
      size_ = rhs.size_;
      label_ = std::move(rhs.label_);
      rhs.size_ = 0;
    }
    return *this;
  }

  bool operator==(const axis_base &rhs) const noexcept {
    return size_ == rhs.size_ && label_ == rhs.label_;
  }

private:
  int size_ = 0;
  std::string label_;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/// Base class for axes with overflow/underflow bins.
class axis_base_uoflow : public axis_base {
public:
  /// Returns the number of bins, including overflow/underflow.
  inline int shape() const noexcept { return shape_; }
  /// Returns whether axis has extra overflow and underflow bins.
  inline bool uoflow() const noexcept { return shape_ > size(); }

protected:
  axis_base_uoflow(unsigned n, string_view label, enum uoflow uo)
      : axis_base(n, label), shape_(n + 2u * static_cast<unsigned>(uo)) {}

  axis_base_uoflow() = default;
  axis_base_uoflow(const axis_base_uoflow &) = default;
  axis_base_uoflow &operator=(const axis_base_uoflow &) = default;
  axis_base_uoflow(axis_base_uoflow &&rhs)
      : axis_base(std::move(rhs)), shape_(rhs.shape_) {
    rhs.shape_ = 0;
  }
  axis_base_uoflow &operator=(axis_base_uoflow &&rhs) {
    if (this != &rhs) {
      axis_base::operator=(std::move(rhs));
      shape_ = rhs.shape_;
      rhs.shape_ = 0;
    }
    return *this;
  }

  bool operator==(const axis_base_uoflow &rhs) const noexcept {
    return axis_base::operator==(rhs) && shape_ == rhs.shape_;
  }

private:
  int shape_ = 0;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

namespace transform {
namespace detail {
struct stateless {
  bool operator==(const stateless &) const noexcept { return true; }
  template <class Archive> void serialize(Archive &, unsigned) {}
};
} // namespace detail

struct identity : public detail::stateless {
  template <typename T> static T forward(T v) { return v; }
  template <typename T> static T inverse(T v) { return v; }
};

struct log : public detail::stateless {
  template <typename T> static T forward(T v) { return std::log(v); }
  template <typename T> static T inverse(T v) { return std::exp(v); }
};

struct sqrt : public detail::stateless {
  template <typename T> static T forward(T v) { return std::sqrt(v); }
  template <typename T> static T inverse(T v) { return v * v; }
};

struct cos : public detail::stateless {
  template <typename T> static T forward(T v) { return std::cos(v); }
  template <typename T> static T inverse(T v) { return std::acos(v); }
};

struct pow {
  pow() = default;
  pow(double exponent) : value(exponent) {}
  template <typename T> T forward(T v) const { return std::pow(v, value); }
  template <typename T> T inverse(T v) const {
    return std::pow(v, 1.0 / value);
  }
  double value = 1.0;
  bool operator==(const pow &other) const noexcept {
    return value == other.value;
  }
private:
  friend ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};
} // namespace transform

/** Axis for equidistant intervals on the real line.
 *
 * The most common binning strategy.
 * Very fast. Binning is a O(1) operation.
 */
template <typename RealType, typename Transform>
class regular : public axis_base_uoflow, Transform {
public:
  using value_type = RealType;
  using bin_type = interval<value_type>;
  using const_iterator = axis_iterator<regular>;

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
          string_view label = {},
          enum uoflow uo = ::boost::histogram::axis::uoflow::on,
          Transform trans = Transform())
      : axis_base_uoflow(n, label, uo), Transform(trans),
        min_(trans.forward(lower)),
        delta_((trans.forward(upper) - trans.forward(lower)) / n) {
    if (!(lower < upper)) {
      throw std::logic_error("lower < upper required");
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
    const value_type z = (this->forward(x) - min_) / delta_;
    return z >= 0.0 ? (z > size() ? size() : static_cast<int>(z)) : -1;
  }

  /// Returns the starting edge of the bin.
  bin_type operator[](int idx) const noexcept {
    auto eval = [this](int i) {
      const auto n = size();
      if (i < 0)
        return this->inverse(-std::numeric_limits<value_type>::infinity());
      if (i > n)
        return this->inverse(std::numeric_limits<value_type>::infinity());
      const auto z = value_type(i) / n;
      return this->inverse((1.0 - z) * min_ + z * (min_ + delta_ * n));
    };
    return {eval(idx), eval(idx + 1)};
  }

  bool operator==(const regular &o) const noexcept {
    return axis_base_uoflow::operator==(o) && Transform::operator==(o) &&
           min_ == o.min_ && delta_ == o.delta_;
  }

  const_iterator begin() const { return const_iterator(*this, 0); }

  const_iterator end() const { return const_iterator(*this, size()); }

  const Transform &transform() const noexcept {
    return static_cast<const Transform &>(*this);
  }

private:
  value_type min_ = 0.0, delta_ = 1.0;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/** Axis for real values on a circle.
 *
 * The axis is circular and wraps around reaching the
 * perimeter value. Therefore, there are no overflow/underflow
 * bins for this axis. Binning is a O(1) operation.
 */
template <typename RealType> class circular : public axis_base {
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
                    string_view label = {})
      : axis_base(n, label), phase_(phase), perimeter_(perimeter) {}

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

  bool operator==(const circular &o) const noexcept {
    return axis_base::operator==(o) && phase_ == o.phase_ &&
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

/** Axis for non-equidistant bins on the real line.
 *
 * Binning is a O(log(N)) operation. If speed matters and the problem
 * domain allows it, prefer a regular axis, possibly with a transform.
 */
template <typename RealType> class variable : public axis_base_uoflow {
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
  variable(std::initializer_list<value_type> x, string_view label = {},
           enum uoflow uo = ::boost::histogram::axis::uoflow::on)
      : axis_base_uoflow(x.size() - 1, label, uo),
        x_(new value_type[x.size()]) {
    if (x.size() < 2) {
      throw std::logic_error("at least two values required");
    }
    std::copy(x.begin(), x.end(), x_.get());
    std::sort(x_.get(), x_.get() + size() + 1);
  }

  template <typename Iterator>
  variable(Iterator begin, Iterator end, string_view label = {},
           enum uoflow uo = ::boost::histogram::axis::uoflow::on)
      : axis_base_uoflow(std::distance(begin, end) - 1, label, uo),
        x_(new value_type[std::distance(begin, end)]) {
    std::copy(begin, end, x_.get());
    std::sort(x_.get(), x_.get() + size() + 1);
  }

  variable() = default;
  variable(const variable &o)
      : axis_base_uoflow(o), x_(new value_type[size() + 1]) {
    std::copy(o.x_.get(), o.x_.get() + size() + 1, x_.get());
  }
  variable &operator=(const variable &o) {
    if (this != &o) {
      axis_base_uoflow::operator=(o);
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

  bool operator==(const variable &o) const noexcept {
    if (!axis_base_uoflow::operator==(o)) {
      return false;
    }
    return std::equal(x_.get(), x_.get() + size() + 1, o.x_.get());
  }

  const_iterator begin() const { return const_iterator(*this, 0); }

  const_iterator end() const { return const_iterator(*this, size()); }

private:
  std::unique_ptr<value_type[]> x_; // smaller size compared to std::vector

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/** Axis for an interval of integral values with unit steps.
 *
 * Binning is a O(1) operation. This axis operates
 * faster than a regular.
 */
template <typename IntType> class integer : public axis_base_uoflow {
public:
  using value_type = IntType;
  using bin_type = interval<value_type>;
  using const_iterator = axis_iterator<integer>;

  /** Construct axis over a semi-open integer interval [lower, upper).
   *
   * \param lower smallest integer of the covered range.
   * \param upper largest integer of the covered range.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   */
  integer(value_type lower, value_type upper, string_view label = {},
          enum uoflow uo = ::boost::histogram::axis::uoflow::on)
      : axis_base_uoflow(upper - lower, label, uo), min_(lower) {
    if (lower > upper) {
      throw std::logic_error("lower <= upper required");
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
  bin_type operator[](int idx) const {
    auto eval = [this](int i) {
      if (i < 0) {
        return -std::numeric_limits<value_type>::max();
      }
      if (i > size()) {
        return std::numeric_limits<value_type>::max();
      }
      return min_ + i;
    };
    return {eval(idx), eval(idx + 1)};
  }

  bool operator==(const integer &o) const noexcept {
    return axis_base_uoflow::operator==(o) && min_ == o.min_;
  }

  const_iterator begin() const { return const_iterator(*this, 0); }

  const_iterator end() const { return const_iterator(*this, size()); }

private:
  value_type min_ = 0;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

/** Axis which maps unique single values to bins (one on one).
 *
 * The axis maps a set of values to bins, following the order of
 * arguments in the constructor. There is an optional overflow bin
 * for this axis, which counts values that are not part of the set.
 * Binning is a O(1) operation. The value type must be hashable.
 */
template <typename T> class category : public axis_base {
  using map_type = bimap<T, int>;

public:
  using value_type = T;
  using bin_type = T;
  using const_iterator = axis_iterator<category<T>>;

  category() = default;
  category(const category &rhs)
      : axis_base(rhs), map_(new map_type(*rhs.map_)) {}
  category &operator=(const category &rhs) {
    if (this != &rhs) {
      axis_base::operator=(rhs);
      map_.reset(new map_type(*rhs.map_));
    }
    return *this;
  }
  category(category &&rhs) = default;
  category &operator=(category &&rhs) = default;

  /** Construct from an initializer list of strings.
   *
   * \param seq sequence of unique values.
   */
  category(std::initializer_list<T> seq, string_view label = {})
      : axis_base(seq.size(), label), map_(new map_type()) {
    int index = 0;
    for (const auto &x : seq)
      map_->insert({x, index++});
    if (index == 0)
      throw std::logic_error("sequence is empty");
  }

  template <typename Iterator>
  category(Iterator begin, Iterator end, string_view label = {})
      : axis_base(std::distance(begin, end), label), map_(new map_type()) {
    int index = 0;
    while (begin != end)
      map_->insert({*begin++, index++});
    if (index == 0)
      throw std::logic_error("iterator range is empty");
  }

  /// Returns the bin index for the passed argument.
  /// Performs a range check.
  inline int index(const value_type &x) const noexcept {
    auto it = map_->left.find(x);
    if (it == map_->left.end())
      return size();
    return it->second;
  }

  /// Returns the value for the bin index.
  bin_type operator[](int idx) const {
    auto it = map_->right.find(idx);
    if (it == map_->right.end())
      throw std::out_of_range("category index out of range");
    return it->second;
  }

  bool operator==(const category &o) const noexcept {
    return axis_base::operator==(o) &&
           std::equal(map_->begin(), map_->end(), o.map_->begin());
  }

  const_iterator begin() const { return const_iterator(*this, 0); }

  const_iterator end() const { return const_iterator(*this, size()); }

private:
  std::unique_ptr<map_type> map_;

  friend class ::boost::serialization::access;
  template <class Archive> void serialize(Archive &, unsigned);
};

namespace detail {
struct size : public static_visitor<int> {
  template <typename A> int operator()(const A &a) const { return a.size(); }
};

struct shape : public static_visitor<int> {
  template <typename A> int operator()(const A &a) const { return a.shape(); }
};

struct uoflow : public static_visitor<bool> {
  template <typename A> bool operator()(const A &a) const { return a.uoflow(); }
};

struct get_label : public static_visitor<string_view> {
  template <typename A> ::boost::string_view operator()(const A& a) const { return a.label(); }
};

struct set_label : public static_visitor<void> {
  const ::boost::string_view label;
  set_label(const ::boost::string_view x) : label(x) {}
  template <typename A> void operator()(A& a) const { a.label(label); }
};

template <typename T> struct index : public static_visitor<int> {
  const T &t;
  explicit index(const T &arg) : t(arg) {}
  template <typename Axis> int operator()(const Axis &a) const {
    return impl(std::is_convertible<T, typename Axis::value_type>(), a);
  }
  template <typename Axis> int impl(std::true_type, const Axis& a) const {
    return a.index(t);
  }
  template <typename Axis> int impl(std::false_type, const Axis&) const {
    throw std::runtime_error("index argument not convertible to axis value type");
  }
};

struct bin : public static_visitor<axis::interval<double>> {
  using double_interval = axis::interval<double>;
  const int i;
  bin(const int v) : i(v) {}
  template <typename A> double_interval operator()(const A &a) const {
    return impl(is_convertible<typename A::bin_type, double_interval>(),
                std::forward<typename A::bin_type>(a[i]));
  }
  template <typename B> double_interval impl(true_type, B &&b) const {
    return b;
  }
  template <typename B> double_interval impl(false_type, B &&) const {
    throw std::runtime_error("cannot convert bin_type to interval<double>");
  }
};
} // namespace detail

/// Polymorphic axis type
template <typename Axes = builtins>
class any : public make_variant_over<Axes>::type {
  using base_type = typename make_variant_over<Axes>::type;
public:
  using types = typename base_type::types;
  using value_type = double;
  using bin_type = interval<double>;
  using const_iterator = axis_iterator<any>;

  any() = default;
  any(const any& t) = default;
  any(any&& t) = default;
  any& operator=(const any& t) = default;
  any& operator=(any&& t) = default;

  template <typename T, typename = typename std::enable_if<
    mpl::contains<Axes, T>::value
  >::type>
  any(const T& t) : base_type(t) {}

  template <typename T, typename = typename std::enable_if<
    mpl::contains<Axes, T>::value
  >::type>
  any& operator=(const T& t) {
    // ugly workaround for compiler bug
    return reinterpret_cast<any&>(base_type::operator=(t));
  }

  template <typename T, typename = typename std::enable_if<
    mpl::contains<Axes, T>::value
  >::type>
  any& operator=(T&& t) {
    // ugly workaround for compiler bug
    return reinterpret_cast<any&>(base_type::operator=(std::move(t)));
  }

  int size() const {
    return apply_visitor(detail::size(), *this);
  }

  int shape() const {
    return apply_visitor(detail::shape(), *this);
  }

  bool uoflow() const {
    return apply_visitor(detail::uoflow(), *this);
  }

  // note: this only works for axes with compatible value type
  int index(const value_type x) const {
    return apply_visitor(detail::index<value_type>(x), *this);
  }

  string_view label() const {
    return apply_visitor(detail::get_label(), *this);
  }

  void label(const string_view x) {
    return apply_visitor(detail::set_label(x), *this);
  }

  // this only works for axes with compatible bin type
  // and will raise an error otherwise
  bin_type operator[](const int i) const {
    return apply_visitor(detail::bin(i), *this);
  }

  bool operator==(const any& rhs) const {
    return base_type::operator==(static_cast<const base_type&>(rhs));
  }

  const_iterator begin() const { return const_iterator(*this, 0); }

  const_iterator end() const { return const_iterator(*this, size()); }

private:
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive&, unsigned);
};

// dynamic casts
template <typename T, typename Axes>
typename std::add_lvalue_reference<T>::type cast(any<Axes>& any) { return get<T>(any); }

template <typename T, typename Axes>
const typename std::add_lvalue_reference<T>::type cast(const any<Axes>& any) { return get<T>(any); }

template <typename T, typename Axes>
typename std::add_pointer<T>::type cast(any<Axes>* any) { return get<T>(&any); }

template <typename T, typename Axes>
const typename std::add_pointer<T>::type cast(const any<Axes>* any) { return get<T>(&any); }

// pass-through versions for generic programming, i.e. when you switch to static histogram
template <typename T>
typename std::add_lvalue_reference<T>::type cast(T& t) { return t; }

template <typename T>
const typename std::add_lvalue_reference<T>::type cast(const T& t) { return t; }

template <typename T>
typename std::add_pointer<T>::type cast(T* t) { return t; }

template <typename T>
const typename std::add_pointer<T>::type cast(const T* t) { return t; }

} // namespace axis
} // namespace histogram
} // namespace boost

#endif
