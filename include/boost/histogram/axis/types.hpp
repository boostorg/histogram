// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_TYPES_HPP
#define BOOST_HISTOGRAM_AXIS_TYPES_HPP

#include <algorithm>
#include <boost/histogram/axis/base.hpp>
#include <boost/histogram/axis/interval_view.hpp>
#include <boost/histogram/axis/iterator.hpp>
#include <boost/histogram/axis/value_view.hpp>
#include <boost/histogram/detail/buffer.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

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
  bool operator==(const pow& other) const noexcept { return power == other.power; }

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
template <typename Transform, typename RealType, typename Allocator>
class regular : public labeled_base<Allocator>,
                public iterator_mixin<regular<Transform, RealType, Allocator>>,
                Transform {
  using base_type = labeled_base<Allocator>;

public:
  using allocator_type = typename base_type::allocator_type;
  using value_type = RealType;
  using bin_type = interval_view<regular>;
  using transform_type = Transform;

  /** Construct axis with n bins over real range [lower, upper).
   *
   * \param n number of bins.
   * \param lower low edge of first bin.
   * \param upper high edge of last bin.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   * \param trans arguments passed to the transform.
   */
  regular(unsigned n, value_type lower, value_type upper, string_view label = {},
          uoflow_type uo = uoflow_type::on, transform_type trans = transform_type(),
          const allocator_type& a = allocator_type())
      : base_type(n, uo, label, a)
      , transform_type(trans)
      , min_(trans.forward(lower))
      , delta_((trans.forward(upper) - trans.forward(lower)) / n) {
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
    const value_type z = (transform_type::forward(x) - min_) / delta_;
    return z < base_type::size() ? (z >= 0.0 ? static_cast<int>(z) : -1)
                                 : base_type::size();
  }

  /// Returns lower edge of bin.
  value_type lower(int i) const noexcept {
    const auto n = base_type::size();
    value_type x;
    if (i < 0)
      x = -std::numeric_limits<value_type>::infinity();
    else if (i > n)
      x = std::numeric_limits<value_type>::infinity();
    else {
      const auto z = value_type(i) / n;
      x = (1.0 - z) * min_ + z * (min_ + delta_ * n);
    }
    return transform_type::inverse(x);
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const regular& o) const noexcept {
    return base_type::operator==(o) && transform_type::operator==(o) && min_ == o.min_ &&
           delta_ == o.delta_;
  }

  /// Access properties of the transform.
  const transform_type& transform() const noexcept {
    return static_cast<const transform_type&>(*this);
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
template <typename RealType, typename Allocator>
class circular : public labeled_base<Allocator>,
                 public iterator_mixin<circular<RealType, Allocator>> {
  using base_type = labeled_base<Allocator>;

public:
  using allocator_type = typename base_type::allocator_type;
  using value_type = RealType;
  using bin_type = interval_view<circular>;

  // two_pi can be found in boost/math, but it is defined here to reduce deps
  static value_type two_pi() { return 6.283185307179586; }

  /** Constructor for n bins with an optional offset.
   *
   * \param n         number of bins.
   * \param phase     starting phase.
   * \param perimeter range after which value wraps around.
   * \param label     description of the axis.
   */
  explicit circular(unsigned n, value_type phase = 0.0, value_type perimeter = two_pi(),
                    string_view label = {}, const allocator_type& a = allocator_type())
      : base_type(n, uoflow_type::off, label, a), phase_(phase), perimeter_(perimeter) {
    if (perimeter <= 0)
      throw std::invalid_argument("perimeter must be positive");
  }

  circular() = default;
  circular(const circular&) = default;
  circular& operator=(const circular&) = default;
  circular(circular&&) = default;
  circular& operator=(circular&&) = default;

  /// Returns the bin index for the passed argument.
  int index(value_type x) const noexcept {
    const value_type z = (x - phase_) / perimeter_;
    const int i = static_cast<int>(std::floor(z * base_type::size())) % base_type::size();
    return i + (i < 0) * base_type::size();
  }

  /// Returns lower edge of bin.
  value_type lower(int i) const noexcept {
    const value_type z = value_type(i) / base_type::size();
    return z * perimeter_ + phase_;
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const circular& o) const noexcept {
    return base_type::operator==(o) && phase_ == o.phase_ && perimeter_ == o.perimeter_;
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
template <typename RealType, typename Allocator>
class variable : public labeled_base<Allocator>,
                 public iterator_mixin<variable<RealType, Allocator>> {
  using base_type = labeled_base<Allocator>;

public:
  using allocator_type = typename base_type::allocator_type;
  using value_type = RealType;
  using bin_type = interval_view<variable>;

private:
  using value_allocator_type =
      typename std::allocator_traits<allocator_type>::template rebind_alloc<value_type>;
  using value_pointer_type =
      typename std::allocator_traits<value_allocator_type>::pointer;

public:
  /** Construct an axis from bin edges.
   *
   * \param x sequence of bin edges.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   */
  variable(std::initializer_list<value_type> x, string_view label = {},
           uoflow_type uo = uoflow_type::on, const allocator_type& a = allocator_type())
      : variable(x.begin(), x.end(), label, uo, a) {}

  template <typename Iterator,
            typename = boost::histogram::detail::requires_iterator<Iterator>>
  variable(Iterator begin, Iterator end, string_view label = {},
           uoflow_type uo = uoflow_type::on, const allocator_type& a = allocator_type())
      : base_type(begin == end ? 0 : std::distance(begin, end) - 1, uo, label, a) {
    value_allocator_type a2(a);
    using AT = std::allocator_traits<value_allocator_type>;
    x_ = AT::allocate(a2, nx());
    auto xit = x_;
    try {
      AT::construct(a2, xit, *begin++);
      while (begin != end) {
        if (*begin <= *xit) {
          ++xit; // to make sure catch code works
          throw std::invalid_argument("input sequence must be strictly ascending");
        }
        ++xit;
        AT::construct(a2, xit, *begin++);
      }
    } catch (...) {
      // release resources that were already acquired before rethrowing
      while (xit != x_) AT::destroy(a2, --xit);
      AT::deallocate(a2, x_, nx());
      throw;
    }
  }

  variable() = default;

  variable(const variable& o) : base_type(o) {
    value_allocator_type a(o.get_allocator());
    x_ = boost::histogram::detail::create_buffer_from_iter(a, nx(), o.x_);
  }

  variable& operator=(const variable& o) {
    if (this != &o) {
      if (base_type::size() != o.size()) {
        this->~variable();
        base::operator=(o);
        value_allocator_type a(base_type::get_allocator());
        x_ = boost::histogram::detail::create_buffer_from_iter(a, nx(), o.x_);
      } else {
        base::operator=(o);
        std::copy(o.x_, o.x_ + o.nx(), x_);
      }
    }
    return *this;
  }

  variable(variable&& o) : base_type(std::move(o)) {
    x_ = o.x_;
    o.x_ = nullptr;
  }

  variable& operator=(variable&& o) {
    this->~variable();
    base::operator=(std::move(o));
    x_ = o.x_;
    o.x_ = nullptr;
    return *this;
  }

  ~variable() {
    if (x_) { // nothing to do for empty state
      value_allocator_type a(base_type::get_allocator());
      boost::histogram::detail::destroy_buffer(a, x_, nx());
    }
  }

  /// Returns the bin index for the passed argument.
  int index(value_type x) const noexcept {
    return std::upper_bound(x_, x_ + nx(), x) - x_ - 1;
  }

  /// Returns the starting edge of the bin.
  value_type lower(int i) const noexcept {
    if (i < 0) { return -std::numeric_limits<value_type>::infinity(); }
    if (i > base_type::size()) { return std::numeric_limits<value_type>::infinity(); }
    return x_[i];
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const variable& o) const noexcept {
    if (!base::operator==(o)) { return false; }
    return std::equal(x_, x_ + nx(), o.x_);
  }

private:
  int nx() const { return base_type::size() + 1; }

  value_pointer_type x_ = nullptr;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

/** Axis for an interval of integral values with unit steps.
 *
 * Binning is a O(1) operation. This axis operates
 * faster than a regular.
 */
template <typename IntType, typename Allocator>
class integer : public labeled_base<Allocator>,
                public iterator_mixin<integer<IntType, Allocator>> {
  using base_type = labeled_base<Allocator>;

public:
  using allocator_type = typename base_type::allocator_type;
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
          uoflow_type uo = uoflow_type::on, const allocator_type& a = allocator_type())
      : base_type(upper - lower, uo, label, a), min_(lower) {
    if (!(lower < upper)) { throw std::invalid_argument("lower < upper required"); }
  }

  integer() = default;
  integer(const integer&) = default;
  integer& operator=(const integer&) = default;
  integer(integer&&) = default;
  integer& operator=(integer&&) = default;

  /// Returns the bin index for the passed argument.
  int index(value_type x) const noexcept {
    const int z = x - min_;
    return z >= 0 ? (z > base_type::size() ? base_type::size() : z) : -1;
  }

  /// Returns lower edge of the integral bin.
  value_type lower(int i) const noexcept {
    if (i < 0) { return -std::numeric_limits<value_type>::max(); }
    if (i > base_type::size()) { return std::numeric_limits<value_type>::max(); }
    return min_ + i;
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const integer& o) const noexcept {
    return base_type::operator==(o) && min_ == o.min_;
  }

private:
  value_type min_ = 0;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};

/** Axis which maps unique values to bins (one on one).
 *
 * The axis maps a set of values to bins, following the order of
 * arguments in the constructor. There is an optional overflow bin
 * for this axis, which counts values that are not part of the set.
 * Binning is a O(n) operation for n values in the worst case and O(1) in
 * the best case. The value types must be equal-comparable.
 */
template <typename T, typename Allocator>
class category : public labeled_base<Allocator>,
                 public iterator_mixin<category<T, Allocator>> {
  using base_type = labeled_base<Allocator>;

public:
  using allocator_type = typename base_type::allocator_type;
  using value_type = T;
  using bin_type = value_view<category>;

private:
  using value_allocator_type =
      typename std::allocator_traits<allocator_type>::template rebind_alloc<value_type>;
  using value_pointer_type =
      typename std::allocator_traits<value_allocator_type>::pointer;

public:
  /** Construct from an initializer list of strings.
   *
   * \param seq sequence of unique values.
   * \param label description of the axis.
   * \param uoflow whether to add under-/overflow bins.
   */
  category(std::initializer_list<value_type> seq, string_view label = {},
           uoflow_type uo = uoflow_type::oflow,
           const allocator_type& a = allocator_type())
      : category(seq.begin(), seq.end(), label, uo, a) {}

  template <typename Iterator,
            typename = boost::histogram::detail::requires_iterator<Iterator>>
  category(Iterator begin, Iterator end, string_view label = {},
           uoflow_type uo = uoflow_type::oflow,
           const allocator_type& a = allocator_type())
      : base_type(std::distance(begin, end),
                  uo == uoflow_type::on ? uoflow_type::oflow : uo, label, a) {
    value_allocator_type a2(a);
    x_ = boost::histogram::detail::create_buffer_from_iter(a2, nx(), begin);
  }

  category() = default;

  category(const category& o) : base_type(o) {
    value_allocator_type a(o.get_allocator());
    x_ = boost::histogram::detail::create_buffer_from_iter(a, o.nx(), o.x_);
  }

  category& operator=(const category& o) {
    if (this != &o) {
      if (base_type::size() != o.size()) {
        this->~category();
        base_type::operator=(o);
        value_allocator_type a(base_type::get_allocator());
        x_ = boost::histogram::detail::create_buffer_from_iter(a, nx(), o.x_);
      } else {
        base_type::operator=(o);
        std::copy(o.x_, o.x_ + o.nx(), x_);
      }
    }
    return *this;
  }

  category(category&& o) : base_type(std::move(o)) {
    x_ = o.x_;
    o.x_ = nullptr;
  }

  category& operator=(category&& o) {
    this->~category();
    base_type::operator=(std::move(o));
    x_ = o.x_;
    o.x_ = nullptr;
    return *this;
  }

  ~category() {
    if (x_) { // nothing to do for empty state
      value_allocator_type a(base_type::get_allocator());
      boost::histogram::detail::destroy_buffer(a, x_, nx());
    }
  }

  /// Returns the bin index for the passed argument.
  int index(const value_type& x) const noexcept {
    if (last_ < nx() && x_[last_] == x) return last_;
    last_ = 0;
    for (auto xit = x_, xe = x_ + nx(); xit != xe && !(*xit == x); ++xit) ++last_;
    return last_;
  }

  /// Returns the value for the bin index (performs a range check).
  const value_type& value(int idx) const {
    if (idx < 0 || idx >= base_type::size())
      throw std::out_of_range("category index out of range");
    return x_[idx];
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const category& o) const noexcept {
    return base_type::operator==(o) && std::equal(x_, x_ + nx(), o.x_);
  }

private:
  int nx() const { return base_type::size(); }

  value_pointer_type x_ = nullptr;
  mutable int last_ = 0;

  friend class ::boost::serialization::access;
  template <class Archive>
  void serialize(Archive&, unsigned);
};
} // namespace axis
} // namespace histogram
} // namespace boost

#endif
