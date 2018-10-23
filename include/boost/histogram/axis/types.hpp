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
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mp11.hpp>
#include <cmath>
#include <limits>
#include <memory>
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

  constexpr bool operator==(const identity&) const noexcept { return true; }
  template <class Archive>
  void serialize(Archive&, unsigned) {}  // noop
};

template <typename T>
struct log : public identity<T> {
  static T forward(T x) { return std::log(x); }
  static T inverse(T x) { return std::exp(x); }
};

template <typename T>
struct sqrt : public identity<T> {
  static T forward(T x) { return std::sqrt(x); }
  static T inverse(T x) { return x * x; }
};

template <typename T>
struct pow {
  T power = 1.0;

  pow() = default;
  pow(T p) : power(p) {}

  T forward(T v) const { return std::pow(v, power); }
  T inverse(T v) const { return std::pow(v, 1.0 / power); }

  bool operator==(const pow& o) const noexcept { return power == o.power; }
  template <class Archive>
  void serialize(Archive&, unsigned);
};

template <typename Quantity, typename Unit>
struct quantity {
  Unit unit;

  quantity(const Unit& u) : unit(u) {}

  using Dimensionless =
      decltype(std::declval<Quantity&>() / std::declval<Unit&>());

  Dimensionless forward(Quantity x) const { return x / unit; }
  Quantity inverse(Dimensionless x) const { return x * unit; }

  bool operator==(const quantity& o) const noexcept { return unit == o.unit; }
  template <class Archive>
  void serialize(Archive&, unsigned);
};
}  // namespace transform

/** Axis for equidistant intervals on the real line.
 *
 * The most common binning strategy.
 * Very fast. Binning is a O(1) operation.
 */
template <typename Transform, typename MetaData>
class regular : public base<MetaData>,
                public iterator_mixin<regular<Transform, MetaData>> {
  using base_type = base<MetaData>;
  using transform_type = Transform;
  using value_type = detail::arg_type<-1, decltype(&transform_type::forward)>;
  using internal_type = detail::return_type<decltype(&transform_type::forward)>;
  static_assert(std::is_floating_point<internal_type>::value,
                "type returned by forward transform must be floating point");
  using metadata_type = MetaData;
  struct data : transform_type  // empty base class optimization
  {
    internal_type min = 0, delta = 1;

    data(const transform_type& t, unsigned n, value_type b, value_type e)
        : transform_type(t),
          min(this->forward(b)),
          delta((this->forward(e) - this->forward(b)) / n) {}

    data() = default;

    bool operator==(const data& rhs) const noexcept {
      return transform_type::operator==(rhs) && min == rhs.min &&
             delta == rhs.delta;
    }
  };
  using bin_type = interval_view<regular>;

 public:
  /** Construct axis with n bins over real range [begin, end).
   *
   * \param n        number of bins.
   * \param start    low edge of first bin.
   * \param stop     high edge of last bin.
   * \param metadata description of the axis.
   * \param options  extra bin options.
   * \param trans    transform instance to use.
   */
  regular(unsigned n, value_type start, value_type stop,
          metadata_type m = metadata_type(),
          option_type o = option_type::underflow_and_overflow,
          transform_type trans = transform_type())
      : base_type(n, std::move(m), o), data_(std::move(trans), n, start, stop) {
    if (!std::isfinite(data_.min) || !std::isfinite(data_.delta))
      throw std::invalid_argument(
          "forward transform of lower or upper invalid");
  }

  regular() = default;

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    // Runs in hot loop, please measure impact of changes
    const auto z = (data_.forward(x) - data_.min) / data_.delta;
    if (z < base_type::size()) {
      if (z >= 0)
        return static_cast<int>(z);
      else
        return -1;
    }
    return base_type::size();  // also returned if z is NaN

    // const auto lt_max = z < base_type::size();
    // const auto ge_zero = z >= 0;
    // return lt_max * (ge_zero * static_cast<int>(z) - !ge_zero) + !lt_max *
    // base_type::size();
  }

  /// Returns lower edge of bin.
  value_type lower(int idx) const noexcept {
    const auto z = internal_type(idx) / base_type::size();
    internal_type x;
    if (z < 0)
      x = -std::numeric_limits<internal_type>::infinity();
    else if (z > 1)
      x = std::numeric_limits<internal_type>::infinity();
    else {
      x = (1 - z) * data_.min +
          z * (data_.min + data_.delta * base_type::size());
    }
    return data_.inverse(x);
  }

  const transform_type& transform() const { return data_; }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const regular& o) const noexcept {
    return base_type::operator==(o) && data_ == o.data_;
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

 private:
  data data_;
};

/** Axis for real values on a circle.
 *
 * The axis is circular and wraps around reaching the
 * perimeter value. Therefore, there are no overflow/underflow
 * bins for this axis. Binning is a O(1) operation.
 */
template <typename RealType, typename MetaData>
class circular : public base<MetaData>,
                 public iterator_mixin<circular<RealType, MetaData>> {
  using base_type = base<MetaData>;
  using value_type = RealType;
  using metadata_type = MetaData;
  using bin_type = interval_view<circular>;

 public:
  // two_pi can be found in boost/math, but it is defined here to reduce deps
  static constexpr value_type two_pi() { return 6.283185307179586; }

  /** Constructor for n bins with an optional offset.
   *
   * \param n         number of bins.
   * \param phase     starting phase.
   * \param perimeter range after which value wraps around.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   */
  explicit circular(unsigned n, value_type phase = 0.0,
                    value_type perimeter = two_pi(),
                    metadata_type m = metadata_type(),
                    option_type o = option_type::overflow)
      : base_type(n, std::move(m),
                  o == option_type::underflow_and_overflow
                      ? option_type::overflow
                      : o),
        phase_(phase),
        delta_(perimeter / n) {
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

  /// Returns lower edge of bin.
  value_type lower(int i) const noexcept { return phase_ + i * delta_; }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const circular& o) const noexcept {
    return base_type::operator==(o) && phase_ == o.phase_ && delta_ == o.delta_;
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

 private:
  value_type phase_ = 0.0, delta_ = 1.0;
};

/** Axis for non-equidistant bins on the real line.
 *
 * Binning is a O(log(N)) operation. If speed matters and the problem
 * domain allows it, prefer a regular axis, possibly with a transform.
 */
template <typename RealType, typename Allocator, typename MetaData>
class variable
    : public base<MetaData>,
      public iterator_mixin<variable<RealType, Allocator, MetaData>> {
  using base_type = base<MetaData>;
  using value_type = RealType;
  using allocator_type = Allocator;
  using metadata_type = MetaData;
  using bin_type = interval_view<variable>;

  struct data : allocator_type  // empty base class optimization
  {
    typename std::allocator_traits<allocator_type>::pointer x = nullptr;

    using allocator_type::allocator_type;

    data(const allocator_type& a) : allocator_type(a) {}
    data() = default;

    friend void swap(data& a, data& b) noexcept {
      std::swap(a.x, b.x);
      auto tmp = static_cast<allocator_type&&>(a);
      a = static_cast<allocator_type&&>(b);
      b = static_cast<allocator_type&&>(tmp);
    }
  };

 public:
  /** Construct an axis from iterator range of bin edges.
   *
   * \param begin     begin of edge sequence.
   * \param end       end of edge sequence.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename Iterator, typename = detail::requires_iterator<Iterator>>
  variable(Iterator begin, Iterator end, metadata_type m = metadata_type(),
           option_type o = option_type::underflow_and_overflow,
           allocator_type a = allocator_type())
      : base_type(begin == end ? 0 : std::distance(begin, end) - 1,
                  std::move(m), o),
        data_(std::move(a)) {
    using AT = std::allocator_traits<allocator_type>;
    data_.x = AT::allocate(data_, nx());
    try {
      auto xit = data_.x;
      try {
        AT::construct(data_, xit, *begin++);
        while (begin != end) {
          if (*begin <= *xit) {
            ++xit;  // to make sure catch code works
            throw std::invalid_argument(
                "input sequence must be strictly ascending");
          }
          ++xit;
          AT::construct(data_, xit, *begin++);
        }
      } catch (...) {
        // release resources that were already acquired before rethrowing
        while (xit != data_.x) AT::destroy(data_, --xit);
        throw;
      }
    } catch (...) {
      // release resources that were already acquired before rethrowing
      AT::deallocate(data_, data_.x, nx());
      throw;
    }
  }

  /** Construct an axis from iterable range of bin edges.
   *
   * \param iterable  iterable range of bin edges.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename T, typename = detail::requires_iterable<T>>
  variable(const T& t, metadata_type m = metadata_type(),
           option_type o = option_type::underflow_and_overflow,
           allocator_type a = allocator_type())
      : variable(std::begin(t), std::end(t), std::move(m), o, std::move(a)) {}

  template <typename T>
  variable(std::initializer_list<T> t, metadata_type m = metadata_type(),
           option_type o = option_type::underflow_and_overflow,
           allocator_type a = allocator_type())
      : variable(t.begin(), t.end(), std::move(m), o, std::move(a)) {}

  variable() = default;

  variable(const variable& o) : base_type(o), data_(o.data_) {
    data_.x = detail::create_buffer_from_iter(data_, nx(), o.data_.x);
  }

  variable& operator=(const variable& o) {
    if (this != &o) {
      if (base_type::size() != o.size()) {
        detail::destroy_buffer(data_, data_.x, nx());
        data_ = o.data_;
        base_type::operator=(o);
        data_.x = detail::create_buffer_from_iter(data_, nx(), o.data_.x);
      } else {
        base_type::operator=(o);
        std::copy(o.data_.x, o.data_.x + nx(), data_.x);
      }
    }
    return *this;
  }

  variable(variable&& o) : base_type(std::move(o)), data_(std::move(o.data_)) {
    o.data_.x = nullptr;
  }

  variable& operator=(variable&& o) {
    if (this != &o) {
      std::swap(static_cast<base_type&>(*this), static_cast<base_type&>(o));
      std::swap(data_, o.data_);
    }
    return *this;
  }

  ~variable() { detail::destroy_buffer(data_, data_.x, nx()); }

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    return std::upper_bound(data_.x, data_.x + nx(), x) - data_.x - 1;
  }

  /// Returns the starting edge of the bin.
  value_type lower(int i) const noexcept {
    if (i < 0) {
      return -std::numeric_limits<value_type>::infinity();
    }
    if (i > static_cast<int>(base_type::size())) {
      return std::numeric_limits<value_type>::infinity();
    }
    return data_.x[i];
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const variable& o) const noexcept {
    return base_type::operator==(o) &&
           std::equal(data_.x, data_.x + nx(), o.data_.x);
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

 private:
  int nx() const { return base_type::size() + 1; }
  data data_;
};

/** Axis for an interval of integral values with unit steps.
 *
 * Binning is a O(1) operation. This axis operates
 * faster than a regular.
 */
template <typename IntType, typename MetaData>
class integer : public base<MetaData>,
                public iterator_mixin<integer<IntType, MetaData>> {
  using base_type = base<MetaData>;
  using value_type = IntType;
  using metadata_type = MetaData;
  using bin_type = interval_view<integer>;

 public:
  /** Construct axis over a semi-open integer interval [begin, end).
   *
   * \param begin    first integer of covered range.
   * \param end      one past last integer of covered range.
   * \param metadata description of the axis.
   * \param options  extra bin options.
   */
  integer(value_type begin, value_type end, metadata_type m = metadata_type(),
          option_type o = option_type::underflow_and_overflow)
      : base_type(end - begin, std::move(m), o), min_(begin) {
    if (begin >= end) {
      throw std::invalid_argument("begin < end required");
    }
  }

  integer() = default;
  integer(const integer&) = default;
  integer& operator=(const integer&) = default;
  integer(integer&&) = default;
  integer& operator=(integer&&) = default;

  /// Returns the bin index for the passed argument.
  int operator()(value_type x) const noexcept {
    const int z = x - min_;
    return z >= 0 ? (z > static_cast<int>(base_type::size()) ? base_type::size()
                                                             : z)
                  : -1;
  }

  /// Returns lower edge of the integral bin.
  value_type lower(int i) const noexcept {
    if (i < 0) {
      return std::numeric_limits<value_type>::min();
    }
    if (i > static_cast<int>(base_type::size())) {
      return std::numeric_limits<value_type>::max();
    }
    return min_ + i;
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const integer& o) const noexcept {
    return base_type::operator==(o) && min_ == o.min_;
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

 private:
  value_type min_ = 0;
};

/** Axis which maps unique values to bins (one on one).
 *
 * The axis maps a set of values to bins, following the order of
 * arguments in the constructor. There is an optional overflow bin
 * for this axis, which counts values that are not part of the set.
 * Binning is a O(n) operation for n values in the worst case and O(1) in
 * the best case. The value types must be equal-comparable.
 */
template <typename ValueType, typename Allocator, typename MetaData>
class category
    : public base<MetaData>,
      public iterator_mixin<category<ValueType, Allocator, MetaData>> {
  using base_type = base<MetaData>;
  using metadata_type = MetaData;
  using value_type = ValueType;
  using allocator_type = Allocator;
  using bin_type = value_view<category>;

  struct data : allocator_type {
    typename std::allocator_traits<allocator_type>::pointer x = nullptr;
    using allocator_type::allocator_type;
    data(const allocator_type& a) : allocator_type(a) {}
    data() = default;

    friend void swap(data& a, data& b) noexcept {
      std::swap(a.x, b.x);
      auto tmp = static_cast<allocator_type&&>(a);
      a = static_cast<allocator_type&&>(b);
      b = static_cast<allocator_type&&>(tmp);
    }
  };

 public:
  /** Construct an axis from iterator range of categories.
   *
   * \param begin     begin of category range of unique values.
   * \param end       end of category range of unique values.
   * \param metadata  description of the axis.
   * \param options   extra bin options.
   * \param allocator allocator instance to use.
   */
  template <typename Iterator, typename = detail::requires_iterator<Iterator>>
  category(Iterator begin, Iterator end, metadata_type m = metadata_type(),
           option_type o = option_type::overflow,
           allocator_type a = allocator_type())
      : base_type(std::distance(begin, end), std::move(m), o),
        data_(std::move(a)) {
    data_.x = detail::create_buffer_from_iter(data_, base_type::size(), begin);
  }

  /** Construct from an initializer list of strings.
   *
   * \param seq sequence of unique values.
   * \param metadata description of the axis.
   */
  template <typename T, typename = detail::requires_iterable<T>>
  category(const T& t, metadata_type m = metadata_type(),
           option_type o = option_type::overflow,
           allocator_type a = allocator_type())
      : category(std::begin(t), std::end(t), std::move(m), o, std::move(a)) {}

  template <typename T>
  category(std::initializer_list<T> t, metadata_type m = metadata_type(),
           option_type o = option_type::overflow,
           allocator_type a = allocator_type())
      : category(t.begin(), t.end(), std::move(m), o, std::move(a)) {}

  category() = default;

  category(const category& o) : base_type(o), data_(o.data_) {
    data_.x =
        detail::create_buffer_from_iter(data_, base_type::size(), o.data_.x);
  }

  category& operator=(const category& o) {
    if (this != &o) {
      if (base_type::size() != o.size()) {
        detail::destroy_buffer(data_, data_.x, base_type::size());
        base_type::operator=(o);
        data_ = o.data_;
        data_.x = detail::create_buffer_from_iter(data_, base_type::size(),
                                                  o.data_.x);
      } else {
        base_type::operator=(o);
        std::copy(o.data_.x, o.data_.x + base_type::size(), data_.x);
      }
    }
    return *this;
  }

  category(category&& o) : base_type(std::move(o)), data_(std::move(o.data_)) {
    o.data_.x = nullptr;
  }

  category& operator=(category&& o) {
    if (this != &o) {
      std::swap(static_cast<base_type&>(*this), static_cast<base_type&>(o));
      std::swap(data_, o.data_);
    }
    return *this;
  }

  ~category() { detail::destroy_buffer(data_, data_.x, base_type::size()); }

  /// Returns the bin index for the passed argument.
  int operator()(const value_type& x) const noexcept {
    const auto begin = data_.x;
    const auto end = begin + base_type::size();
    return std::distance(begin, std::find(begin, end, x));
  }

  /// Returns the value for the bin index (performs a range check).
  const value_type& value(int idx) const {
    if (idx < 0 || idx >= static_cast<int>(base_type::size()))
      throw std::out_of_range("category index out of range");
    return data_.x[idx];
  }

  bin_type operator[](int idx) const noexcept { return bin_type(idx, *this); }

  bool operator==(const category& o) const noexcept {
    return base_type::operator==(o) &&
           std::equal(data_.x, data_.x + base_type::size(), o.data_.x);
  }

  template <class Archive>
  void serialize(Archive&, unsigned);

 private:
  data data_;
};
}  // namespace axis
}  // namespace histogram
}  // namespace boost

#endif
