// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_HPP_
#define _BOOST_HISTOGRAM_AXIS_HPP_

#include <boost/math/constants/constants.hpp>
#include <boost/mpl/vector.hpp>
#include <type_traits>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace boost {
namespace histogram {

/// Common base class for most axes.
class axis_base {
public:
  /// Returns the number of bins, excluding overflow/underflow.
  inline int bins() const { return size_ ; }
  /// Returns the number of bins, including overflow/underflow.
  inline int shape() const { return shape_; }
  /// Returns whether axis has extra overflow and underflow bins.
  inline bool uoflow() const { return shape_ > size_; }
  /// Returns the axis label, which is a name or description (not implemented for category_axis).
  const std::string& label() const { return label_; }
  /// Change the label of an axis (not implemented for category_axis).
  void label(const std::string& label) { label_ = label; }

protected:
  axis_base(unsigned n, const std::string& label, bool uoflow) :
    size_(n), shape_(size_ + 2 * uoflow), label_(label)
  {
    if (n == 0)
      throw std::logic_error("bins > 0 required");
  }

  axis_base() = default;
  axis_base(const axis_base&) = default;
  axis_base& operator=(const axis_base&) = default;
  axis_base(axis_base&&) = default;
  axis_base& operator=(axis_base&&) = default;

  bool operator==(const axis_base& o) const
  { return size_ == o.size_ && shape_ == o.shape_ && label_ == o.label_; }

private:
  int size_;
  int shape_;
  std::string label_;

  template <class Archive>
  friend void serialize(Archive&, axis_base&, unsigned);
};

/// Mixin for real-valued axes.
template <typename RealType, typename Derived>
class real_axis {
public:
  /// Lower edge of the bin (left side).
  RealType left(int idx) const {
    return static_cast<const Derived&>(*this)[idx];
  }

  /// Upper edge of the bin (right side).
  RealType right(int idx) const {
    return static_cast<const Derived&>(*this)[idx + 1];
  }
};

/** Axis for binning real-valued data into equidistant bins.
  *
  * The simplest and common binning strategy.
  * Very fast. Binning is a O(1) operation.
  */
template <typename RealType=double>
class regular_axis: public axis_base,
                    public real_axis<RealType, regular_axis<RealType>> {
public:
  using value_type = RealType;

  /** Construct axis with n bins over range [min, max).
    *
    * \param n number of bins.
    * \param min low edge of first bin.
    * \param max high edge of last bin.
    * \param label description of the axis.
    * \param uoflow whether to add under-/overflow bins.
    */
  regular_axis(unsigned n, value_type min, value_type max,
               const std::string& label = std::string(),
               bool uoflow = true) :
    axis_base(n, label, uoflow),
    min_(min),
    delta_((max - min) / n)
  {
    if (!(min < max))
      throw std::logic_error("min < max required");
  }

  regular_axis() = default;
  regular_axis(const regular_axis&) = default;
  regular_axis& operator=(const regular_axis&) = default;
  regular_axis(regular_axis&&) = default;
  regular_axis& operator=(regular_axis&&) = default;

  /// Returns the bin index for the passed argument.
  inline int index(value_type x) const
  {
    // Optimized code
    const value_type z = (x - min_) / delta_;
    return z >= 0.0 ? (z > bins() ? bins() : static_cast<int>(z)) : -1;
  }

  /// Returns the starting edge of the bin.
  value_type operator[](int idx) const
  {
    if (idx < 0)
        return -std::numeric_limits<value_type>::infinity();
    if (idx > bins())
        return std::numeric_limits<value_type>::infinity();
    const value_type z = value_type(idx) / bins();
    return (1.0 - z) * min_ + z * (min_ + delta_ * bins());
  }

  bool operator==(const regular_axis& o) const
  {
    return axis_base::operator==(o) &&
           min_ == o.min_ &&
           delta_ == o.delta_;
  }

private:
  value_type min_ = 0.0, delta_ = 1.0;

  template <class Archive, typename RealType1>
  friend void serialize(Archive&, regular_axis<RealType1>&, unsigned);
};

/** Axis for real-valued angles.
  *
  * There are no overflow/underflow bins for this axis,
  * since the axis is circular and wraps around after
  * \f$2 \pi\f$.
  * Binning is a O(1) operation.
  */
template <typename RealType=double>
class circular_axis: public axis_base,
                     public real_axis<RealType, circular_axis<RealType>> {
public:
  using value_type = RealType;

  /** Constructor for n bins with an optional offset.
    *
    * \param n         number of bins.
    * \param phase     starting phase.
    * \param perimeter range after which value wraps around.
    * \param label     description of the axis.
 	  */
  explicit
  circular_axis(unsigned n, value_type phase = 0.0,
                value_type perimeter = math::double_constants::two_pi,
                const std::string& label = std::string()) :
    axis_base(n, label, false),
    phase_(phase), perimeter_(perimeter)
  {}

  circular_axis() = default;
  circular_axis(const circular_axis&) = default;
  circular_axis& operator=(const circular_axis&) = default;
  circular_axis(circular_axis&&) = default;
  circular_axis& operator=(circular_axis&&) = default;

  /// Returns the bin index for the passed argument.
  inline int index(value_type x) const {
    const value_type z = (x - phase_) / perimeter_;
    const int i = static_cast<int>(std::floor(z * bins())) % bins();
    return i + (i < 0) * bins();
  }

  /// Returns the starting edge of the bin.
  value_type operator[](int idx) const
  {
    const value_type z = value_type(idx) / bins();
    return z * perimeter_ + phase_;
  }

  bool operator==(const circular_axis& o) const
  {
    return axis_base::operator==(o) &&
           phase_ == o.phase_ &&
           perimeter_ == o.perimeter_;
  }

  value_type perimeter() const { return perimeter_; }
  value_type phase() const { return phase_; }

private:
  value_type phase_ = 0.0, perimeter_ = 1.0;

  template <class Archive, typename RealType1>
  friend void serialize(Archive&, circular_axis<RealType1>&, unsigned);
};

/** An axis for real-valued data and bins of varying width.
  *
  * Binning is a O(log(N)) operation. If speed matters
  * and the problem domain allows it, prefer a regular_axis.
  */
template <typename RealType=double>
class variable_axis : public axis_base,
                      public real_axis<RealType, variable_axis<RealType>> {
public:
  using value_type = RealType;

	/** Construct an axis from bin edges.
	  *
	  * \param x sequence of bin edges.
	  * \param label description of the axis.
	  * \param uoflow whether to add under-/overflow bins.
	  */
  explicit
  variable_axis(const std::initializer_list<value_type>& x,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(x.size() - 1, label, uoflow),
      x_(new value_type[x.size()])
  {
      if (x.size() < 2)
          throw std::logic_error("at least two values required");
      std::copy(x.begin(), x.end(), x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  variable_axis(const std::vector<value_type>& x,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(x.size() - 1, label, uoflow),
      x_(new value_type[x.size()])
  {
      std::copy(x.begin(), x.end(), x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  template <typename Iterator>
  variable_axis(Iterator begin, Iterator end,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(std::distance(begin, end) - 1, label, uoflow),
      x_(new value_type[std::distance(begin, end)])
  {
      std::copy(begin, end, x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  variable_axis() = default;
  variable_axis(const variable_axis& o) :
    axis_base(o),
    x_(new value_type[bins() + 1])
  {
    std::copy(o.x_.get(), o.x_.get() + bins() + 1, x_.get());
  }
  variable_axis& operator=(const variable_axis& o)
  {
    if (this != &o) {
        axis_base::operator=(o);
        x_.reset(new value_type[bins() + 1]);
        std::copy(o.x_.get(), o.x_.get() + bins() + 1, x_.get());
    }
    return *this;
  }
  variable_axis(variable_axis&&) = default;
  variable_axis& operator=(variable_axis&&) = default;

  /// Returns the bin index for the passed argument.
  inline int index(value_type x) const {
    return std::upper_bound(x_.get(), x_.get() + bins() + 1, x)
           - x_.get() - 1;
  }

  /// Returns the starting edge of the bin.
  value_type operator[](int idx) const
  {
    if (idx < 0)
        return -std::numeric_limits<value_type>::infinity();
    if (idx > bins())
        return std::numeric_limits<value_type>::infinity();
    return x_[idx];
  }

  bool operator==(const variable_axis& o) const
  {
    if (!axis_base::operator==(o))
        return false;
    return std::equal(x_.get(), x_.get() + bins() + 1, o.x_.get());
  }

private:
  std::unique_ptr<value_type[]> x_; // smaller size compared to std::vector

  template <class Archive, typename RealType1>
  friend void serialize(Archive&, variable_axis<RealType1>&, unsigned);
};

/** An axis for a contiguous range of integers.
  *
  * There are no underflow/overflow bins for this axis.
  * Binning is a O(1) operation.
  */
class integer_axis: public axis_base {
public:
  using value_type = int;

  /** Construct axis over integer range [min, max].
    *
    * \param min smallest integer of the covered range.
    * \param max largest integer of the covered range.
    */
  integer_axis(value_type min, value_type max,
               const std::string& label = std::string(),
               bool uoflow = true) :
    axis_base(max + 1 - min, label, uoflow),
    min_(min)
  {
    if (min > max)
      throw std::logic_error("min <= max required");
  }

  integer_axis() = default;
  integer_axis(const integer_axis&) = default;
  integer_axis& operator=(const integer_axis&) = default;
  integer_axis(integer_axis&&) = default;
  integer_axis& operator=(integer_axis&&) = default;

  /// Returns the bin index for the passed argument.
  inline int index(value_type x) const
  {
    const int z = x - min_;
    return z >= 0 ? (z > bins() ? bins() : z) : -1;
  }

  /// Returns the integer that is mapped to the bin index.
  value_type operator[](int idx) const { return min_ + idx; }

  bool operator==(const integer_axis& o) const
  {
    return axis_base::operator==(o) && min_ == o.min_;
  }

private:
  value_type min_ = 0;

  template <class Archive>
  friend void serialize(Archive&, integer_axis&, unsigned);
};

/** An axis for enumerated categories.
  *
  * The axis stores the category labels, and expects that they
  * are addressed using an integer from ``0`` to ``n-1``.
  * There are no underflow/overflow bins for this axis.
  * Binning is a O(1) operation.
  */
class category_axis {
public:
  using value_type = const std::string&;

  template <typename Iterator>
  category_axis(Iterator begin, Iterator end) :
    size_(std::distance(begin, end)),
    ptr_(new std::string[size_])
  {
    if (size_ == 0)
      throw std::logic_error("at least one argument required");
    std::copy(begin, end, ptr_.get());
  }

  /** Construct from a list of strings.
    *
    * \param categories sequence of labeled categories.
    */
  explicit
  category_axis(const std::initializer_list<std::string>& categories) :
    category_axis(categories.begin(), categories.end())
  {}

  explicit
  category_axis(const std::vector<std::string>& categories) :
    category_axis(categories.begin(), categories.end())
  {}

  category_axis() = default;

  category_axis(const category_axis& other) :
    category_axis(other.ptr_.get(),
                  other.ptr_.get() + other.size_)
  {}
  category_axis& operator=(const category_axis& other) {
    if (this != &other) {
      size_ = other.size_;
      ptr_.reset(new std::string[other.size_]);
      std::copy(other.ptr_.get(), other.ptr_.get() + other.size_, ptr_.get());
    }
    return *this;
  }

  category_axis(category_axis&& other) :
    size_(other.size_),
    ptr_(std::move(other.ptr_))
  {
    other.size_ = 0;
  }

  category_axis& operator=(category_axis&& other) {
    if (this != &other) {
      size_ = other.size_;
      ptr_ = std::move(other.ptr_);
      other.size_ = 0;
    }
    return *this;
  }

  inline int bins() const { return size_; }
  inline int shape() const { return size_; }
  inline bool uoflow() const { return false; }

  /// Returns the bin index for the passed argument.
  /// Performs a range check.
  inline int index(int x) const
  { if (!(0 <= x && x < size_))
      throw std::out_of_range("category index is out of range");
    return x; }

  /// Returns the category for the bin index.
  value_type operator[](int idx) const
  { return ptr_.get()[idx]; }

  bool operator==(const category_axis& other) const
  {
    return size_ == other.size_ &&
      std::equal(ptr_.get(), ptr_.get() + size_, other.ptr_.get());
  }

private:
  int size_ = 0;
  std::unique_ptr<std::string[]> ptr_;

  template <class Archive>
  friend void serialize(Archive&, category_axis&, unsigned);
};

using default_axes = mpl::vector<
  regular_axis<double>, regular_axis<float>,
  circular_axis<double>, circular_axis<float>,
  variable_axis<double>, variable_axis<float>,
  integer_axis, category_axis
>::type;

}
}

#endif
