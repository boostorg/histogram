// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_HPP_
#define _BOOST_HISTOGRAM_AXIS_HPP_

#include <boost/math/constants/constants.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/utility/string_ref.hpp>
#include <type_traits>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace boost {
namespace histogram {

namespace detail {

  template <typename Value>
  struct bin
  {
    int idx;
    Value value;
  };

  template <>
  struct bin<const std::string&>
  {
    int idx;
    boost::string_ref value;
  };

  template <typename Value>
  struct real_bin
  {
    int idx;
    Value left, right;
  };

  template <typename Value>
  using axis_bin = typename std::conditional<
    std::is_floating_point<Value>::value,
    real_bin<Value>,
    bin<Value>
  >::type;

  template <typename Axis>
  class axis_iterator : public iterator_facade<
      axis_iterator<Axis>,
      const axis_bin<typename Axis::value_type>,
      random_access_traversal_tag
    >
  {
    using bin_type = axis_bin<typename Axis::value_type>;

  public:
    explicit axis_iterator(const Axis& axis, int idx) :
      axis_(axis), value_()
    { value_.idx = idx; set_impl(value_); }

  private:
    void increment() { ++value_.idx; set_impl(value_); }
    void decrement() { --value_.idx; set_impl(value_); }
    void advance(int n) { value_.idx += n; set_impl(value_); }
    int distance_to(const axis_iterator& other) const
    { return other.value_.idx - value_.idx; }
    bool equal(const axis_iterator& other) const
    { return value_.idx == other.value_.idx; }
    const bin_type& dereference() const { return value_; }
    template <typename Value>
    void set_impl(bin<Value>& v)
    { v.value = axis_[v.idx]; }
    template <typename Value>
    void set_impl(real_bin<Value>& v)
    { v.left = axis_[v.idx]; v.right = axis_[v.idx + 1]; }
    const Axis& axis_;
    bin_type value_;
    friend class boost::iterator_core_access;
  };
} // NS detail

/// Common base class for axes.
template <bool UOFlow>
class axis_base;

template <>
class axis_base<true>
{
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
  axis_base(axis_base&& other) :
    size_(other.size_),
    shape_(other.shape_),
    label_(std::move(other.label_))
  { other.size_ = 0; other.shape_ = 0; }
  axis_base& operator=(axis_base&& other) {
    if (this != &other) {
      size_ = other.size_;
      shape_ = other.shape_;
      label_ = std::move(other.label_);
      other.size_ = 0;
      other.shape_ = 0;
    }
    return *this;
  }

  bool operator==(const axis_base& o) const
  { return size_ == o.size_ && shape_ == o.shape_ && label_ == o.label_; }

private:
  int size_ = 0;
  int shape_ = 0;
  std::string label_;

  template <class Archive>
  friend void serialize(Archive&, axis_base<true>&, unsigned);
};

template <>
class axis_base<false>
{
public:
  /// Returns the number of bins, excluding overflow/underflow.
  inline int bins() const { return size_ ; }
  /// Returns the number of bins, including overflow/underflow.
  inline int shape() const { return size_; }
  /// Returns whether axis has extra overflow and underflow bins.
  inline bool uoflow() const { return false; }
  /// Returns the axis label, which is a name or description (not implemented for category_axis).
  const std::string& label() const { return label_; }
  /// Change the label of an axis (not implemented for category_axis).
  void label(const std::string& label) { label_ = label; }

protected:
  axis_base(unsigned n, const std::string& label) :
    size_(n), label_(label)
  {
    if (n == 0)
      throw std::logic_error("bins > 0 required");
  }

  axis_base() = default;
  axis_base(const axis_base&) = default;
  axis_base& operator=(const axis_base&) = default;
  axis_base(axis_base&& other) :
    size_(other.size_),
    label_(std::move(other.label_))
  { other.size_ = 0; }
  axis_base& operator=(axis_base&& other) {
    if (this != &other) {
      size_ = other.size_;
      label_ = std::move(other.label_);
      other.size_ = 0;
    }
    return *this;
  }

  bool operator==(const axis_base& other) const
  { return size_ == other.size_ && label_ == other.label_; }

private:
  int size_ = 0;
  std::string label_;

  template <class Archive>
  friend void serialize(Archive&, axis_base<false>&, unsigned);
};

/** Axis for binning real-valued data into equidistant bins.
  *
  * The simplest and common binning strategy.
  * Very fast. Binning is a O(1) operation.
  */
template <typename RealType=double>
class regular_axis: public axis_base<true> {
public:
  using value_type = RealType;
  using const_iterator = detail::axis_iterator<regular_axis>;

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
    axis_base<true>(n, label, uoflow),
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
    return axis_base<true>::operator==(o) &&
           min_ == o.min_ &&
           delta_ == o.delta_;
  }

  const_iterator begin() const
  { return const_iterator(*this, uoflow() ? -1 : 0); }

  const_iterator end() const
  { return const_iterator(*this, uoflow() ? bins() + 1 : bins()); }

private:
  value_type min_ = 0.0, delta_ = 1.0;

  template <class Archive, typename RealType1>
  friend void serialize(Archive&, regular_axis<RealType1>&, unsigned);
};

/** Axis for real-valued angles.
  *
  * The axis is circular and wraps around reaching the
  * perimeter value. Therefore, there are no overflow/underflow
  * bins for this axis. Binning is a O(1) operation.
  */
template <typename RealType=double>
class circular_axis: public axis_base<false> {
public:
  using value_type = RealType;
  using const_iterator = detail::axis_iterator<circular_axis>;

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
    axis_base<false>(n, label),
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
    return axis_base<false>::operator==(o) &&
           phase_ == o.phase_ &&
           perimeter_ == o.perimeter_;
  }

  value_type perimeter() const { return perimeter_; }
  value_type phase() const { return phase_; }

  const_iterator begin() const
  { return const_iterator(*this, 0); }

  const_iterator end() const
  { return const_iterator(*this, bins()); }

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
class variable_axis : public axis_base<true> {
public:
  using value_type = RealType;
  using const_iterator = detail::axis_iterator<variable_axis>;

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
      axis_base<true>(x.size() - 1, label, uoflow),
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
      axis_base<true>(x.size() - 1, label, uoflow),
      x_(new value_type[x.size()])
  {
      std::copy(x.begin(), x.end(), x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  template <typename Iterator>
  variable_axis(Iterator begin, Iterator end,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base<true>(std::distance(begin, end) - 1, label, uoflow),
      x_(new value_type[std::distance(begin, end)])
  {
      std::copy(begin, end, x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  variable_axis() = default;
  variable_axis(const variable_axis& o) :
    axis_base<true>(o),
    x_(new value_type[bins() + 1])
  {
    std::copy(o.x_.get(), o.x_.get() + bins() + 1, x_.get());
  }
  variable_axis& operator=(const variable_axis& o)
  {
    if (this != &o) {
        axis_base<true>::operator=(o);
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
    if (!axis_base<true>::operator==(o))
        return false;
    return std::equal(x_.get(), x_.get() + bins() + 1, o.x_.get());
  }

  const_iterator begin() const
  { return const_iterator(*this, uoflow() ? -1 : 0); }

  const_iterator end() const
  { return const_iterator(*this, uoflow() ? bins() + 1 : bins()); }

private:
  std::unique_ptr<value_type[]> x_; // smaller size compared to std::vector

  template <class Archive, typename RealType1>
  friend void serialize(Archive&, variable_axis<RealType1>&, unsigned);
};

/** An axis for a contiguous range of integers.
  *
  * Binning is a O(1) operation. This axis operates
  * faster than a regular_axis.
  */
class integer_axis: public axis_base<true> {
public:
  using value_type = int;
  using const_iterator = detail::axis_iterator<integer_axis>;

  /** Construct axis over integer range [min, max].
    *
    * \param min smallest integer of the covered range.
    * \param max largest integer of the covered range.
    */
  integer_axis(value_type min, value_type max,
               const std::string& label = std::string(),
               bool uoflow = true) :
    axis_base<true>(max + 1 - min, label, uoflow),
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
    return axis_base<true>::operator==(o) && min_ == o.min_;
  }

  const_iterator begin() const
  { return const_iterator(*this, uoflow() ? -1 : 0); }

  const_iterator end() const
  { return const_iterator(*this, uoflow() ? bins() + 1 : bins()); }

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
class category_axis : public axis_base<false> {
public:
  using value_type = const std::string&;
  using const_iterator = detail::axis_iterator<category_axis>;

  template <typename Iterator>
  category_axis(Iterator begin, Iterator end,
                const std::string& label = std::string()) :
    axis_base<false>(std::distance(begin, end), label),
    ptr_(new std::string[bins()])
  {
    std::copy(begin, end, ptr_.get());
  }

  /** Construct from a list of strings.
    *
    * \param categories sequence of labeled categories.
    */
  explicit
  category_axis(const std::initializer_list<std::string>& categories,
                const std::string& label = std::string()) :
    category_axis(categories.begin(), categories.end(), label)
  {}

  explicit
  category_axis(const std::vector<std::string>& categories,
                const std::string& label = std::string()) :
    category_axis(categories.begin(), categories.end(), label)
  {}

  category_axis() = default;

  category_axis(const category_axis& other) :
    category_axis(other.ptr_.get(),
                  other.ptr_.get() + other.bins(),
                  other.label())
  {}
  category_axis& operator=(const category_axis& other) {
    if (this != &other) {
      axis_base<false>::operator=(other);
      ptr_.reset(new std::string[other.bins()]);
      std::copy(other.ptr_.get(), other.ptr_.get() + other.bins(), ptr_.get());
    }
    return *this;
  }

  category_axis(category_axis&& other) :
    axis_base<false>(std::move(other)),
    ptr_(std::move(other.ptr_))
  {}

  category_axis& operator=(category_axis&& other) {
    if (this != &other) {
      axis_base<false>::operator=(std::move(other));
      ptr_ = std::move(other.ptr_);
    }
    return *this;
  }

  /// Returns the bin index for the passed argument.
  /// Performs a range check.
  inline int index(int x) const
  { if (!(0 <= x && x < bins()))
      throw std::out_of_range("category index is out of range");
    return x; }

  /// Returns the category for the bin index.
  value_type operator[](int idx) const
  { return ptr_.get()[idx]; }

  bool operator==(const category_axis& other) const
  {
    return axis_base<false>::operator==(other) &&
      std::equal(ptr_.get(), ptr_.get() + bins(), other.ptr_.get());
  }

  const_iterator begin() const
  { return const_iterator(*this, 0); }

  const_iterator end() const
  { return const_iterator(*this, bins()); }

private:
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
