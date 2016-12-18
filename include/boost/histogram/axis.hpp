// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_HPP_
#define _BOOST_HISTOGRAM_AXIS_HPP_

#include <boost/histogram/detail/utility.hpp>
#include <boost/variant.hpp>
#include <boost/math/constants/constants.hpp>
#include <type_traits>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>


/** \file boost/histogram/axis
 *  \brief Defines the axis classes.
 *
 */

namespace boost {
namespace histogram {

/// Common base class for most axes
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
  axis_base(axis_base&&) = default;
  axis_base& operator=(const axis_base&) = default;
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

/// Mixin for real-valued axes
template <typename Derived>
class real_axis {
public:
  typedef double value_type;

  double left(int idx) const {
    return static_cast<const Derived&>(*this)[idx];
  }

  double right(int idx) const {
    return static_cast<const Derived&>(*this)[idx + 1];
  }

  double center(int idx) const {
    return 0.5 * (left(idx) + right(idx));
  }
};

/** Axis for binning real-valued data into equidistant bins
  *
  * This is the simplest and common binning strategy.
  * Binning is a O(1) operation.
  */
class regular_axis: public axis_base, public real_axis<regular_axis> {
public:
  /** Constructor
   *
   * \param n number of bins
   * \param min low edge of first bin
   * \param max high edge of last bin
   * \param label description of the axis
   * \param uoflow add underflow and overflow bins to the histogram for this axis or not
   */
  regular_axis(unsigned n, double min, double max,
               const std::string& label = std::string(),
               bool uoflow = true) :
    axis_base(n, label, uoflow),
    min_(min),
    delta_((max - min) / n)
  {
    if (min >= max)
      throw std::logic_error("min < max required");
  }

  regular_axis() : min_(0), delta_(0) {}
  regular_axis(const regular_axis&) = default;
  regular_axis(regular_axis&&) = default;
  regular_axis& operator=(const regular_axis&) = default;
  regular_axis& operator=(regular_axis&&) = default;

  /// Returns the bin index for the passed argument (optimized code).
  inline int index(double x) const
  {
    const double z = (x - min_) / delta_;
    return z >= 0.0 ? (z > bins() ? bins() : static_cast<int>(z)) : -1;
  }

  double operator[](int idx) const
  {
    if (idx < 0)
        return -std::numeric_limits<double>::infinity();
    if (idx > bins())
        return std::numeric_limits<double>::infinity();
    const double z = double(idx) / bins();
    return (1.0 - z) * min_ + z * (min_ + delta_ * bins());
  }

  bool operator==(const regular_axis& o) const
  {
    return axis_base::operator==(o) &&
           min_ == o.min_ &&
           delta_ == o.delta_;
  }

private:
  double min_, delta_;

  template <class Archive>
  friend void serialize(Archive&, regular_axis&, unsigned);
};

/** Axis for real-valued angles
  *
  * There are no overflow/underflow bins for this axis,
  * since the axis is circular and wraps around after :math:`2 \pi`.
  * Binning is a O(1) operation.
  */
class polar_axis: public axis_base, public real_axis<polar_axis> {
public:
  /** Constructor
   * \param n  number of bins
   * \param start  starting phase of the angle
   * \param label  description of the axis
	 */
  explicit
  polar_axis(unsigned n, double start = 0.0,
             const std::string& label = std::string()) :
    axis_base(n, label, false),
    start_(start)
  {}

  polar_axis() : start_(0) {}
  polar_axis(const polar_axis&) = default;
  polar_axis(polar_axis&&) = default;
  polar_axis& operator=(const polar_axis&) = default;
  polar_axis& operator=(polar_axis&&) = default;

  ///Returns the bin index for the passed argument.
  inline int index(double x) const {
    using namespace boost::math::double_constants;
    const double z = (x - start_) / two_pi;
    const int i = static_cast<int>(std::floor(z * bins())) % bins();
    return i + (i < 0) * bins();
  }

  double operator[](int idx) const
  {
    using namespace boost::math::double_constants;
    const double z = double(idx) / bins();
    return z * two_pi + start_;
  }

  bool operator==(const polar_axis& o) const
  { return axis_base::operator==(o) && start_ == o.start_; }

private:
  double start_;

  template <class Archive>
  friend void serialize(Archive&, polar_axis&, unsigned);
};

/** An axis for real-valued data and bins of varying width.
  *
  * Binning is a O(log(N)) operation. If speed matters and the problem domain
  * allows it, prefer a regular_axis.
  */
class variable_axis : public axis_base, public real_axis<variable_axis> {
public:
	/** Constructor
	 *
	 * \param x sequence of bin edges
	 * \param label description of the axis
	 * \param uoflow add under-/overflow bins for this axis or not
	 */
  explicit
  variable_axis(const std::initializer_list<double>& x,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(x.size() - 1, label, uoflow),
      x_(new double[x.size()])
  {
      std::copy(x.begin(), x.end(), x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  variable_axis(const std::vector<double>& x,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(x.size() - 1, label, uoflow),
      x_(new double[x.size()])
  {
      std::copy(x.begin(), x.end(), x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  template <typename Iterator>
  variable_axis(Iterator begin, Iterator end,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(std::distance(begin, end) - 1, label, uoflow),
      x_(new double[std::distance(begin, end)])
  {
      std::copy(begin, end, x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  variable_axis() = default;
  variable_axis(const variable_axis& o) :
    axis_base(o),
    x_(new double[bins() + 1])
  {
    std::copy(o.x_.get(), o.x_.get() + bins() + 1, x_.get());
  }
  variable_axis(variable_axis&&) = default;
  variable_axis& operator=(const variable_axis& o)
  {
    if (this != &o) {
        axis_base::operator=(o);
        x_.reset(new double[bins() + 1]);
        std::copy(o.x_.get(), o.x_.get() + bins() + 1, x_.get());
    }
    return *this;
  }
  variable_axis& operator=(variable_axis&&) = default;

  inline int index(double x) const {
    return std::upper_bound(x_.get(), x_.get() + bins() + 1, x)
           - x_.get() - 1;
  }

  double operator[](int idx) const
  {
    if (idx < 0)
        return -std::numeric_limits<double>::infinity();
    if (idx > bins())
        return std::numeric_limits<double>::infinity();
    return x_[idx];
  }

  bool operator==(const variable_axis& o) const
  {
    if (!axis_base::operator==(o))
        return false;
    for (unsigned i = 0, n = bins() + 1; i < n; ++i)
        if (x_[i] != o.x_[i])
            return false;
    return true;
  }

private:
  std::unique_ptr<double[]> x_; // smaller size compared to std::vector

  template <class Archive>
  friend void serialize(Archive&, variable_axis&, unsigned);
};

/** An axis for a contiguous range of integers. There are no underflow/overflow
 *  bins for this axis. Binning is a O(1) operation.
 */
class integer_axis: public axis_base {
public:
  typedef int value_type;

  /**Construct axis over consecutive sequence of integers
   *
   * \param min smallest integer of the covered range
   * \param max largest integer of the covered range
   */
  integer_axis(int min, int max,
               const std::string& label = std::string(),
               bool uoflow = true) :
    axis_base(max + 1 - min, label, uoflow),
    min_(min)
  {
    if (min >= max)
      throw std::logic_error("min < max required");
  }

  integer_axis() : min_(0) {}
  integer_axis(const integer_axis&) = default;
  integer_axis(integer_axis&&) = default;
  integer_axis& operator=(const integer_axis&) = default;
  integer_axis& operator=(integer_axis&&) = default;

  ///Returns the bin index for the passed argument.
  inline int index(double x) const
  {
    const double z = x - min_;
    return z >= 0.0 ? (z > bins() ? bins() : static_cast<int>(z)) : -1;
  }
  ///Returns the integer that is mapped to the bin index.
  int operator[](int idx) const { return min_ + idx; }

  bool operator==(const integer_axis& o) const
  {
    return axis_base::operator==(o) && min_ == o.min_;
  }

private:
  int min_;

  template <class Archive>
  friend void serialize(Archive&, integer_axis&, unsigned);
};

///An axis for enumerated categories. The axis stores the category labels, and expects that they are addressed using an integer from ``0`` to ``n-1``. There are no underflow/overflow bins for this axis.  Binning is a O(1) operation.
class category_axis {
public:
  typedef std::string value_type;

  /**Construct from initializer list of strings
   *
   * \param categories ordered sequence of categories that this axis discriminates
   */
  explicit
  category_axis(const std::initializer_list<std::string>& categories) :
    categories_(categories)
  {}

  template <typename Iterator>
  category_axis(Iterator cat_begin, Iterator cat_end) :
    categories_(std::distance(cat_begin, cat_end))
  {
    std::copy(cat_begin, cat_end, categories_.begin());
  }

  category_axis(const std::vector<std::string>& cat) :
    categories_(cat)
  {}

  category_axis(std::vector<std::string>&& cat) :
    categories_(std::move(cat))
  {}

  category_axis() {}
  category_axis(const category_axis&) = default;
  category_axis(category_axis&&) = default;
  category_axis& operator=(const category_axis&) = default;
  category_axis& operator=(category_axis&&) = default;

  inline int bins() const { return categories_.size(); }
  inline int shape() const { return categories_.size(); }
  inline bool uoflow() const { return false; }

  ///Returns the bin index for the passed argument.
  inline int index(double x) const
  { if (!(0.0 <= x && x < bins()))
      throw std::out_of_range("category index is out of range");
    return std::rint(x); }
  ///Returns the category for the bin index.
  const std::string& operator[](int idx) const
  { return categories_[idx]; }

  bool operator==(const category_axis& o) const
  { return categories_ == o.categories_; }

private:
  std::vector<std::string> categories_;

  template <class Archive>
  friend void serialize(Archive&, category_axis&, unsigned);
};

using axis_t = variant<
                 regular_axis, // most common type
                 polar_axis,
                 variable_axis,
                 category_axis,
                 integer_axis
               >;

// axis_t is automatically output-streamable if all its bounded types are

}
}

#endif
