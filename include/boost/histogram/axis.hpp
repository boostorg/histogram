// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_AXIS_HPP_
#define _BOOST_HISTOGRAM_AXIS_HPP_

#include <boost/algorithm/clamp.hpp>
#include <boost/variant.hpp>
#include <boost/scoped_array.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/utility/enable_if.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <ostream>

/** \file boost/histogram/axis
 *  \brief Defines the axis classes.
 *
 */

namespace boost {
namespace histogram {

/// common base class for most axes
class axis_base {
public:
  ///Returns the number of bins.
  inline int bins() const { return size_ > 0 ? size_ : -size_; }
  ///Returns whether overflow and underflow bins will be added in the histogram.
  inline bool uoflow() const { return size_ > 0; }
  ///Change the label of an axis (not implemented for category_axis).
  const std::string& label() const { return label_; }
  ///Returns the axis label, which is a name or description (not implemented for category_axis).
  void label(const std::string& label) { label_ = label; }

protected:
  axis_base(int, const std::string&, bool);

  axis_base() : size_(0) {}
  axis_base(const axis_base&);
  axis_base& operator=(const axis_base&);

  bool operator==(const axis_base&) const;

private:
  int size_;
  std::string label_;

  template <class Archive>
  friend void serialize(Archive& ar, axis_base & base, unsigned version);
};

// mixin for real-valued axes
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

///An axis for real-valued data and bins of equal width. Binning is a O(1) operation.
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
  regular_axis(int n, double min, double max,
               const std::string& label = std::string(),
               bool uoflow = true);

  regular_axis() : min_(0), range_(0) {}
  regular_axis(const regular_axis&);
  regular_axis& operator=(const regular_axis&);

  ///Returns the bin index for the passed argument.
  inline int index(double x) const {
    const double z = (x - min_) / range_;
    return algorithm::clamp(static_cast<int>(floor(z * bins())), -1, bins());
  }

  double operator[](int idx) const;
  bool operator==(const regular_axis&) const;
private:
  double min_, range_;

  template <class Archive>
  friend void serialize(Archive& ar, regular_axis & axis ,unsigned version);
};

///An axis for real-valued angles. There are no overflow/underflow bins for this axis, since the axis is circular and wraps around after :math:`2 \pi`. Binning is a O(1) operation.
class polar_axis: public axis_base, public real_axis<polar_axis> {
public:
  /** Constructor
    \param n  number of bins
	\param start  starting phase of the angle
	\param label  description of the axis
	*/
  explicit 
  polar_axis(int n, double start = 0.0,
             const std::string& label = std::string());

  polar_axis() : start_(0) {}
  polar_axis(const polar_axis&);
  polar_axis& operator=(const polar_axis&);

  ///Returns the bin index for the passed argument.
  inline int index(double x) const { 
    using namespace boost::math::double_constants;
    const double z = (x - start_) / two_pi;
    const int i = static_cast<int>(floor(z * bins())) % bins();
    return i + (i < 0) * bins();
  }

  double operator[](int idx) const;
  bool operator==(const polar_axis&) const;
private:
  double start_;

  template <class Archive>
  friend void serialize(Archive& ar, polar_axis & axis, unsigned version);
};

///An axis for real-valued data and bins of varying width. Binning is a O(log(N)) operation. If speed matters and the problem domain allows it, prefer a regular_axis.
class variable_axis : public axis_base, public real_axis<variable_axis> {
public:
	/** Constructor
	 *
	 * \param x     An axis for real-valued data and bins of varying width. Binning is a O(log(N)) operation. If speed matters and the problem domain allows it, prefer a regular_axis.
	 * \param label description of the axis
	 * \param uoflow add underflow and overflow bins to the histogram for this axis or not
	 */
  template <typename Container>
  explicit
  variable_axis(const Container& x,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(x.size() - 1, label, uoflow),
      x_(new double[x.size()])
  {
      std::copy(x.begin(), x.end(), x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  template <typename Iterator>
  variable_axis(const Iterator& begin, const Iterator& end,
                const std::string& label = std::string(),
                bool uoflow = true) :
      axis_base(end - begin - 1, label, uoflow),
      x_(new double[end - begin])
  {
      std::copy(begin, end, x_.get());
      std::sort(x_.get(), x_.get() + bins() + 1);
  }

  variable_axis() {}
  variable_axis(const variable_axis&);
  variable_axis& operator=(const variable_axis&);

  inline int index(double x) const { 
    return std::upper_bound(x_.get(), x_.get() + bins() + 1, x)
           - x_.get() - 1;
  }

  double operator[](int idx) const;
  bool operator==(const variable_axis&) const;
private:
  boost::scoped_array<double> x_;

  template <class Archive>
  friend void serialize(Archive& ar, variable_axis & axis, unsigned version);
};


///An axis for enumerated categories. The axis stores the category labels, and expects that they are addressed using an integer from ``0`` to ``n-1``. There are no underflow/overflow bins for this axis.  Binning is a O(1) operation.
class category_axis {
public:
  typedef std::string value_type;
  /**Construct from a single string
   *
   * @param categories a string of categories separated by the character \a ;
   */
  explicit
  category_axis(const std::string& categories);
  /**Construct from a single string
   *
   * @param categories an ordered sequence of categories that this axis discriminates
   */
  explicit
  category_axis(const std::vector<std::string>& categories);

  category_axis() {}
  category_axis(const category_axis&);
  category_axis& operator=(const category_axis&);

  inline int bins() const { return categories_.size(); }

  ///Returns the bin index for the passed argument.
  inline int index(double x) const { return static_cast<int>(x + 0.5); }
  ///Returns the category for the bin index.
  const std::string& operator[](int idx) const
  { return categories_[idx]; }

  inline bool uoflow() const { return false; }

  bool operator==(const category_axis&) const;
private:
  std::vector<std::string> categories_;

  template <class Archive>
  friend void serialize(Archive& ar, category_axis & axis, unsigned version);
};

/** An axis for a contiguous range of integers. There are no underflow/overflow
 *  bins for this axis. Binning is a O(1) operation.
 */
class integer_axis: public axis_base {
public:
  typedef int value_type;
  ///Constructor
  integer_axis(int min, int max,
               const std::string& label = std::string(),
               bool uoflow = true);

  integer_axis() : min_(0) {}
  integer_axis(const integer_axis&);
  integer_axis& operator=(const integer_axis&);

  ///Returns the bin index for the passed argument.
  inline int index(double x) const
  { return algorithm::clamp(rint(x) - min_, -1, bins()); }
  ///Returns the integer that is mapped to the bin index.
  int operator[](int idx) const { return min_ + idx; }

  bool operator==(const integer_axis&) const;
private:
  int min_;

  template <class Archive>
  friend void serialize(Archive& ar, integer_axis & axis, unsigned version);
};

namespace detail {
  struct linearize : public static_visitor<void>
  {
    typedef uintptr_t size_type;
    double x;
    int j;
    size_type k, stride;

    linearize() :
      x(std::numeric_limits<double>::quiet_NaN()),
      j(0),
      k(0),
      stride(1)
    {}

    template <typename A>
    void operator()(const A& a) {
      if (k < size_type(-1)) {
        if (!std::isnan(x))
          j = a.index(x);
        const int bins = a.bins();
        const int range = bins + 2 * a.uoflow();
        // the following three lines work for any overflow setting
        j += (j < 0) * (bins + 2); // wrap around if j < 0
        if (j < range) {
          k += j * stride;
          stride *= range;
        }
        else {
          k = size_type(-1); // indicate out of range
        }
      }
    }
  };
}

typedef variant<
  regular_axis, // most common type
  polar_axis,
  variable_axis,
  category_axis,
  integer_axis
> axis_type;

std::ostream& operator<<(std::ostream&, const regular_axis&);
std::ostream& operator<<(std::ostream&, const polar_axis&);
std::ostream& operator<<(std::ostream&, const variable_axis&);
std::ostream& operator<<(std::ostream&, const category_axis&);
std::ostream& operator<<(std::ostream&, const integer_axis&);
// axis_type is automatically output-streamable if all its bounded types are 

}
}

#endif
