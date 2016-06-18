// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/basic_histogram.hpp>
#include <boost/histogram/detail/nstore.hpp>
#include <boost/preprocessor.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/assert.hpp>
#include <boost/move/move.hpp>
#include <boost/range.hpp>
#include <boost/config.hpp>
#include <stdexcept>
#include <iterator>

/** \file boost/histogram/histogram.hpp
 *  \brief Defines the core class of this library, \ref boost::histogram::histogram .
 *
 */

namespace boost {
namespace histogram {

/** The class implements an n-dimensional histogram, managing counts in bins.

    It inherits from \ref boost::histogram::basic_histogram "basic_histogram",
    which manages the stored axis
    instances and the conversion of an n-dimensional tuple or index into an
    internal linear offset that is used to address the bin count. How the bin
    count is stored is an encapsulated implementation detail.
 */
class histogram : public basic_histogram {
  BOOST_COPYABLE_AND_MOVABLE(histogram)
public:
  histogram() {}

  explicit histogram(const axes_type& axes);

#define BOOST_HISTOGRAM_CTOR(z, n, unused)                                 \
  explicit histogram( BOOST_PP_ENUM_PARAMS_Z(z, n, const axis_type& a) ) : \
    basic_histogram( BOOST_PP_ENUM_PARAMS_Z(z, n, a) ),                    \
    data_(field_count())                                                   \
  {}

// generates constructors taking 1 to AXIS_LIMIT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_HISTOGRAM_CTOR, nil)

#if defined(BOOST_HISTOGRAM_DOXYGEN)
  /**Constructors for a variable number of axis types, each defining the binning
     scheme for its dimension. Up to \ref BOOST_HISTOGRAM_AXIS_LIMIT axis types
     can be passed to the constructor, yielding the same number of dimensions.

  */
  explicit histogram(const axis_type& a0, ...);
#endif
  // copy semantics (implementation needs to be here, workaround for gcc-bug)
  histogram(const histogram& o) :
    basic_histogram(o),
    data_(o.data_)
  {}

  histogram& operator=(BOOST_COPY_ASSIGN_REF(histogram) o)
  {
    if (this != &o) {
      basic_histogram::operator=(static_cast<const basic_histogram&>(o));
      data_ = o.data_;
    }
    return *this;
  }

  // move semantics (implementation needs to be here, workaround for gcc-bug)
  histogram(BOOST_RV_REF(histogram) o) :
    basic_histogram(::boost::move(static_cast<basic_histogram&>(o))),
    data_(::boost::move(o.data_))
  {}

  histogram& operator=(BOOST_RV_REF(histogram) o)
  {
    if (this != &o) {
      basic_histogram::operator=(::boost::move(static_cast<basic_histogram&>(o)));
      data_ = ::boost::move(o.data_);
    }
    return *this;
  }

  /** Fills the histogram with a range. It checks at run-time
   *  that the size agrees with the dimensions of the histogram.

      Allocation of internal memory is delayed until the first call to this function.
   *
   */
  template<typename Iterator>
  inline void fill(boost::iterator_range<Iterator> range)
  {
	BOOST_ASSERT_MSG(range.size() == dim(), "wrong number of arguments");
    const size_type k = pos(range);
    if (k != uintmax_t(-1))
      data_.increase(k);
  }

#define BOOST_HISTOGRAM_FILL(z, n, unused)                   \
  inline                                                     \
  void fill( BOOST_PP_ENUM_PARAMS_Z(z, n, double x) )        \
  {                                                          \
    const double buffer[n] = { BOOST_PP_ENUM_PARAMS(n, x) }; \
    fill(boost::make_iterator_range(boost::begin(buffer), boost::end(buffer)));\
  }

// generates fill functions taking 1 to AXIS_LIMT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_HISTOGRAM_FILL, nil)  

#if defined(BOOST_HISTOGRAM_DOXYGEN)
/**\overload void fill(boost::iterator_range<Iterator> range)
	Overload taking a variadic sequence.
*/
   void fill(double x0, ...);
#endif


  /** Fills the histogram from a iterator range,
   * using a weight. It checks at run-time that the length agrees with the
   * dimensions of the histogram.

     Allocation of internal memory is delayed until the first call to this function.
     If the histogram was filled with :cpp:func:`fill_c` before, the internal
     memory is converted to the wide format used for storing weighted counts.

     If the data is not weighted (all weights are 1.0), using \ref fill is much
     more space-efficient. In the most extreme case, storing of weighted counts
     consumes 16x more memory.
   *
   *
   */
  template<typename Iterator>
  inline void wfill(boost::iterator_range<Iterator> range, double w)
  {
    BOOST_ASSERT_MSG(range.size() == dim(), "wrong number of arguments");
    const size_type k = pos(range);
    if (k != uintmax_t(-1))
      data_.increase(k, w);
  }

#define BOOST_HISTOGRAM_WFILL(z, n, unused)                      \
  inline                                                         \
  void wfill( BOOST_PP_ENUM_PARAMS_Z(z, n, double x), double w ) \
  {                                                              \
    const double buffer[n] = { BOOST_PP_ENUM_PARAMS(n, x) };     \
    wfill(boost::make_iterator_range(boost::begin(buffer), boost::end(buffer)), w); \
  }

// generates wfill functions taking 1 to AXIS_LIMT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_HISTOGRAM_WFILL, nil)  


#if defined(BOOST_HISTOGRAM_DOXYGEN)
/**\overload void wfill(boost::iterator_range<Iterator> range, double weight)
	Overload taking a variadic sequence.
*/
   void fill(double x0, ...);
#endif
  // C-style call
  inline
  double value_c(unsigned n, const int* idx)
    const
  {
    BOOST_ASSERT_MSG(n == dim(), "wrong number of arguments");
    return data_.value(linearize(idx));
  }

#define BOOST_HISTOGRAM_VALUE(z, n, unused)                 \
  inline                                                    \
  double value( BOOST_PP_ENUM_PARAMS_Z(z, n, int i) )       \
    const                                                   \
  {                                                         \
    const int idx[n] = { BOOST_PP_ENUM_PARAMS_Z(z, n, i) }; \
    return value_c(n, idx); /* size is checked here */        \
  }

// generates value functions taking 1 to AXIS_LIMT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_HISTOGRAM_VALUE, nil)  

  // C-style call
  inline
  double variance_c(unsigned n, const int* idx)
    const
  {
    BOOST_ASSERT_MSG(n == dim(), "wrong number of arguments");
    return data_.variance(linearize(idx));
  }

#define BOOST_HISTOGRAM_VARIANCE(z, n, unused)              \
  inline                                                    \
  double variance( BOOST_PP_ENUM_PARAMS_Z(z, n, int i) )    \
    const                                                   \
  {                                                         \
    const int idx[n] = { BOOST_PP_ENUM_PARAMS_Z(z, n, i) }; \
    return variance_c(n, idx); /* size is checked here */     \
  }

// generates variance functions taking 1 to AXIS_LIMT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_HISTOGRAM_VARIANCE, nil)  

  unsigned depth() const { return data_.depth(); }

  double sum() const;

  bool operator==(const histogram& o) const 
  { return basic_histogram::operator==(o) &&
           data_ == o.data_; }

  histogram& operator+=(const histogram& o)
  {
    if (!basic_histogram::operator==(o))
      throw std::logic_error("histograms have different axes");
    data_ += o.data_;
    return *this;
  }

private:
  detail::nstore data_;

  template <class Archive>
  friend void serialize(Archive& ar, histogram & h, unsigned version);
};

inline histogram operator+(const histogram& a, const histogram& b) {
  histogram result(a);
  return result += b;
}

}
}

#endif
