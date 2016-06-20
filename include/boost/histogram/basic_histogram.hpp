// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_BASE_HPP_
#define _BOOST_HISTOGRAM_BASE_HPP_

#include <boost/histogram/axis.hpp>
#include <boost/histogram/visitors.hpp>
#include <boost/cstdint.hpp>
#include <boost/preprocessor.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/not.hpp>
#include <boost/move/move.hpp>

#include <bitset>

#if !defined(BOOST_HISTOGRAM_AXIS_LIMIT)
#define BOOST_HISTOGRAM_AXIS_LIMIT 16
#endif

namespace boost {
namespace histogram {

/// This class holds collection of axis instances and computes the internal index
class basic_histogram {
  BOOST_COPYABLE_AND_MOVABLE(basic_histogram)
public:
  typedef container::static_vector<axis_type, BOOST_HISTOGRAM_AXIS_LIMIT> axes_type;
  typedef uintptr_t size_type;

  ~basic_histogram() {}

  // copy semantics (implementation needs to be here, workaround for gcc-bug)
  basic_histogram(const basic_histogram& o) :
    axes_(o.axes_) {}

  basic_histogram& operator=(BOOST_COPY_ASSIGN_REF(basic_histogram) o)
  { if (this != &o) axes_ = o.axes_; return *this; }

  // move semantics (implementation needs to be here, workaround for gcc-bug)
  basic_histogram(BOOST_RV_REF(basic_histogram) o) :
    axes_()
  { std::swap(axes_, o.axes_); }

  basic_histogram& operator=(BOOST_RV_REF(basic_histogram) o)
  { if (this != &o) std::swap(axes_, o.axes_); return *this; }

  ///Returns the number of dimensions of the histogram, how many axis it has
  unsigned dim() const { return axes_.size(); }
  ///Returns the number of bins for axis \a i.
  int bins(unsigned i) const { return apply_visitor(visitor::bins(), axes_[i]); }
  /**Returns the actual number of fields used by the axis. If the axis has no
   * underflow and overflow bins, this is equal to \a bins. Otherwise, the number is larger by 2.
   */
  unsigned shape(unsigned i) const { return apply_visitor(visitor::shape(), axes_[i]); }

  template <typename T>
  typename enable_if<is_same<T, axis_type>, T&>::type
  axis(unsigned i) { return axes_[i]; }

  template <typename T>
  typename enable_if<mpl::not_<is_same<T, axis_type> >, T&>::type
  axis(unsigned i) { return boost::get<T&>(axes_[i]); }

  template <typename T>
  typename enable_if<is_same<T, axis_type>, const T&>::type
  axis(unsigned i) const { return axes_[i]; }

  template <typename T>
  typename enable_if<mpl::not_<is_same<T, axis_type> >, const T&>::type
  axis(unsigned i) const { return boost::get<const T&>(axes_[i]); }

#if defined(BOOST_HISTOGRAM_DOXYGEN)
  /** Returns the axis object at index \a i, casted to type \a T.
   *  A runtime exception is thrown if the type cast is invalid.
   */
  template <typename T> T& axis(unsigned i);
  /** The ``const``-version of the previous member function. */
  template <typename T> const T& axis(unsigned i) const
#endif

protected:
  basic_histogram() {}
  explicit basic_histogram(const axes_type& axes);

#define BOOST_HISTOGRAM_BASE_APPEND(z, n, unused) axes_.push_back(a ## n);
#define BOOST_HISTOGRAM_BASE_CTOR(z, n, unused)                       \
  basic_histogram( BOOST_PP_ENUM_PARAMS_Z(z, n, const axis_type& a) ) \
  {                                                                   \
    axes_.reserve(n);                                                 \
    BOOST_PP_REPEAT(n, BOOST_HISTOGRAM_BASE_APPEND, unused)           \
  }

// generates constructors taking 1 to AXIS_LIMIT arguments
BOOST_PP_REPEAT_FROM_TO(1, BOOST_HISTOGRAM_AXIS_LIMIT, BOOST_HISTOGRAM_BASE_CTOR, nil)

  bool operator==(const basic_histogram&) const;

  template <typename Iterator>
  inline
  size_type pos(const Iterator& begin, const Iterator & end) const {
    detail::linearize lin(true);
    int i = axes_.size();
    Iterator itr = end;
    while (i-- && (itr-- != begin)) {
      lin.x = *itr;
      apply_visitor(lin, axes_[i]);
    }
    return lin.k;
  }

  template<typename Iterator>
  inline
  size_type linearize(const Iterator &begin, const Iterator & end) const {
    detail::linearize lin(false);
    int i = axes_.size();
    Iterator itr = end;
    while (i-- && (itr-- != begin)) {
      lin.j = *itr;
      apply_visitor(lin, axes_[i]);
    }
    return lin.k;
  }

  // compute the number of fields needed for storage
  size_type field_count() const;

private:
  axes_type axes_;

  template <class Archive>
  friend void serialize(Archive& ar, basic_histogram & h, unsigned version);
};

}
}

#endif
