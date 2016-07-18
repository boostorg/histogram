// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_HPP_

#include <boost/config.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/visitors.hpp>
#include <boost/histogram/static_storage.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <cstddef>
#include <type_traits>
#include <array>
#include <vector>
#include <stdexcept>
#include <iterator>
#include <iostream>

#define DEBUG(x) std::cout << #x << " " << x << std::endl
#define TRACE(msg) std::cout << __LINE__ << " " msg << std::endl

namespace boost {
namespace histogram {

///Use dynamic dimension
constexpr unsigned Dynamic = 0;

template <unsigned Dim, typename StoragePolicy = dynamic_storage>
class histogram_t
{
public:
  using value_t = typename StoragePolicy::value_t;
  using variance_t = typename StoragePolicy::variance_t;

  histogram_t() = default;
  histogram_t(const histogram_t& other) = default;
  histogram_t(histogram_t&& other) = default;
  histogram_t& operator=(const histogram_t& other) = default;
  histogram_t& operator=(histogram_t&& other) = default;

  template <typename... Axes>
  histogram_t(axis_t a, Axes... axes)
  {
    assign_axis(a, axes...);
    storage_ = StoragePolicy(field_count());
  }

  constexpr unsigned dim() const { return Dim; }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const { return storage_.size(); }

  /// Number of bins along axis \a i, including underflow/overflow
  std::size_t shape(unsigned i) const
  {
    BOOST_ASSERT(i < Dim);
    return apply_visitor(visitor::shape(), axes_[i]);
  }

  /// Number of bins along axis \a i, excluding underflow/overflow
  int bins(unsigned i) const
  {
    BOOST_ASSERT(i < Dim);
    return apply_visitor(visitor::bins(), axes_[i]);
  }

  template <typename... Args>
  void fill(Args... args)
  {
    static_assert(sizeof...(args) == Dim,
                  "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, args...);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename... Args>
  void wfill(Args... args)
  {
    static_assert(sizeof...(args) == (Dim + 1),
                  "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    const double w = windex_impl(lin, args...);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename... Args>
  value_t value(Args... args)
  {
    static_assert(sizeof...(args) == Dim,
                  "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(lin.out);
  }

  template <typename... Args>
  variance_t variance(Args... args)
  {
    static_assert(sizeof...(args) == Dim,
                  "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, std::forward<Args>(args)...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.variance(lin.out);
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, axis_t>::value, T&>::type
  axis(unsigned i) { return axes_[i]; }

  template <typename T>
  typename std::enable_if<!std::is_same<T, axis_t>::value, T&>::type
  axis(unsigned i) { return boost::get<T&>(axes_[i]); }

  template <typename T>
  typename std::enable_if<std::is_same<T, axis_t>::value, const T&>::type
  axis(unsigned i) const { return axes_[i]; }

  template <typename T>
  typename std::enable_if<!std::is_same<T, axis_t>::value, const T&>::type
  axis(unsigned i) const { return boost::get<const T&>(axes_[i]); }

#if defined(BOOST_HISTOGRAM_DOXYGEN)
  /** Returns the axis object at index \a i, casted to type \a T.
   *  A runtime exception is thrown if the type cast is invalid.
   */
  template <typename T> T& axis(unsigned i);
  /** The ``const``-version of the previous member function. */
  template <typename T> const T& axis(unsigned i) const
#endif

  double sum() const
  {
    double result = 0.0;
    for (std::size_t i = 0, n = size(); i < n; ++i)
      result += storage_.value(i);
    return result;
  }

  template <unsigned OtherDim>
  bool operator==(const histogram_t<OtherDim, StoragePolicy>& other) const
  {
    return dim() == other.dim() && axes_ == other.axes_ &&
           storage_ == other.storage_;
  }

  template <unsigned OtherDim, typename OtherStoragePolicy>
  bool operator==(const histogram_t<OtherDim, OtherStoragePolicy>&) const
  { return false; }

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t& operator+=(const histogram_t<OtherDim, OtherStoragePolicy>& other)
  {
    static_assert(std::is_convertible<OtherStoragePolicy, StoragePolicy>::value,
                  "dimensions or storage policies incompatible"); 
    if (dim() != other.dim())
      throw std::logic_error("dimensions of histograms differ");
    if (size() != other.size())
      throw std::logic_error("sizes of histograms differ");
    if (axes_ != other.axes_)
      throw std::logic_error("axes of histograms differ");
    storage_ += other.storage_;
    return *this;
  }

private:
  std::array<axis_t, Dim> axes_;
  StoragePolicy storage_;

  std::size_t field_count() const
  {
    std::size_t fc = 1;
    for (auto& a : axes_)
      fc *= apply_visitor(visitor::shape(), a);
    return fc;
  }

  template <typename First, typename... Rest>
  void assign_axis(First a, Rest... rest)
  {
    static_assert(std::is_convertible<First, axis_t>::value,
                  "argument must be axis type");
    axes_[dim() - sizeof...(Rest) - 1] = a;
    assign_axis(rest...);
  }
  void assign_axis() {} // stop recursion

  template <typename First, typename... Rest>
  void index_impl(detail::linearize_x& lin, First first, Rest... rest)
  {
    static_assert(std::is_convertible<First, double>::value,
                  "argument not convertible to double");
    lin.in = first;
    apply_visitor(lin, axes_[dim() - sizeof...(Rest) - 1]);
    index_impl(lin, rest...);
  }
  void index_impl(detail::linearize_x&) {} // stop recursion

  template <typename First, typename... Rest>
  double windex_impl(detail::linearize_x& lin, First first, Rest... rest)
  {
    static_assert(std::is_convertible<First, double>::value,
                  "argument not convertible to double");
    lin.in = first;
    apply_visitor(lin, axes_[dim() - sizeof...(Rest)]);
    return windex_impl(lin, rest...);
  }
  template <typename Last>
  double windex_impl(detail::linearize_x&, Last w)
  { 
    static_assert(std::is_convertible<Last, double>::value,
                  "argument not convertible to double");
    return w; 
  } // stop recursion

  template <typename First, typename... Rest>
  void index_impl(detail::linearize& lin, First first, Rest... rest)
  {
    static_assert(std::is_convertible<First, int>::value,
                  "argument not convertible to integer");
    lin.in = first;
    apply_visitor(lin, axes_[dim() - sizeof...(Rest) - 1]);
    index_impl(lin, rest...);      
  }
  void index_impl(detail::linearize&) {} // stop recursion
};

template <unsigned DimA, typename StoragePolicyA,
          unsigned DimB, typename StoragePolicyB>
typename std::common_type<
  histogram_t<DimA, StoragePolicyA>,
  histogram_t<DimB, StoragePolicyB>
>::type
operator+(const histogram_t<DimA, StoragePolicyA>& a,
          const histogram_t<DimB, StoragePolicyB>& b)
{
  typename std::common_type<
    histogram_t<DimA, StoragePolicyA>,
    histogram_t<DimB, StoragePolicyB>
  >::type tmp = a;
  tmp += b;
  return tmp;
}

/// Type factory
template <typename... Axes>
histogram_t<sizeof...(Axes)>
histogram(Axes... axes)
{
  return histogram_t<sizeof...(Axes)>(std::forward<Axes>(axes)...);
}

}
}

#endif
