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
#include <boost/histogram/storage.hpp>
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

  template <unsigned Dim, typename StoragePolicy = static_storage<unsigned> >
  class histogram_t : private StoragePolicy
  {
  public:
    using value_t = typename StoragePolicy::value_t;
    using variance_t = typename StoragePolicy::variance_t;

    histogram_t() = default;
    histogram_t(const histogram_t& other) = default;
    histogram_t(histogram_t&& other) = default;
    histogram_t& operator=(const histogram_t& other) = default;
    histogram_t& operator=(histogram_t&& other)
    {
      if (this != &other) {
        axes_ = std::move(other.axes_);
        StoragePolicy::operator=(static_cast<StoragePolicy&&>(other));        
      }
      return *this;
    }

    template <typename... Axes>
    histogram_t(axis_t a, Axes... axes)
    {
      assign_axis(a, axes...);
      StoragePolicy::allocate(field_count());
    }

    constexpr unsigned dim() const { return Dim; }

    std::size_t size() const { return StoragePolicy::size(); }

    // for convenience
    std::size_t shape(unsigned i) const
    {
      BOOST_ASSERT(i < Dim);
      return apply_visitor(visitor::shape(), axes_[i]);
    }

    // for convenience
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
      fill_impl(lin, std::forward<Args>(args)...);
      if (lin.stride)
        StoragePolicy::increase(lin.out);
    }

    template <typename... Args>
    value_t value(Args... args)
    {
      static_assert(sizeof...(args) == Dim,
                    "number of arguments does not match histogram dimension");
      detail::linearize lin;
      index_impl(lin, std::forward<Args>(args)...);
      if (lin.stride == 0)
        throw std::out_of_range("invalid index");
      return StoragePolicy::value(lin.out);
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
      return StoragePolicy::variance(lin.out);
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
        result += StoragePolicy::value(i);
      return result;
    }

    template <unsigned OtherDim>
    bool operator==(const histogram_t<OtherDim, StoragePolicy>& other) const
    {
      return dim() == other.dim() && axes_ == other.axes_ && 
             StoragePolicy::operator==(static_cast<const StoragePolicy&>(other));
    }

    template <unsigned OtherDim, typename OtherStoragePolicy>
    bool operator==(const histogram_t<OtherDim, OtherStoragePolicy>&) const
    { return false; }

    template <unsigned OtherDim, typename OtherStoragePolicy>
    histogram_t& operator+=(const histogram_t<OtherDim, OtherStoragePolicy>& other)
    {
      static_assert(std::is_same<StoragePolicy, OtherStoragePolicy>::value,
                    "dimensions or storage policies incompatible"); 
      if (dim() != other.dim())
        throw std::logic_error("dimensions of histograms differ");
      if (size() != other.size())
        throw std::logic_error("sizes of histograms differ");
      if (axes_ != other.axes_)
        throw std::logic_error("axes of histograms differ");
      StoragePolicy::operator+=(static_cast<const StoragePolicy&>(other));
      return *this;
    }

  private:
    std::array<axis_t, Dim> axes_;

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
      axes_[dim() - sizeof...(Rest) - 1] = a;
      assign_axis(rest...);
    }
    void assign_axis() {} // stop recursion

    template <typename First, typename... Rest>
    void fill_impl(detail::linearize_x& lin, First first, Rest... rest)
    {
      lin.in = first;
      apply_visitor(lin, axes_[dim() - sizeof...(Rest) - 1]);
      fill_impl(lin, rest...);
    }
    void fill_impl(detail::linearize_x&) {} // stop recursion

    template <typename First, typename... Rest>
    void index_impl(detail::linearize& lin, First first, Rest... rest)
    {
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
