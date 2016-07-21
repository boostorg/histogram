// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_HPP_

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/visitors.hpp>
#include <boost/histogram/static_storage.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <cstddef>
#include <array>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <algorithm>

namespace boost {
namespace histogram {

/// Use dynamic dimensions
constexpr unsigned Dynamic = 0;

namespace {

  template <typename Axes>
  void axes_init(Axes&, unsigned) {}

  template <>
  void axes_init(std::vector<axis_t>& axes, unsigned n) {
    axes.resize(n);
  }

  template <typename Axes1, typename Axes2>
  void copy_axes(const Axes1 src, Axes2& dst) {
      axes_init(dst, src.size());
      std::copy(src.begin(), src.end(), dst.begin());    
  }

  /// Histogram interface and data structures common to any dimension
  template <typename Axes, typename Storage = dynamic_storage>
  class histogram_common {
  public:
    /// Number of axes (dimensions) of histogram
    unsigned dim() const { return axes_.size(); }

    /// Total number of bins in the histogram (including underflow/overflow)
    std::size_t size() const { return storage_.size(); }

    /// Access to internal data (used by Python bindings)
    const char* data() const { return storage_.data(); }

    /// Number of bins along axis \a i, including underflow/overflow
    std::size_t shape(unsigned i) const
    {
      BOOST_ASSERT(i < dim());
      return apply_visitor(visitor::shape(), axes_[i]);
    }

    /// Number of bins along axis \a i, excluding underflow/overflow
    int bins(unsigned i) const
    {
      BOOST_ASSERT(i < dim());
      return apply_visitor(visitor::bins(), axes_[i]);
    }

    /// Sum of all counts in the histogram
    double sum() const
    {
      double result = 0.0;
      for (std::size_t i = 0, n = size(); i < n; ++i)
        result += storage_.value(i);
      return result;
    }

    template <typename T = axis_t>
    typename std::enable_if<std::is_same<T, axis_t>::value, T&>::type
    axis(unsigned i) { return axes_[i]; }

    template <typename T>
    typename std::enable_if<!std::is_same<T, axis_t>::value, T&>::type
    axis(unsigned i) { return boost::get<T&>(axes_[i]); }

    template <typename T = axis_t>
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

  protected:
    Axes axes_;
    Storage storage_;

    histogram_common() = default;

    histogram_common(unsigned naxis)
    { axes_init(axes_, naxis); }

    template <typename OtherAxes, typename OtherStorage>
    histogram_common(const OtherAxes& axes,
                     const OtherStorage& storage) :
      storage_(storage)
    {
      axes_init(axes_, axes.size());
      std::copy(axes.begin(), axes.end(), axes_.begin());
    }

    template <typename OtherAxes, typename OtherStorage>
    histogram_common(OtherAxes&& axes, OtherStorage&& storage) :
      storage_(std::move(storage))
    {
      axes_init(axes_, axes.size());
      std::copy(axes.begin(), axes.end(), axes_.begin());
    }

    template <typename OtherStorage>
    histogram_common(Axes&& axes, OtherStorage&& storage) :
      axes_(std::move(axes)),
      storage_(std::move(storage))
    {}

    std::size_t field_count() const
    {
      std::size_t fc = 1;
      for (auto& a : axes_)
        fc *= apply_visitor(visitor::shape(), a);
      return fc;
    }

    template <typename First, typename... Rest>
    void axis_assign(First a, Rest... rest)
    {
      static_assert(std::is_convertible<First, axis_t>::value,
                    "argument must be axis type");
      axes_[dim() - sizeof...(Rest) - 1] = a;
      axis_assign(rest...);
    }
    void axis_assign() {} // stop recursion

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
}

template <unsigned Dim, typename Storage = dynamic_storage>
class histogram_t: public histogram_common<std::array<axis_t, Dim>, Storage>
{
  using base_t = histogram_common<std::array<axis_t, Dim>, Storage>;
public:
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  histogram_t() = default;

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t(const histogram_t<OtherDim, OtherStoragePolicy>& other) :
    base_t(other.axes_, other.storage_)
  {}

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t(histogram_t<OtherDim, OtherStoragePolicy>&& other) :
    base_t(std::move(other.axes_), std::move(other.storage_))
  {}

  template <typename... Axes>
  histogram_t(axis_t a, Axes... axes)
  {
    base_t::axis_assign(a, axes...);
    base_t::storage_ = Storage(base_t::field_count());
  }

  template <typename OtherStoragePolicy>
  histogram_t& operator=(const histogram_t<Dim, OtherStoragePolicy>& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      base_t::axes_ = other.axes_;
      base_t::storage_ = other.storage_;
    }
    return *this;
  }

  template <typename OtherStoragePolicy>
  histogram_t& operator=(const histogram_t<Dynamic, OtherStoragePolicy>& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      if (base_t::dim() != other.dim())
        throw std::logic_error("dimensions do not match");
      copy_axes(other.axes_, base_t::axes_);
      base_t::storage_ = other.storage_;
    }
    return *this;
  }

  template <typename OtherStoragePolicy>
  histogram_t& operator=(histogram_t<Dim, OtherStoragePolicy>&& other)
  {
    base_t::axes_ = std::move(other.axes_);
    base_t::storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename OtherStoragePolicy>
  histogram_t& operator=(histogram_t<Dynamic, OtherStoragePolicy>&& other)
  {
    if (base_t::dim() != other.dim())
      throw std::logic_error("dimensions do not match");
    std::copy(other.axes_.begin(), other.axes_.end(), base_t::axes_.begin());
    base_t::storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename... Args>
  void fill(Args... args)
  {
    static_assert(sizeof...(args) == Dim,
                  "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    base_t::index_impl(lin, args...);
    if (lin.stride)
      base_t::storage_.increase(lin.out);
  }

  template <typename... Args>
  void wfill(Args... args)
  {
    static_assert(sizeof...(args) == (Dim + 1),
                  "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    const double w = base_t::windex_impl(lin, args...);
    if (lin.stride)
      base_t::storage_.increase(lin.out, w);
  }

  template <typename... Args>
  value_t value(Args... args)
  {
    static_assert(sizeof...(args) == Dim,
                  "number of arguments does not match histogram dimension");
    detail::linearize lin;
    base_t::index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return base_t::storage_.value(lin.out);
  }

  template <typename... Args>
  variance_t variance(Args... args)
  {
    static_assert(sizeof...(args) == Dim,
                  "number of arguments does not match histogram dimension");
    detail::linearize lin;
    base_t::index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return base_t::storage_.variance(lin.out);
  }

  template <unsigned OtherDim, typename OtherStoragePolicy>
  bool operator==(const histogram_t<OtherDim, OtherStoragePolicy>& other) const
  {
    if (base_t::dim() != other.dim())
      return false;
    for (unsigned i = 0, n = base_t::dim(); i < n; ++i)
      if (!(base_t::axes_[i] == other.axes_[i]))
        return false;
    for (std::size_t i = 0, n = base_t::size(); i < n; ++i)
      if (base_t::storage_.value(i) != other.storage_.value(i))
        return false;
    return true;
  }

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t& operator+=(const histogram_t<OtherDim, OtherStoragePolicy>& other)
  {
    static_assert((OtherDim == Dim) || OtherDim == Dynamic,
                  "dimensions incompatible"); 
    if (base_t::dim() != other.dim())
      throw std::logic_error("dimensions of histograms differ");
    if (base_t::size() != other.size())
      throw std::logic_error("sizes of histograms differ");
    if (base_t::axes_ != other.axes_)
      throw std::logic_error("axes of histograms differ");
    base_t::storage_ += other.storage_;
    return *this;
  }

  template <unsigned OtherDim, typename OtherStorage>
  friend class histogram_t;
};


template <typename Storage>
class histogram_t<Dynamic, Storage>: public histogram_common<std::vector<axis_t>, Storage>
{
  using base_t = histogram_common<std::vector<axis_t>, Storage>;
public:
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  histogram_t() = default;

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t(const histogram_t<OtherDim, OtherStoragePolicy>& other) :
    base_t(other.axes_, other.storage_)
  {}

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t(histogram_t<OtherDim, OtherStoragePolicy>&& other) :
    base_t(std::move(other.axes_), std::move(other.storage_))
  {}

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t& operator=(const histogram_t<OtherDim, OtherStoragePolicy>& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      copy_axes(other.axes_, base_t::axes_);
      base_t::storage_ = other.storage_;
    }
    return *this;
  }

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t& operator=(histogram_t<OtherDim, OtherStoragePolicy>&& other)
  {
    copy_axes(other.axes_, base_t::axes_);
    base_t::storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename OtherStoragePolicy>
  histogram_t& operator=(histogram_t<Dynamic, OtherStoragePolicy>&& other)
  {
    base_t::axes_ = std::move(other.axes_);
    base_t::storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename... Axes>
  histogram_t(axis_t a, Axes... axes)
    : base_t(1 + sizeof...(axes))
  {
    base_t::axis_assign(a, axes...);
    base_t::storage_ = Storage(base_t::field_count());
  }

  template <typename... Args>
  void fill(Args... args)
  {
    BOOST_ASSERT_MSG(sizeof...(args) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, args...);
    if (lin.stride)
      base_t::storage_.increase(lin.out);
  }

  template <typename... Args>
  void wfill(Args... args)
  {
    BOOST_ASSERT_MSG(sizeof...(args) == (base_t::dim() + 1),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    const double w = windex_impl(lin, args...);
    if (lin.stride)
      base_t::storage_.increase(lin.out, w);
  }

  template <typename... Args>
  value_t value(Args... args)
  {
    BOOST_ASSERT_MSG(sizeof...(args) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return base_t::storage_.value(lin.out);
  }

  template <typename... Args>
  variance_t variance(Args... args)
  {
    BOOST_ASSERT_MSG(sizeof...(args) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return base_t::storage_.variance(lin.out);
  }

  template <unsigned OtherDim, typename OtherStoragePolicy>
  bool operator==(const histogram_t<OtherDim, OtherStoragePolicy>& other) const
  {
    if (base_t::dim() != other.dim())
      return false;
    for (unsigned i = 0, n = base_t::dim(); i < n; ++i)
      if (!(base_t::axes_[i] == other.axes_[i]))
        return false;
    for (std::size_t i = 0, n = base_t::size(); i < n; ++i)
      if (base_t::storage_.value(i) != other.storage_.value(i))
        return false;
    return true;
  }

  template <unsigned OtherDim, typename OtherStoragePolicy>
  histogram_t& operator+=(const histogram_t<OtherDim, OtherStoragePolicy>& other)
  {
    if (base_t::dim() != other.dim())
      throw std::logic_error("dimensions of histograms differ");
    if (base_t::size() != other.size())
      throw std::logic_error("sizes of histograms differ");
    if (base_t::axes_ != other.axes_)
      throw std::logic_error("axes of histograms differ");
    base_t::storage_ += other.storage_;
    return *this;
  }

  // all histogram types are friends to share access of private members
  template <unsigned OtherDim, typename OtherStorage>
  friend class histogram_t;
};


template <unsigned DimA, typename StoragePolicyA,
          unsigned DimB, typename StoragePolicyB>
histogram_t<
  (DimA > DimB ? DimA : DimB),
  typename std::conditional<(sizeof(typename StoragePolicyA::value_t) >
                             sizeof(typename StoragePolicyB::value_t)),
    StoragePolicyA, StoragePolicyB>::type
>
operator+(const histogram_t<DimA, StoragePolicyA>& a,
          const histogram_t<DimB, StoragePolicyB>& b)
{
  histogram_t<
    (DimA > DimB ? DimA : DimB),
    typename std::conditional<(sizeof(typename StoragePolicyA::value_t) >
                               sizeof(typename StoragePolicyB::value_t)),
      StoragePolicyA, StoragePolicyB>::type
  > tmp = a;
  tmp += b;
  return tmp;
}


/// Standard type factory
template <typename... Axes>
histogram_t<sizeof...(Axes)>
histogram(Axes... axes)
{
  return histogram_t<sizeof...(Axes)>(std::forward<Axes>(axes)...);
}


}
}

#endif
