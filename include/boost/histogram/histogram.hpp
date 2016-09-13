// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_HPP_

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/variant/static_visitor.hpp>
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
  inline void axes_init(Axes&, unsigned) {}

  template <>
  inline void axes_init(std::vector<axis_t>& axes, unsigned n) {
    axes.resize(n);
  }

  template <typename Axes1, typename Axes2>
  inline void copy_axes(const Axes1 src, Axes2& dst) {
      axes_init(dst, src.size());
      std::copy(src.begin(), src.end(), dst.begin());    
  }

  struct linearize : public static_visitor<void>
  {
    int in;
    std::size_t out = 0, stride = 1;

    template <typename A>
    void operator()(const A& a) {
      // The following is highly optimized code that runs in a hot loop.
      // If you change it, please also measure the performance impact.
      int j = in;
      const int uoflow = a.uoflow();
      // set stride to zero if j is not in range,
      // this communicates the out-of-range condition to the caller
      stride *= (j >= -uoflow) * (j < (a.bins() + uoflow));
      j += (j < 0) * (a.bins() + 2); // wrap around if j < 0
      out += j * stride;
      #pragma GCC diagnostic ignored "-Wstrict-overflow"
      stride *= a.shape();
    }
  };

  struct linearize_x : public static_visitor<void>
  {
    double in;
    std::size_t out = 0, stride = 1;

    template <typename A>
    void operator()(const A& a) {
      int j = a.index(in); // j is guaranteed to be in range [-1, bins]
      j += (j < 0) * (a.bins() + 2); // wrap around if j < 0
      out += j * stride;
      #pragma GCC diagnostic ignored "-Wstrict-overflow"
      stride *= (j < a.shape()) * a.shape(); // stride == 0 indicates out-of-range
    }
  };

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
  const void* data() const { return storage_.data(); }

  /// Memory used by a bin in bytes
  unsigned depth() const { return storage_.depth(); }

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
  axis(unsigned i) { return boost::get<T>(axes_[i]); }

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

  template <class Archive>
  friend void serialize(Archive&, histogram_common&, unsigned);

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
  void index_impl(linearize_x& lin, First first, Rest... rest) const
  {
    static_assert(std::is_convertible<First, double>::value,
                  "argument not convertible to double");
    lin.in = first;
    apply_visitor(lin, axes_[dim() - sizeof...(Rest) - 1]);
    index_impl(lin, rest...);
  }
  void index_impl(linearize_x&) const {} // stop recursion

  template <typename First, typename... Rest>
  double windex_impl(linearize_x& lin, First first, Rest... rest) const
  {
    static_assert(std::is_convertible<First, double>::value,
                  "argument not convertible to double");
    lin.in = first;
    apply_visitor(lin, axes_[dim() - sizeof...(Rest)]);
    return windex_impl(lin, rest...);
  }
  template <typename Last>
  double windex_impl(linearize_x&, Last w) const
  { 
    static_assert(std::is_convertible<Last, double>::value,
                  "argument not convertible to double");
    return w; 
  } // stop recursion

  template <typename First, typename... Rest>
  void index_impl(linearize& lin, First first, Rest... rest) const
  {
    static_assert(std::is_convertible<First, int>::value,
                  "argument not convertible to integer");
    lin.in = first;
    apply_visitor(lin, axes_[dim() - sizeof...(Rest) - 1]);
    index_impl(lin, rest...);      
  }
  void index_impl(linearize&) const {} // stop recursion
};

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
  histogram_t(Axes... axes)
  {
    base_t::axes_ = {{std::forward<Axes>(axes)...}};
    base_t::storage_ = Storage(base_t::field_count());
  }

  template <typename Iterator,
            typename = decltype(*std::declval<Iterator&>(), void(), ++std::declval<Iterator&>(), void())>
  histogram_t(Iterator axes_begin, Iterator axes_end)
  {
    std::copy(axes_begin, axes_end, base_t::axes_.begin());
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
    linearize_x lin;
    base_t::index_impl(lin, args...);
    if (lin.stride)
      base_t::storage_.increase(lin.out);
  }

  template <typename... Args>
  void wfill(Args... args)
  {
    static_assert(sizeof...(args) == (Dim + 1),
                  "number of arguments does not match histogram dimension");
    linearize_x lin;
    const double w = base_t::windex_impl(lin, args...);
    if (lin.stride)
      base_t::storage_.increase(lin.out, w);
  }

  template <typename... Args>
  value_t value(Args... args) const
  {
    static_assert(sizeof...(args) == Dim,
                  "number of arguments does not match histogram dimension");
    linearize lin;
    base_t::index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return base_t::storage_.value(lin.out);
  }

  template <typename... Args>
  variance_t variance(Args... args) const
  {
    static_assert(sizeof...(args) == Dim,
                  "number of arguments does not match histogram dimension");
    linearize lin;
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
  template <class Archive>
  friend void serialize(Archive&, histogram_t&, unsigned);
};


template <typename Storage>
class histogram_t<Dynamic, Storage>: public histogram_common<std::vector<axis_t>, Storage>
{
public:
  using base_t = histogram_common<std::vector<axis_t>, Storage>;
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  histogram_t() = default;

  histogram_t(const std::vector<axis_t>& axes)
  {
    base_t::axes_ = axes;
    base_t::storage_ = Storage(base_t::field_count());
  }

  histogram_t(std::vector<axis_t>&& axes)
  {
    base_t::axes_ = std::move(axes);
    base_t::storage_ = Storage(base_t::field_count());
  }

  template <typename Iterator,
            typename = decltype(*std::declval<Iterator&>(), void(), ++std::declval<Iterator&>(), void())>
  histogram_t(Iterator axes_begin, Iterator axes_end)
  {
    std::copy(axes_begin, axes_end, base_t::axes_.begin());
    base_t::storage_ = Storage(base_t::field_count());
  }

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
  histogram_t(Axes... axes)
  {
    base_t::axes_ = {{std::forward<Axes>(axes)...}};
    base_t::storage_ = Storage{base_t::field_count()};
  }

  template <typename... Args>
  void fill(Args... args)
  {
    BOOST_ASSERT_MSG(sizeof...(args) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    linearize_x lin;
    base_t::index_impl(lin, args...);
    if (lin.stride)
      base_t::storage_.increase(lin.out);
  }

  template <typename Iterator>
  void fill_iter(Iterator args_begin, Iterator args_end)
  {
    BOOST_ASSERT_MSG(std::distance(args_begin, args_end) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    linearize_x lin;
    auto axes_iter = base_t::axes_.begin();
    while (args_begin != args_end) {
      lin.in = *args_begin;
      apply_visitor(lin, *axes_iter);
      ++args_begin;
      ++axes_iter;
    }
    if (lin.stride)
      base_t::storage_.increase(lin.out);
  }

  template <typename... Args>
  void wfill(Args... args)
  {
    BOOST_ASSERT_MSG(sizeof...(args) == (base_t::dim() + 1),
                     "number of arguments does not match histogram dimension");
    linearize_x lin;
    const double w = base_t::windex_impl(lin, args...);
    if (lin.stride)
      base_t::storage_.increase(lin.out, w);
  }

  template <typename Iterator>
  void wfill_iter(Iterator args_begin, Iterator args_end, double w)
  {
    BOOST_ASSERT_MSG(std::distance(args_begin, args_end) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    linearize_x lin;
    auto axes_iter = base_t::axes_.begin();
    while (args_begin != args_end) {
      lin.in = *args_begin;
      apply_visitor(lin, *axes_iter);
      ++args_begin;
      ++axes_iter;
    }
    if (lin.stride)
      base_t::storage_.increase(lin.out, w);
  }

  template <typename... Args>
  value_t value(Args... args) const
  {
    BOOST_ASSERT_MSG(sizeof...(args) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    linearize lin;
    base_t::index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return base_t::storage_.value(lin.out);
  }

  template <typename Iterator>
  value_t value_iter(Iterator args_begin, Iterator args_end) const
  {
    BOOST_ASSERT_MSG(std::distance(args_begin, args_end) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    linearize lin;
    auto axes_iter = base_t::axes_.begin();
    while (args_begin != args_end) {
      lin.in = *args_begin;
      apply_visitor(lin, *axes_iter);
      ++args_begin;
      ++axes_iter;
    }
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return base_t::storage_.value(lin.out);
  }

  template <typename... Args>
  variance_t variance(Args... args) const
  {
    BOOST_ASSERT_MSG(sizeof...(args) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    linearize lin;
    base_t::index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return base_t::storage_.variance(lin.out);
  }

  template <typename Iterator>
  value_t variance_iter(Iterator args_begin, Iterator args_end) const
  {
    BOOST_ASSERT_MSG(std::distance(args_begin, args_end) == base_t::dim(),
                     "number of arguments does not match histogram dimension");
    linearize lin;
    auto axes_iter = base_t::axes_.begin();
    while (args_begin != args_end) {
      lin.in = *args_begin;
      apply_visitor(lin, *axes_iter);
      ++args_begin;
      ++axes_iter;
    }
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
  template <class Archive>
  friend void serialize(Archive&, histogram_t&, unsigned);
};


template <unsigned DimA, typename StoragePolicyA,
          unsigned DimB, typename StoragePolicyB>
inline
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
inline
histogram_t<sizeof...(Axes)>
histogram(Axes... axes)
{
  return histogram_t<sizeof...(Axes)>(std::forward<Axes>(axes)...);
}


}
}

#endif
