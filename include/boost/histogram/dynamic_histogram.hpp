// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DYNAMIC_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_DYNAMIC_HISTOGRAM_HPP_

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/variant.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/static_storage.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <boost/histogram/utility.hpp>
#include <cstddef>
#include <array>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace boost {
namespace histogram {
namespace dynamic {

template <typename Storage=dynamic_storage, typename Axes=default_axes>
class histogram
{
public:
  struct histogram_tag {};
  using axis_t = typename make_variant_over<Axes>::type;
  using axes_t = std::vector<axis_t>;
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  histogram() = default;

  template <typename... Axes1>
  explicit
  histogram(Axes1... axes) :
    axes_({axis_t(axes)...})
  {
    storage_ = Storage(field_count());
  }

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  histogram(Iterator axes_begin, Iterator axes_end)
  {
    std::copy(axes_begin, axes_end, axes_.begin());
    storage_ = Storage(field_count());
  }

  explicit
  histogram(const axes_t& axes) :
    axes_(axes)
  {
    storage_ = Storage(field_count());
  }

  explicit
  histogram(axes_t&& axes) :
    axes_(std::move(axes))
  {
    storage_ = Storage(field_count());
  }

  template <typename Histogram,
            typename = typename Histogram::histogram_tag>
  histogram(const Histogram& other) :
    axes_(other.axes_), storage_(other.storage_)
  {}

  template <typename Histogram,
            typename = typename Histogram::histogram_tag>
  histogram(Histogram&& other) :
    axes_(std::move(other.axes_)), storage_(std::move(other.storage_))
  {}

  template <typename Histogram,
            typename = typename Histogram::histogram_tag>
  histogram& operator=(const Histogram& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  template <typename Histogram,
            typename = typename Histogram::histogram_tag>
  histogram& operator=(Histogram&& other)
  {
    axes_ = std::move(other.axes_);
    storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename Histogram,
            typename = typename Histogram::histogram_tag>
  bool operator==(const Histogram& other) const
  {
    if (dim() != other.dim())
      return false;
    for (decltype(dim()) i = 0, n = dim(); i < n; ++i)
      if (!(axes_[i] == other.axes_[i]))
        return false;
    if (!(storage_ == other.storage_))
        return false;
    return true;
  }

  template <typename Histogram,
            typename = typename Histogram::histogram_tag>
  histogram& operator+=(const Histogram& other)
  {
    if (dim() != other.dim())
      throw std::logic_error("dimensions of histograms differ");
    if (size() != other.size())
      throw std::logic_error("sizes of histograms differ");
    if (axes_ != other.axes_)
      throw std::logic_error("axes of histograms differ");
    storage_ += other.storage_;
    return *this;
  }

  template <typename... Vs>
  void fill(Vs... values)
  {
    BOOST_ASSERT_MSG(sizeof...(values) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename Iterator>
  void fill_iter(Iterator values_begin, Iterator values_end)
  {
    BOOST_ASSERT_MSG(std::distance(values_begin, values_end) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    iter_args_impl(lin, values_begin, values_end);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename... Args>
  void wfill(Args... args)
  {
    BOOST_ASSERT_MSG((sizeof...(args) - 1) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    const double w = windex_impl(lin, args...);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename Iterator>
  void wfill_iter(Iterator values_begin, Iterator values_end, double w)
  {
    BOOST_ASSERT_MSG(std::distance(values_begin, values_end) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    iter_args_impl(lin, values_begin, values_end);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename... Indices>
  value_t value(Indices... indices) const
  {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, indices...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(lin.out);
  }

  template <typename Iterator>
  value_t value_iter(Iterator indices_begin, Iterator indices_end) const
  {
    BOOST_ASSERT_MSG(std::distance(indices_begin, indices_end) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    iter_args_impl(lin, indices_begin, indices_end);    
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(lin.out);
    return 0;
  }

  template <typename... Indices>
  variance_t variance(Indices... indices) const
  {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, indices...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.variance(lin.out);
  }

  template <typename Iterator>
  value_t variance_iter(Iterator indices_begin, Iterator indices_end) const
  {
    BOOST_ASSERT_MSG(std::distance(indices_begin, indices_end) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    iter_args_impl(lin, indices_begin, indices_end);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.variance(lin.out);
  }

  /// Number of axes (dimensions) of histogram
  unsigned dim() const { return axes_.size(); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const { return storage_.size(); }

  /// Access to internal data (used by Python bindings)
  const void* data() const { return storage_.data(); }

  /// Memory used by a bin in bytes
  unsigned depth() const { return storage_.depth(); }

  /// Sum of all counts in the histogram
  double sum() const
  {
    double result = 0.0;
    for (std::size_t i = 0, n = size(); i < n; ++i)
      result += storage_.value(i);
    return result;
  }

  /// Return axis \a i
  const axis_t& axis(unsigned i) const { return axes_[i]; }

private:
  axes_t axes_;
  Storage storage_;

  std::size_t field_count() const
  {
    std::size_t fc = 1;
    for (auto& a : axes_) fc *= shape(a);
    return fc;
  }

  template <typename Linearize, typename First, typename... Rest>
  void index_impl(Linearize& lin, First first, Rest... rest) const
  {
    lin.set(first);
    apply_visitor(lin, axes_[dim() - sizeof...(Rest) - 1]);
    index_impl(lin, rest...);
  }

  template <typename Linearize>
  void index_impl(Linearize& lin) const {}

  template <typename Linearize, typename First, typename... Rest>
  double windex_impl(Linearize& lin, First first, Rest... rest) const
  {
    lin.set(first);
    apply_visitor(lin, axes_[dim() - sizeof...(Rest)]);
    return windex_impl(lin, rest...);
  }

  template <typename Linearize, typename Last>
  double windex_impl(Linearize& lin, Last last) const
  {
    return last;
  }

  template <typename Linearize, typename Iterator>
  void iter_args_impl(Linearize& lin,
                      Iterator& args_begin,
                      const Iterator& args_end) const {
    auto axes_iter = axes_.begin();
    while (args_begin != args_end) {
      lin.set(*args_begin);
      apply_visitor(lin, *axes_iter);
      ++args_begin;
      ++axes_iter;
    }
  }

  template <typename OtherStorage, typename OtherAxes>
  friend class histogram;

  template <typename Archiv>
  friend void serialize(Archiv&, histogram&, unsigned);
};


// when adding histograms with different storage, use storage with more capacity as return type
template <typename StoragePolicyA,
          typename StoragePolicyB,
          typename Axes>
inline
histogram<
  typename std::conditional<
    (std::numeric_limits<typename StoragePolicyA::value_t>::max() >
     std::numeric_limits<typename StoragePolicyB::value_t>::max()),
    StoragePolicyA, StoragePolicyB
  >::type,
  Axes
>
operator+(const histogram<StoragePolicyA, Axes>& a,
          const histogram<StoragePolicyB, Axes>& b)
{
  histogram<
    typename std::conditional<
      (std::numeric_limits<typename StoragePolicyA::value_t>::max() >
       std::numeric_limits<typename StoragePolicyB::value_t>::max()),
      StoragePolicyA, StoragePolicyB>::type,
    Axes
  > tmp = a;
  tmp += b;
  return tmp;
}

/// standard type factory
template <typename... Axes>
histogram<>
make_histogram(Axes... axes)
{
  return histogram<>(std::forward<Axes>(axes)...);
}


}
}
}

#endif
