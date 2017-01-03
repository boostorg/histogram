// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_HPP_

#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/variant.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/visitors.hpp>
#include <boost/histogram/static_storage.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <cstddef>
#include <array>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <utility>

namespace boost {
namespace histogram {
namespace static {

template <typename Storage, typename... Axes>
class dynamic_histogram
{
public:
  using axis_t = boost::variant<Axes...>;
  using axes_t = std::vector<axis_t>;
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  dynamic_histogram() = default;

  template <typename... Axes1>
  dynamic_histogram(Axes1... axes) :
    axes_(std::forward<Axes1>(axes)...)
  {
    storage_ = Storage(field_count());
  }

  template <typename Iterator,
            typename = is_iterator<Iterator>>
  dynamic_histogram(Iterator axes_begin, Iterator axes_end)
  {
    std::copy(axes_begin, axes_end, axes_.begin());
    storage_ = Storage(field_count());
  }

  template <typename Histogram>
  dynamic_histogram(const Histogram& other) :
    axes_(other.axes_), storage_(other.storage_)
  {}

  template <typename Histogram>
  dynamic_histogram(Histogram&& other) :
    axes_(std::move(other.axes_)), storage_(std::move(other.storage_))
  {}

  template <typename Histogram>
  dynamic_histogram& operator=(const Histogram& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  template <typename Histogram>
  dynamic_histogram& operator=(Histogram&& other)
  {
    axes_ = std::move(other.axes_);
    storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename Histogram>
  bool operator==(const Histogram& other) const
  {
    if (dim() != other.dim())
      return false;
    for (decltype(dim()) i = 0, n = dim(); i < n; ++i)
      if (!(axes_[i] == other.axes_[i]))
        return false;
    for (decltype(size()) i = 0, n = size(); i < n; ++i)
      if (storage_.value(i) != other.storage_.value(i))
        return false;
    return true;
  }

  template <typename Histogram>
  dynamic_histogram& operator+=(const Histogram& other)
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
    linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename Iterator,
            typename = is_iterator<Iterator>>
  void fill(Iterator values_begin, Iterator values_end)
  {
    BOOST_ASSERT_MSG(std::distance(values_begin, values_end) == dim(),
                     "number of arguments does not match histogram dimension");
    linearize_x lin;
    auto axes_iter = axes_.begin();
    while (values_begin != values_end) {
      apply_visitor(lin, *axes_iter, *values_begin);
      ++values_begin;
      ++axes_iter;
    }
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename... Vs>
  void wfill(Vs... values, double w)
  {
    BOOST_ASSERT_MSG(sizeof...(values) == dim(),
                     "number of arguments does not match histogram dimension");
    linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename Iterator,
            typename = is_iterator<Iterator>>
  void wfill(Iterator values_begin, Iterator values_end, double w)
  {
    BOOST_ASSERT_MSG(std::distance(values_begin, values_end) == dim(),
                     "number of arguments does not match histogram dimension");
    linearize_x lin;
    auto axes_iter = axes_.begin();
    while (values_begin != values_end) {
      apply_visitor(lin, *axes_iter, *values_begin);
      ++values_begin;
      ++axes_iter;
    }
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename... Indices>
  value_t value(Indices... indices) const
  {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    linearize lin;
    index_impl(lin, indices...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(lin.out);
  }

  template <typename Iterator,
            typename = is_iterator<Iterator>>
  value_t value(Iterator indices_begin, Iterator indices_end) const
  {
    BOOST_ASSERT_MSG(std::distance(indices_begin, indices_end) == dim(),
                     "number of arguments does not match histogram dimension");
    linearize lin;
    auto axes_iter = axes_.begin();
    while (indices_begin != indices_end) {
      apply_visitor(lin, *axes_iter, *indices_begin);
      ++indices_begin;
      ++axes_iter;
    }
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(lin.out);
  }

  template <typename... Indices>
  variance_t variance(Indices... indices) const
  {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    linearize lin;
    index_impl(lin, indices...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.variance(lin.out);
  }

  template <typename Iterator,
            typename = is_iterator<Iterator>>
  value_t variance(Iterator indices_begin, Iterator indices_end) const
  {
    BOOST_ASSERT_MSG(std::distance(indices_begin, indices_end) == dim(),
                     "number of arguments does not match histogram dimension");
    linearize lin;
    auto axes_iter = axes_.begin();
    while (indices_begin != indices_end) {
      apply_visitor(lin, *axes_iter, *indices_begin);
      ++indices_begin;
      ++axes_iter;
    }
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

  // /// Number of bins along axis \a i, including underflow/overflow
  // std::size_t shape(unsigned i) const
  // {
  //   BOOST_ASSERT(i < dim());
  //   return apply_visitor(visitor::shape(), axes_[i]);
  // }

  // /// Number of bins along axis \a i, excluding underflow/overflow
  // int bins(unsigned i) const
  // {
  //   BOOST_ASSERT(i < dim());
  //   return apply_visitor(visitor::bins(), axes_[i]);
  // }

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
    visitor::shape s;
    for (auto& a : axes_)
      fc *= apply_visitor(s, a);
    return fc;
  }

  template <typename Linearize, typename First, typename... Rest>
  void index_impl(Linearize& lin, First first, Rest... rest) const
  {
    apply_visitor(lin, axes_[dim() - sizeof...(Rest) - 1], first);
    index_impl(lin, rest...);
  }

  template <typename Linearize>
  void index_impl(Linearize&) const {} // stop recursion

  template <class Archive>
  friend void serialize(Archive&, dynamic_histogram&, unsigned);
};


template <typename Storage, typename... Axes>
class static_histogram
{
public:
  using axes_t = std::tuple<Axes...>;
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  static_histogram() = default;

  static_histogram(Axes... axes) :
    axes_(std::forward<Axes...>(axes...))
  {
    storage_ = Storage(field_count());
  }

  template <typename OtherStorage>
  static_histogram(const static_histogram<OtherStorage, Axes...>& other) :
    axes_(other.axes_), storage_(other.storage_)
  {}

  template <typename OtherStorage>
  static_histogram(static_histogram<OtherStorage, Axes...>&& other) :
    axes_(std::move(other.axes_)), storage_(std::move(other.storage_))
  {}

  template <typename OtherStorage>
  static_histogram& operator=(const static_histogram<OtherStorage, Axes...>& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  template <typename OtherStorage>
  static_histogram& operator=(static_histogram<OtherStorage, Axes...>&& other)
  {
    axes_ = std::move(other.axes_);
    storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename Histogram>
  bool operator==(const Histogram& other) const
  {
    if (axes_ != other.axes_)
      return false;
    for (decltype(size()) i = 0, n = size(); i < n; ++i)
      if (storage_.value(i) != other.storage_.value(i))
        return false;
    return true;
  }

  template <typename Histogram>
  static_histogram& operator+=(const Histogram& other)
  {
    if (axes_ != other.axes_)
      throw std::logic_error("axes of histograms differ");
    storage_ += other.storage_;
    return *this;
  }

  // template <typename... Vs>
  // void fill(Vs... values)
  // {
  //   BOOST_ASSERT_MSG(sizeof...(values) == dim(),
  //                    "number of arguments does not match histogram dimension");
  //   linearize_x lin;
  //   index_impl(lin, values...);
  //   if (lin.stride)
  //     storage_.increase(lin.out);
  // }

  // template <typename... Args>
  // void wfill(Args... args)
  // {
  //   BOOST_ASSERT_MSG(sizeof...(args) == (dim() + 1),
  //                    "number of arguments does not match histogram dimension");
  //   linearize_x lin;
  //   const double w = windex_impl(lin, args...);
  //   if (lin.stride)
  //     storage_.increase(lin.out, w);
  // }

  // template <typename... Args>
  // value_t value(Args... args) const
  // {
  //   BOOST_ASSERT_MSG(sizeof...(args) == dim(),
  //                    "number of arguments does not match histogram dimension");
  //   linearize lin;
  //   index_impl(lin, args...);
  //   if (lin.stride == 0)
  //     throw std::out_of_range("invalid index");
  //   return storage_.value(lin.out);
  // }

  // template <typename... Args>
  // variance_t variance(Args... args) const
  // {
  //   BOOST_ASSERT_MSG(sizeof...(args) == dim(),
  //                    "number of arguments does not match histogram dimension");
  //   linearize lin;
  //   index_impl(lin, args...);
  //   if (lin.stride == 0)
  //     throw std::out_of_range("invalid index");
  //   return storage_.variance(lin.out);
  // }

  /// Number of axes (dimensions) of histogram
  constexpr unsigned dim() const { return sizeof...(Axes); }

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
    for (decltype(size()) i = 0, n = size(); i < n; ++i)
      result += storage_.value(i);
    return result;
  }

  /// Return axis \a i
  template <unsigned N>
  const typename std::tuple_element<N, axes_t>::type& axis() const { return std::get<N>(axes_); }

private:
  axes_t axes_;
  Storage storage_;

  std::size_t field_count() const
  {
    std::size_t fc = 1;
    field_count_impl<Axes...>(fc);
    return fc;
  }

  template <typename First, typename... Rest>
  void field_count_impl(std::size_t& fc) {
    fc *= std::get<(sizeof...(Axes) - sizeof...(Rest) - 1)>(axes_).shape();
  }

  template <typename Linearize, typename First, typename... Rest>
  void index_impl(Linearize& lin, First first, Rest... rest) const
  {
    apply_visitor(lin, axes_[dim() - sizeof...(Rest) - 1], first);
    index_impl(lin, rest...);
  }

  template <typename Linearize>
  void index_impl(Linearize&) const {} // stop recursion

  template <class Archive>
  friend void serialize(Archive&, static_histogram&, unsigned);
};


// // when adding different types, use the more flexible type as return type
// template <template<typename..., typename> class HistogramA,
//           template<typename..., typename> class HistogramB,
//           typename StoragePolicyA,
//           typename StoragePolicyB>
// inline
// dynamic_histogram<
//   typename std::conditional<
//     (sizeof(typename StoragePolicyA::value_t) >
//      sizeof(typename StoragePolicyB::value_t)),
//     StoragePolicyA, StoragePolicyB>::type
// >
// operator+(const dynamic_histogram<StoragePolicyA>& a,
//           const dynamic_histogram<StoragePolicyB>& b)
// {
//   dynamic_histogram<
//     typename std::conditional<
//       (sizeof(typename StoragePolicyA::value_t) >
//        sizeof(typename StoragePolicyB::value_t)),
//       StoragePolicyA, StoragePolicyB>::type
//   > tmp = a;
//   tmp += b;
//   return tmp;
// }


/// static type factory
template <typename... Axes>
inline
static_histogram<Axes...>
make_histogram(Axes... axes)
{
  return static_histogram<Axes...>(std::forward<Axes>(axes)...);
}


}
}
}

#endif
