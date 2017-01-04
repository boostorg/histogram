// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STATIC_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_STATIC_HISTOGRAM_HPP_

#include <boost/config.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/include/sequence.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/include/algorithm.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/static_storage.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <type_traits>

namespace boost {
namespace histogram {

template <typename Storage, typename Axis, typename... Axes>
class static_histogram
{
public:
  using axes_t = fusion::vector<const Axis, const Axes...>;
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  static_histogram() = delete;

  static_histogram(Axis axis, Axes... axes) :
    axes_(axis, axes...)
  {
    storage_ = Storage(field_count());
  }

  static_histogram(const static_histogram& other) :
    axes_(other.axes_), storage_(other.storage_)
  {}

  static_histogram(static_histogram&& other) :
    axes_(std::move(other.axes_)),
    storage_(std::move(other.storage_))
  {}

  static_histogram& operator=(const static_histogram& other)
  {
    if (this != &other) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  static_histogram& operator=(static_histogram&& other)
  {
    axes_ = std::move(other.axes_);
    storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename OtherStorage>
  bool operator==(const static_histogram<OtherStorage, Axis, Axes...>& other) const
  {
    if (axes_ != other.axes_)
      return false;
    for (decltype(size()) i = 0, n = size(); i < n; ++i)
      if (storage_.value(i) != other.storage_.value(i))
        return false;
    return true;
  }

  template <typename OtherStorage, typename OtherAxis, typename... OtherAxes>
  bool operator==(const static_histogram<OtherStorage, OtherAxis, OtherAxes...>& other) const
  {
    return false;
  }

  template <typename OtherStorage>
  static_histogram& operator+=(const static_histogram<OtherStorage, Axis, Axes...>& other)
  {
    if (axes_ != other.axes_)
      throw std::logic_error("axes of histograms differ");
    storage_ += other.storage_;
    return *this;
  }

  template <typename... Vs>
  void fill(Vs... values)
  {
    static_assert(sizeof...(values) == dim(),
                  "number of arguments does not match static_histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  // template <typename... Args>
  // void wfill(Args... args)
  // {
  //   BOOST_ASSERT_MSG(sizeof...(args) == (dim() + 1),
  //                    "number of arguments does not match static_histogram dimension");
  //   linearize_x lin;
  //   const double w = windex_impl(lin, args...);
  //   if (lin.stride)
  //     storage_.increase(lin.out, w);
  // }

  // template <typename... Args>
  // value_t value(Args... args) const
  // {
  //   BOOST_ASSERT_MSG(sizeof...(args) == dim(),
  //                    "number of arguments does not match static_histogram dimension");
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
  //                    "number of arguments does not match static_histogram dimension");
  //   linearize lin;
  //   index_impl(lin, args...);
  //   if (lin.stride == 0)
  //     throw std::out_of_range("invalid index");
  //   return storage_.variance(lin.out);
  // }

  /// Number of axes (dimensions) of static_histogram
  constexpr unsigned dim() const { return fusion::size(axes_); }

  /// Total number of bins in the static_histogram (including underflow/overflow)
  std::size_t size() const { return storage_.size(); }

  /// Access to internal data (used by Python bindings)
  const void* data() const { return storage_.data(); }

  /// Memory used by a bin in bytes
  unsigned depth() const { return storage_.depth(); }

  /// Sum of all counts in the static_histogram
  double sum() const
  {
    double result = 0.0;
    for (decltype(size()) i = 0, n = size(); i < n; ++i)
      result += storage_.value(i);
    return result;
  }

  template <unsigned N>
  typename fusion::result_of::at_c<axes_t, N>::type& axis() const
  { return fusion::at_c<N>(axes_); }

private:
  axes_t axes_;
  Storage storage_;

  struct field_counter {
    mutable std::size_t value = 1;
    template <typename T>
    void operator()(const T& t) const { value *= t.shape(); }
  };

  std::size_t field_count() const
  {
    field_counter fc;
    fusion::for_each(axes_, fc);
    return fc.value;
  }

  // template <typename Linearize, typename First, typename... Rest>
  // void index_impl(Linearize& lin, First first, Rest... rest) const
  // {
  //   lin.set(first);
  //   lin(std::get<(dim() - sizeof...(Rest) - 1)>(axes_));
  //   index_impl(lin, rest...);
  // }

  // template <typename Linearize>
  // void index_impl(Linearize&) const {} // stop recursion

  template <typename OtherStorage, typename OtherAxis, typename... OtherAxes>
  friend class static_histogram;

  template <class Archive>
  friend void serialize(Archive&, static_histogram&, unsigned);
};


/// static type factory
template <typename Axis, typename... Axes>
inline
static_histogram<dynamic_storage, Axis, Axes...>
make_static_histogram(const Axis& axis, const Axes&... axes)
{
  return static_histogram<dynamic_storage, Axis, Axes...>(axis, axes...);
}

/// static type factory
template <typename Storage, typename Axis, typename... Axes>
inline
static_histogram<Storage, Axis, Axes...>
make_static_histogram_with(const Axis& axis, const Axes&... axes)
{
  return static_histogram<Storage, Axis, Axes...>(axis, axes...);
}

}
}

#endif
