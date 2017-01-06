// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STATIC_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_STATIC_HISTOGRAM_HPP_

#include <boost/config.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/include/sequence.hpp>
#include <boost/fusion/sequence/comparison.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/include/algorithm.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/static_storage.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <type_traits>

namespace boost {
namespace histogram {

template <typename Storage, typename Axes>
class static_histogram
{
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");

public:
  using axes_t = typename fusion::result_of::as_vector<Axes>::type;
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  static_histogram() = default;

  template <typename... Axes1>
  explicit
  static_histogram(const Axes1&... axes) :
    axes_(axes...)
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
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
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
  static_histogram(const static_histogram<OtherStorage, Axes>& other) :
    axes_(other.axes_), storage_(other.storage_)
  {}

  template <typename OtherStorage>
  static_histogram(static_histogram<OtherStorage, Axes>&& other) :
    axes_(std::move(other.axes_)),
    storage_(std::move(other.storage_))
  {}

  template <typename OtherStorage>
  static_histogram& operator=(const static_histogram<OtherStorage, Axes>& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  template <typename OtherStorage>
  static_histogram& operator=(static_histogram<OtherStorage, Axes>&& other)
  {
    axes_ = std::move(other.axes_);
    storage_ = std::move(other.storage_);
    return *this;
  }

  template <typename OtherStorage>
  bool operator==(const static_histogram<OtherStorage, Axes>& other) const
  {
    if (!axes_equal_to(other.axes_))
      return false;
    return detail::storage_content_equal(storage_, other.storage_);
  }

  template <typename OtherStorage, typename OtherAxes>
  constexpr bool operator==(const static_histogram<OtherStorage, OtherAxes>& other)
  {
    return false;
  }

  template <typename OtherStorage>
  static_histogram& operator+=(const static_histogram<OtherStorage, Axes>& other)
  {
    if (!axes_equal_to(other.axes_))
      throw std::logic_error("axes of histograms differ");
    storage_ += other.storage_;
    return *this;
  }

  template <typename... Vs>
  void fill(Vs... values)
  {
    static_assert(sizeof...(values) == dim(),
                  "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename... Args>
  void wfill(Args... args)
  {
    static_assert(std::is_same<Storage, dynamic_storage>::value,
                  "wfill only supported for dynamic_storage");
    static_assert(sizeof...(args) == (dim() + 1),
                  "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    const double w = windex_impl(lin, args...);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename... Args>
  value_t value(Args... args) const
  {
    static_assert(sizeof...(args) == dim(),
                  "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(lin.out);
  }

  template <typename... Args>
  variance_t variance(Args... args) const
  {
    static_assert(sizeof...(args) == dim(),
                  "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.variance(lin.out);
  }

  /// Number of axes (dimensions) of histogram
  static constexpr unsigned dim()
  { return fusion::result_of::size<Axes>::type::value; }

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

  template <unsigned N>
  typename std::add_const<
    typename fusion::result_of::value_at_c<axes_t, N>::type
  >::type&
  axis() const
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

  template <typename OtherAxes>
  bool axes_equal_to(const OtherAxes& other_axes) const {
    return false;
  }

  bool axes_equal_to(const axes_t& other_axes) const {
    return axes_ == other_axes;
  }

  template <typename Linearize, typename First, typename... Rest>
  void index_impl(Linearize& lin, First first, Rest... rest) const
  {
    lin.set(first);
    lin(fusion::at_c<(dim() - sizeof...(Rest) - 1)>(axes_));
    index_impl(lin, rest...);
  }

  template <typename Linearize>
  void index_impl(Linearize&) const {} // stop recursion

  template <typename Linearize, typename First, typename... Rest>
  double windex_impl(Linearize& lin, First first, Rest... rest) const
  {
    lin.set(first);
    lin(fusion::at_c<(dim() - sizeof...(Rest))>(axes_));
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
  friend class static_histogram;

  template <class Archive>
  friend void serialize(Archive&, static_histogram&, unsigned);
};


// when adding histograms with different storage, use storage with more capacity as return type
template <typename StoragePolicyA,
          typename StoragePolicyB,
          typename Axes>
inline
static_histogram<
  typename std::conditional<
    (std::numeric_limits<typename StoragePolicyA::value_t>::max() >
     std::numeric_limits<typename StoragePolicyB::value_t>::max()),
    StoragePolicyA, StoragePolicyB
  >::type,
  Axes
>
operator+(const static_histogram<StoragePolicyA, Axes>& a,
          const static_histogram<StoragePolicyB, Axes>& b)
{
  static_histogram<
    typename std::conditional<
      (std::numeric_limits<typename StoragePolicyA::value_t>::max() >
       std::numeric_limits<typename StoragePolicyB::value_t>::max()),
      StoragePolicyA, StoragePolicyB>::type,
    Axes
  > tmp = a;
  tmp += b;
  return tmp;
}


/// default static type factory
template <typename... Axes>
inline
static_histogram<dynamic_storage, mpl::vector<Axes...>>
make_static_histogram(const Axes&... axes)
{
  return static_histogram<dynamic_storage, mpl::vector<Axes...>>(axes...);
}


/// static type factory with variable storage type
template <typename Storage, typename... Axes>
inline
static_histogram<Storage, mpl::vector<Axes...>>
make_static_histogram_with(const Axes&... axes)
{
  return static_histogram<Storage, mpl::vector<Axes...>>(axes...);
}

}
}

#endif
