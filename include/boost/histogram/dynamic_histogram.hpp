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
#include <boost/mpl/empty.hpp>
#include <boost/variant.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/static_storage.hpp>
#include <boost/histogram/dynamic_storage.hpp>
#include <boost/histogram/utility.hpp>
#include <boost/histogram/detail/mpl.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <cstddef>
#include <array>
#include <vector>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <iterator>

namespace boost {
namespace histogram {

template <typename Storage=dynamic_storage, typename Axes=default_axes>
class dynamic_histogram
{
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");

public:
  using axis_t = typename make_variant_over<Axes>::type;
  using axes_t = std::vector<axis_t>;
  using value_t = typename Storage::value_t;
  using variance_t = typename Storage::variance_t;

  dynamic_histogram() = default;

  template <typename... Axes1>
  explicit
  dynamic_histogram(const Axes1&... axes) :
    axes_({axis_t(axes)...})
  {
    storage_ = Storage(field_count());
  }

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  dynamic_histogram(Iterator axes_begin, Iterator axes_end)
  {
    std::copy(axes_begin, axes_end, axes_.begin());
    storage_ = Storage(field_count());
  }

  explicit
  dynamic_histogram(const axes_t& axes) :
    axes_(axes)
  {
    storage_ = Storage(field_count());
  }

  explicit
  dynamic_histogram(axes_t&& axes) :
    axes_(std::move(axes))
  {
    storage_ = Storage(field_count());
  }

  template <typename OtherStorage, typename OtherAxes>
  dynamic_histogram(const dynamic_histogram<OtherStorage, OtherAxes>& other) :
    axes_(other.axes_.begin(), other.axes_.end()), storage_(other.storage_)
  {}

  template <typename OtherStorage, typename OtherAxes>
  dynamic_histogram(dynamic_histogram<OtherStorage, OtherAxes>&& other) :
    axes_(std::move(other.axes_)), storage_(std::move(other.storage_))
  {}

  template <typename OtherStorage, typename OtherAxes>
  dynamic_histogram& operator=(const dynamic_histogram<OtherStorage, OtherAxes>& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  template <typename OtherStorage, typename OtherAxes>
  dynamic_histogram& operator=(dynamic_histogram<OtherStorage, OtherAxes>&& other)
  {
    axes_ = std::move(other.axes_);
    storage_ = std::move(other.storage_);
    return *this;
  } 

  template <typename OtherStorage, typename OtherAxes>
  bool operator==(const dynamic_histogram<OtherStorage, OtherAxes>& other) const
  {
    if (mpl::empty<
          typename detail::intersection<Axes, OtherAxes>::type
        >::value)
      return false;
    if (dim() != other.dim())
      return false;
    if (!axes_equal_to(other.axes_))
      return false;
    if (!detail::storage_content_equal(storage_, other.storage_))
      return false;
    return true;
  }

  template <typename OtherStorage, typename OtherAxes>
  dynamic_histogram& operator+=(const dynamic_histogram<OtherStorage, OtherAxes>& other)
  {
    static_assert(!mpl::empty<typename detail::intersection<Axes, OtherAxes>::type>::value,
                  "histograms lack common axes types");
    if (dim() != other.dim())
      throw std::logic_error("dimensions of histograms differ");
    if (size() != other.size())
      throw std::logic_error("sizes of histograms differ");
    if (!axes_equal_to(other.axes_))
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

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  void fill(Iterator values_begin, Iterator values_end)
  {
    BOOST_ASSERT_MSG(std::distance(values_begin, values_end) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    iter_args_impl(lin, values_begin, values_end);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename Sequence,
            typename = detail::is_sequence<Sequence>>
  void fill(const Sequence& values)
  {
    fill(std::begin(values), std::end(values));
  }

  template <typename... Args>
  void wfill(Args... args)
  {
    static_assert(std::is_same<Storage, dynamic_storage>::value,
                  "wfill only supported for dynamic_storage");
    BOOST_ASSERT_MSG((sizeof...(args) - 1) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    const double w = windex_impl(lin, args...);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  void wfill(Iterator values_begin, Iterator values_end, double w)
  {
    static_assert(std::is_same<Storage, dynamic_storage>::value,
                  "wfill only supported for dynamic_storage");
    BOOST_ASSERT_MSG(std::distance(values_begin, values_end) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    iter_args_impl(lin, values_begin, values_end);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename Sequence,
            typename = detail::is_sequence<Sequence>>
  void wfill(const Sequence& values, double w)
  {
    wfill(std::begin(values), std::end(values), w);
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

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  value_t value(Iterator indices_begin, Iterator indices_end) const
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

  template <typename Sequence,
            typename = detail::is_sequence<Sequence>>
  value_t value(const Sequence& indices) const
  {
    return value(std::begin(indices), std::end(indices));
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

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  variance_t variance(Iterator indices_begin, Iterator indices_end) const
  {
    BOOST_ASSERT_MSG(std::distance(indices_begin, indices_end) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    iter_args_impl(lin, indices_begin, indices_end);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.variance(lin.out);
  }

  template <typename Sequence,
            typename = detail::is_sequence<Sequence>>
  variance_t variance(const Sequence& indices) const
  {
    return variance(std::begin(indices), std::end(indices));
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

  template <typename OtherAxes>
  bool axes_equal_to(const OtherAxes& other_axes) const
  {
    detail::cmp_axis ca;
    for (unsigned i = 0; i < dim(); ++i)
      if (!apply_visitor(ca, axes_[i], other_axes[i]))
        return false;
    return true;
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
  friend class dynamic_histogram;

  template <typename Archiv, typename OtherStorage, typename OtherAxes>
  friend void serialize(Archiv&, dynamic_histogram<OtherStorage, OtherAxes>&, unsigned);
};


// when adding different histogram types, use a safe return type
template <typename Storage1,
          typename Storage2,
          typename Axes1,
          typename Axes2>
inline
dynamic_histogram<
  typename detail::select_storage<Storage1, Storage2>::type,
  typename detail::intersection<Axes1, Axes2>::type
>
operator+(const dynamic_histogram<Storage1, Axes1>& a,
          const dynamic_histogram<Storage2, Axes2>& b)
{
  dynamic_histogram<
    typename detail::select_storage<Storage1, Storage2>::type,
    typename detail::intersection<Axes1, Axes2>::type
  > tmp = a;
  tmp += b;
  return tmp;
}


template <typename... Axes>
inline
dynamic_histogram<>
make_dynamic_histogram(const Axes&... axes)
{
  return dynamic_histogram<>(axes...);
}


template <typename Storage, typename... Axes>
inline
dynamic_histogram<Storage>
make_dynamic_histogram_with(const Axes&... axes)
{
  return dynamic_histogram<Storage>(axes...);
}


}
}

#endif
