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
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <type_traits>

namespace boost {
namespace histogram {

template <typename Axes, typename Storage=adaptive_storage<>>
class static_histogram
{
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");

public:
  using axes_type = typename fusion::result_of::as_vector<Axes>::type;
  using value_type = typename Storage::value_type;

  static_histogram() = default;

  template <typename... Axes1>
  explicit
  static_histogram(const Axes1&... axes) :
    axes_(axes...)
  {
    storage_ = Storage(field_count());
  }

  static_histogram(const static_histogram& other) = default;
  static_histogram(static_histogram&& other) = default;
  static_histogram& operator=(const static_histogram& other) = default;
  static_histogram& operator=(static_histogram&& other) = default;

  template <typename OtherStorage>
  static_histogram(const static_histogram<Axes, OtherStorage>& other) :
    axes_(other.axes_), storage_(other.storage_)
  {}

  template <typename OtherStorage>
  static_histogram(static_histogram<Axes, OtherStorage>&& other) :
    axes_(std::move(other.axes_)),
    storage_(std::move(other.storage_))
  {}

  template <typename OtherStorage>
  static_histogram& operator=(const static_histogram<Axes, OtherStorage>& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  template <typename OtherStorage>
  static_histogram& operator=(static_histogram<Axes, OtherStorage>&& other)
  {
    if (static_cast<const void*>(this) != static_cast<const void*>(&other)) {
      axes_ = std::move(other.axes_);
      storage_ = std::move(other.storage_);
    }
    return *this;
  }

  template <typename OtherAxes, typename OtherStorage>
  bool operator==(const static_histogram<OtherAxes, OtherStorage>& other) const
  {
    if (!axes_equal_to(other.axes_))
      return false;
    return storage_ == other.storage_;
  }

  template <typename OtherStorage>
  static_histogram& operator+=(const static_histogram<Axes, OtherStorage>& other)
  {
    if (!axes_equal_to(other.axes_))
      throw std::logic_error("axes of histograms differ");
    storage_ += other.storage_;
    return *this;
  }

  template <typename... Values>
  void fill(Values... values)
  {
    static_assert(sizeof...(values) == dim(),
                  "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end)
  {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "iterator range does not match histogram dimension");
    detail::linearize_x lin;
    iter_args_impl<detail::linearize_x, Iterator>::apply(lin, axes_, begin);
    if (lin.stride)
      storage_.increase(lin.out);
  }

  template <typename Sequence,
            typename = detail::is_sequence<Sequence>>
  void fill(const Sequence& values)
  {
    fill(std::begin(values), std::end(values));
  }

  template <typename... Values>
  void wfill(double w, Values... values)
  {
    static_assert(detail::has_weight_support<Storage>::value,
                  "wfill only supported for adaptive_storage");
    static_assert(sizeof...(values) == dim(),
                  "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  void wfill(double w, Iterator begin, Iterator end)
  {
    static_assert(detail::has_weight_support<Storage>::value,
                  "wfill only supported for adaptive_storage");
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "iterator range does not match histogram dimension");
    detail::linearize_x lin;
    iter_args_impl<detail::linearize_x, Iterator>::apply(lin, axes_, begin);
    if (lin.stride)
      storage_.increase(lin.out, w);
  }

  template <typename Sequence,
            typename = detail::is_sequence<Sequence>>
  void wfill(double w, const Sequence& values)
  {
    wfill(w, std::begin(values), std::end(values));
  }

  template <typename... Args>
  value_type value(Args... args) const
  {
    static_assert(sizeof...(args) == dim(),
                  "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, args...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(lin.out);
  }

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  value_type value(Iterator begin, Iterator end) const
  {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "iterator range does not match histogram dimension");
    detail::linearize lin;
    iter_args_impl<detail::linearize, Iterator>::apply(lin, axes_, begin);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(lin.out);
  }

  template <typename Sequence,
            typename = detail::is_sequence<Sequence>>
  value_type value(const Sequence& indices) const
  {
    return value(std::begin(indices), std::end(indices));
  }

  template <typename... Indices>
  value_type variance(Indices... indices) const
  {
    static_assert(sizeof...(indices) == dim(),
                  "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, indices...);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.variance(lin.out);
  }

  template <typename Iterator,
            typename = detail::is_iterator<Iterator>>
  value_type variance(Iterator begin, Iterator end) const
  {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "iterator range does not match histogram dimension");
    detail::linearize lin;
    iter_args_impl<detail::linearize, Iterator>::apply(lin, axes_, begin);
    if (lin.stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.variance(lin.out);
  }

  template <typename Sequence,
            typename = detail::is_sequence<Sequence>>
  value_type variance(const Sequence& indices) const
  {
    return variance(std::begin(indices), std::end(indices));
  }

  /// Number of axes (dimensions) of histogram
  static constexpr unsigned dim()
  { return fusion::result_of::size<Axes>::type::value; }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const { return storage_.size(); }

  /// Sum of all counts in the histogram
  double sum() const
  {
    double result = 0.0;
    for (std::size_t i = 0, n = size(); i < n; ++i)
      result += storage_.value(i);
    return result;
  }

  template <unsigned N = 0>
  typename std::add_const<
    typename fusion::result_of::value_at_c<axes_type, N>::type
  >::type&
  axis() const
  {
    static_assert(N < fusion::result_of::size<axes_type>::value,
                  "axis index out of range");
    return fusion::at_c<N>(axes_);
  }

private:
  axes_type axes_;
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
  bool axes_equal_to(const OtherAxes&) const {
    return false;
  }

  bool axes_equal_to(const axes_type& other_axes) const {
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

  template <typename Linearize, typename Iterator, unsigned N=0>
  struct iter_args_impl {
    static void apply(Linearize& lin, const axes_type& axes, Iterator iter) {
      lin.set(*iter);
      lin(fusion::at_c<N>(axes));
      iter_args_impl<Linearize, Iterator, N+1>::apply(lin, axes, ++iter);
    }
  };

  template <typename Linearize, typename Iterator>
  struct iter_args_impl<Linearize, Iterator, dim()> {
    static void apply(Linearize&, const axes_type&, Iterator) {}
  };

  template <typename OtherAxes, typename OtherStorage>
  friend class static_histogram;

  template <class Archive, class OtherStorage, class OtherAxes>
  friend void serialize(Archive&, static_histogram<OtherStorage, OtherAxes>&, unsigned);
};


/// default static type factory
template <typename... Axes>
inline
static_histogram<mpl::vector<Axes...>>
make_static_histogram(const Axes&... axes)
{
  return static_histogram<mpl::vector<Axes...>>(axes...);
}


/// static type factory with variable storage type
template <typename Storage, typename... Axes>
inline
static_histogram<mpl::vector<Axes...>, Storage>
make_static_histogram_with(const Axes&... axes)
{
  return static_histogram<mpl::vector<Axes...>, Storage>(axes...);
}

}
}

#endif
