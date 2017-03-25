// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DYNAMIC_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_DYNAMIC_HISTOGRAM_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/variance.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/variant.hpp>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {

template <typename Axes = default_axes, typename Storage = adaptive_storage<>>
class dynamic_histogram {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");

public:
  using histogram_tag = detail::histogram_tag;
  using axis_type = typename make_variant_over<Axes>::type;
  using value_type = typename Storage::value_type;

private:
  using axes_type = std::vector<axis_type>;

public:
  dynamic_histogram() = default;

  template <typename... Axes1>
  explicit dynamic_histogram(const Axes1 &... axes)
      : axes_({axis_type(axes)...}) {
    storage_ = Storage(field_count());
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  dynamic_histogram(Iterator axes_begin, Iterator axes_end)
      : axes_(std::distance(axes_begin, axes_end)) {
    std::copy(axes_begin, axes_end, axes_.begin());
    storage_ = Storage(field_count());
  }

  template <typename OtherAxes, typename OtherStorage>
  explicit dynamic_histogram(
      const dynamic_histogram<OtherAxes, OtherStorage> &other)
      : axes_(other.axes_.begin(), other.axes_.end()),
        storage_(other.storage_) {}

  template <typename OtherAxes, typename OtherStorage>
  explicit dynamic_histogram(dynamic_histogram<OtherAxes, OtherStorage> &&other)
      : axes_(std::move(other.axes_)), storage_(std::move(other.storage_)) {}

  template <typename OtherAxes, typename OtherStorage>
  dynamic_histogram &
  operator=(const dynamic_histogram<OtherAxes, OtherStorage> &other) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&other)) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  template <typename OtherAxes, typename OtherStorage>
  dynamic_histogram &
  operator=(dynamic_histogram<OtherAxes, OtherStorage> &&other) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&other)) {
      axes_ = std::move(other.axes_);
      storage_ = std::move(other.storage_);
    }
    return *this;
  }

  template <typename OtherAxes, typename OtherStorage>
  bool
  operator==(const dynamic_histogram<OtherAxes, OtherStorage> &other) const {
    if (mpl::empty<
            typename detail::intersection<Axes, OtherAxes>::type>::value) {
      return false;
    }
    if (dim() != other.dim()) {
      return false;
    }
    if (!axes_equal_to(other.axes_)) {
      return false;
    }
    if (!(storage_ == other.storage_)) {
      return false;
    }
    return true;
  }

  template <template <class, class> class Histogram, typename OtherAxes,
            typename OtherStorage>
  dynamic_histogram &
  operator+=(const Histogram<OtherAxes, OtherStorage> &other) {
    static_assert(
        !mpl::empty<
            typename detail::intersection<Axes, OtherAxes>::type>::value,
        "histograms lack common axes types");
    if (dim() != other.dim()) {
      throw std::logic_error("dimensions of histograms differ");
    }
    if (size() != other.size()) {
      throw std::logic_error("sizes of histograms differ");
    }
    if (!axes_equal_to(other.axes_)) {
      throw std::logic_error("axes of histograms differ");
    }
    storage_ += other.storage_;
    return *this;
  }

  template <typename... Values> void fill(Values... values) {
    BOOST_ASSERT_MSG(sizeof...(values) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride) {
      storage_.increase(lin.out);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end) {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    iter_args_impl(lin, begin, end);
    if (lin.stride) {
      storage_.increase(lin.out);
    }
  }

  template <typename Sequence, typename = detail::is_sequence<Sequence>>
  void fill(const Sequence &values) {
    fill(std::begin(values), std::end(values));
  }

  template <typename... Values> void wfill(value_type w, Values... values) {
    static_assert(detail::has_weight_support<Storage>::value,
                  "wfill only supported for adaptive_storage");
    BOOST_ASSERT_MSG(sizeof...(values) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize_x lin;
    index_impl(lin, values...);
    if (lin.stride) {
      storage_.increase(lin.out, w);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void wfill(value_type w, Iterator begin, Iterator end) {
    static_assert(detail::has_weight_support<Storage>::value,
                  "wfill only supported for adaptive_storage");
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "iterator range does not match histogram dimension");
    detail::linearize_x lin;
    iter_args_impl(lin, begin, end);
    if (lin.stride) {
      storage_.increase(lin.out, w);
    }
  }

  template <typename Sequence, typename = detail::is_sequence<Sequence>>
  void wfill(value_type w, const Sequence &values) {
    wfill(w, std::begin(values), std::end(values));
  }

  template <typename... Indices> value_type value(Indices... indices) const {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, indices...);
    if (lin.stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(lin.out);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type value(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "iterator range does not match histogram dimension");
    detail::linearize lin;
    iter_args_impl(lin, begin, end);
    if (lin.stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(lin.out);
  }

  template <typename Sequence, typename = detail::is_sequence<Sequence>>
  value_type value(const Sequence &indices) const {
    return value(std::begin(indices), std::end(indices));
  }

  template <typename... Indices> value_type variance(Indices... indices) const {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    detail::linearize lin;
    index_impl(lin, indices...);
    if (lin.stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return detail::variance(storage_, lin.out);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type variance(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "iterator range does not match histogram dimension");
    detail::linearize lin;
    iter_args_impl(lin, begin, end);
    if (lin.stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return detail::variance(storage_, lin.out);
  }

  template <typename Sequence, typename = detail::is_sequence<Sequence>>
  value_type variance(const Sequence &indices) const {
    return variance(std::begin(indices), std::end(indices));
  }

  /// Number of axes (dimensions) of histogram
  unsigned dim() const { return axes_.size(); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const { return storage_.size(); }

  /// Sum of all counts in the histogram
  double sum() const {
    double result = 0.0;
    for (std::size_t i = 0, n = size(); i < n; ++i) {
      result += storage_.value(i);
    }
    return result;
  }

  /// Return axis \a i
  const axis_type &axis(unsigned i = 0) const {
    BOOST_ASSERT_MSG(i < dim(), "axis index out of range");
    return axes_[i];
  }

  /// Return axis \a i (added for conformity with static_histogram interface)
  template <unsigned N = 0u> const axis_type &axis() const {
    BOOST_ASSERT_MSG(N < dim(), "axis index out of range");
    return axes_[N];
  }

private:
  axes_type axes_;
  Storage storage_;

  std::size_t field_count() const {
    std::size_t fc = 1;
    for (const auto &a : axes_) {
      fc *= apply_visitor(detail::shape(), a);
    }
    return fc;
  }

  template <typename OtherAxes>
  bool axes_equal_to(const OtherAxes &other_axes) const {
    detail::cmp_axis ca;
    for (unsigned i = 0; i < dim(); ++i) {
      if (!apply_visitor(ca, axes_[i], other_axes[i])) {
        return false;
      }
    }
    return true;
  }

  template <typename Linearize, typename First, typename... Rest>
  void index_impl(Linearize &lin, First first, Rest... rest) const {
    lin.set(first);
    apply_visitor(lin, axes_[dim() - sizeof...(Rest)-1]);
    index_impl(lin, rest...);
  }

  template <typename Linearize> void index_impl(Linearize & /*unused*/) const {}

  template <typename Linearize, typename Iterator>
  void iter_args_impl(Linearize &lin, Iterator begin, Iterator end) const {
    auto axes_iter = axes_.begin();
    while (begin != end) {
      lin.set(*begin);
      apply_visitor(lin, *axes_iter);
      ++begin;
      ++axes_iter;
    }
  }

  friend struct storage_access;

  template <class OtherStorage, class OtherAxes, class Visitor>
  friend void for_each_axis(const dynamic_histogram<OtherStorage, OtherAxes> &,
                            Visitor &);

  template <typename OtherAxes, typename OtherStorage>
  friend class dynamic_histogram;

  template <typename Archiv, typename OtherAxes, typename OtherStorage>
  friend void serialize(Archiv &, dynamic_histogram<OtherAxes, OtherStorage> &,
                        unsigned);
};

template <typename... Axes>
inline dynamic_histogram<> make_dynamic_histogram(Axes &&... axes) {
  return dynamic_histogram<>(std::forward<Axes>(axes)...);
}

template <typename Storage, typename... Axes>
inline dynamic_histogram<default_axes, Storage>
make_dynamic_histogram_with(Axes &&... axes) {
  return dynamic_histogram<default_axes, Storage>(std::forward<Axes>(axes)...);
}
} // namespace histogram
} // namespace boost

#endif
