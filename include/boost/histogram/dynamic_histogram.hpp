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
  using pairs = std::pair<std::size_t, std::size_t>;
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
    const auto p = apply_lin<detail::xlin, Values...>(pairs(0, 1), values...);
    if (p.second) {
      storage_.increase(p.first);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end) {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin_iter<detail::xlin>(pairs(0, 1), begin);
    if (p.second) {
      storage_.increase(p.first);
    }
  }

  template <
      bool has_weight_support = detail::has_weight_support<Storage>::value,
      typename... Values>
  typename std::enable_if<has_weight_support>::type wfill(value_type w,
                                                          Values... values) {
    BOOST_ASSERT_MSG(sizeof...(values) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin<detail::xlin, Values...>(pairs(0, 1), values...);
    if (p.second) {
      storage_.increase(p.first, w);
    }
  }

  template <
      bool has_weight_support = detail::has_weight_support<Storage>::value,
      typename Iterator, typename = detail::is_iterator<Iterator>>
  typename std::enable_if<has_weight_support>::type
  wfill(value_type w, Iterator begin, Iterator end) {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin_iter<detail::xlin>(pairs(0, 1), begin);
    if (p.second) {
      storage_.increase(p.first, w);
    }
  }

  template <typename... Indices> value_type value(Indices... indices) const {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin<detail::lin, Indices...>(pairs(0, 1), indices...);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(p.first);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type value(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin_iter<detail::lin>(pairs(0, 1), begin);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(p.first);
  }

  template <typename... Indices> value_type variance(Indices... indices) const {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin<detail::lin, Indices...>(pairs(0, 1), indices...);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return detail::variance(storage_, p.first);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type variance(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin_iter<detail::lin>(pairs(0, 1), begin);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return detail::variance(storage_, p.first);
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

  /// Apply unary functor/function to each axis
  template <typename Unary> void for_each_axis(Unary &unary) const {
    for (const auto &a : axes_) {
      apply_visitor(detail::unary_visitor<Unary>(unary), a);
    }
  }

private:
  axes_type axes_;
  Storage storage_;

  std::size_t field_count() const {
    detail::field_count fc;
    for (const auto &a : axes_) {
      apply_visitor(fc, a);
    }
    return fc.value;
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

  template <template <class, class> class Lin, typename Value>
  struct lin_visitor : public static_visitor<pairs> {
    mutable pairs pa;
    const Value &val;
    lin_visitor(const pairs& p, const Value &v)
        : pa(p), val(v) {}
    template <typename A> pairs operator()(const A &a) const {
      Lin<A, Value>::apply(pa.first, pa.second, a, val);
      return pa;
    }
  };

  template <template <class, class> class Lin, typename First, typename... Rest>
  pairs apply_lin(pairs&& p, const First &first,
                 const Rest &... rest) const {
    p = apply_visitor(lin_visitor<Lin, First>(p, first),
                      axes_[dim() - 1 - sizeof...(Rest)]);
    return apply_lin<Lin, Rest...>(std::move(p), rest...);
  }

  template <template <class, class> class Lin>
  pairs apply_lin(pairs&& p) const { return p; }

  template <template <class, class> class Lin, typename Iterator>
  pairs apply_lin_iter(pairs&& p, Iterator iter) const {
    for (const auto &a : axes_) {
      p = apply_visitor(lin_visitor<Lin, decltype(*iter)>(p, *iter), a);
      ++iter;
    }
    return p;
  }

  friend struct storage_access;

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
