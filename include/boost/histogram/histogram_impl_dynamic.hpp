// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_DYNAMIC_IMPL_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_DYNAMIC_IMPL_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
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

template <typename Axes, typename Storage>
class histogram<Dynamic, Axes, Storage> {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");
  using size_pair = std::pair<std::size_t, std::size_t>;

public:
  using axis_type = typename make_variant_over<Axes>::type;
  using value_type = typename Storage::value_type;

private:
  using axes_type = std::vector<axis_type>;

public:
  histogram() = default;

  template <typename... Axes1>
  explicit histogram(const Axes1 &... axes) : axes_({axis_type(axes)...}) {
    storage_ = Storage(field_count());
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  histogram(Iterator axes_begin, Iterator axes_end)
      : axes_(std::distance(axes_begin, axes_end)) {
    std::copy(axes_begin, axes_end, axes_.begin());
    storage_ = Storage(field_count());
  }

  template <typename A, typename S>
  explicit histogram(const histogram<Dynamic, A, S> &rhs)
      : axes_(rhs.axes_.begin(), rhs.axes_.end()), storage_(rhs.storage_) {}

  template <typename A, typename S>
  histogram &operator=(const histogram<Dynamic, A, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <typename A, typename S>
  explicit histogram(histogram<Dynamic, A, S> &&rhs)
      : axes_(std::move(rhs.axes_)), storage_(std::move(rhs.storage_)) {}

  template <typename A, typename S>
  histogram &operator=(histogram<Dynamic, A, S> &&rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      axes_ = std::move(rhs.axes_);
      storage_ = std::move(rhs.storage_);
    }
    return *this;
  }

  template <typename A, typename S>
  bool operator==(const histogram<Dynamic, A, S> &rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename A, typename S>
  bool operator!=(const histogram<Dynamic, A, S> &rhs) const noexcept {
    return !operator==(rhs);
  }

  template <type D, typename A, typename S>
  histogram &operator+=(const histogram<D, A, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_)) {
      throw std::logic_error("axes of histograms differ");
    }
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename... Values> void fill(Values... values) noexcept {
    BOOST_ASSERT_MSG(sizeof...(values) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p =
        apply_lin<detail::xlin, Values...>(size_pair(0, 1), values...);
    if (p.second) {
      storage_.increase(p.first);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end) noexcept {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin_iter<detail::xlin>(size_pair(0, 1), begin);
    if (p.second) {
      storage_.increase(p.first);
    }
  }

  template <typename... Values> void wfill(value_type w, Values... values) noexcept {
    BOOST_ASSERT_MSG(sizeof...(values) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p =
        apply_lin<detail::xlin, Values...>(size_pair(0, 1), values...);
    if (p.second) {
      storage_.increase(p.first, w);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void wfill(value_type w, Iterator begin, Iterator end) noexcept {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin_iter<detail::xlin>(size_pair(0, 1), begin);
    if (p.second) {
      storage_.increase(p.first, w);
    }
  }

  template <typename... Indices> value_type value(Indices... indices) const {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p =
        apply_lin<detail::lin, Indices...>(size_pair(0, 1), indices...);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(p.first);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type value(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin_iter<detail::lin>(size_pair(0, 1), begin);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(p.first);
  }

  template <typename... Indices> value_type variance(Indices... indices) const {
    static_assert(detail::has_variance<Storage>::value,
                  "Storage lacks variance support");
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p =
        apply_lin<detail::lin, Indices...>(size_pair(0, 1), indices...);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.variance(p.first);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type variance(Iterator begin, Iterator end) const {
    static_assert(detail::has_variance<Storage>::value,
                  "Storage lacks variance support");
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    const auto p = apply_lin_iter<detail::lin>(size_pair(0, 1), begin);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.variance(p.first);
  }

  /// Number of axes (dimensions) of histogram
  unsigned dim() const noexcept { return axes_.size(); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const noexcept { return storage_.size(); }

  /// Sum of all counts in the histogram
  double sum() const noexcept {
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

  /// Return axis \a i (for conformity with histogram<Static, ...> interface)
  template <unsigned N = 0> const axis_type &axis() const {
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

  template <template <class, class> class Lin, typename Value>
  struct lin_visitor : public static_visitor<size_pair> {
    mutable size_pair pa;
    const Value &val;
    lin_visitor(const size_pair &p, const Value &v) : pa(p), val(v) {}
    template <typename A> size_pair operator()(const A &a) const {
      Lin<A, Value>::apply(pa.first, pa.second, a, val);
      return pa;
    }
  };

  template <template <class, class> class Lin, typename First, typename... Rest>
  size_pair apply_lin(size_pair &&p, const First &first,
                      const Rest &... rest) const {
    p = apply_visitor(lin_visitor<Lin, First>(p, first),
                      axes_[dim() - 1 - sizeof...(Rest)]);
    return apply_lin<Lin, Rest...>(std::move(p), rest...);
  }

  template <template <class, class> class Lin>
  size_pair apply_lin(size_pair &&p) const {
    return p;
  }

  template <template <class, class> class Lin, typename Iterator>
  size_pair apply_lin_iter(size_pair &&p, Iterator iter) const {
    for (const auto &a : axes_) {
      p = apply_visitor(lin_visitor<Lin, decltype(*iter)>(p, *iter), a);
      ++iter;
    }
    return p;
  }

  friend struct storage_access;

  template <type D, typename A, typename S> friend class histogram;

  template <typename Archiv, typename A, typename S>
  friend void serialize(Archiv &, histogram<Dynamic, A, S> &, unsigned);
};

template <typename... Axes>
inline histogram<Dynamic,
  typename detail::combine<default_axes, mpl::vector<Axes...>>::type>
make_dynamic_histogram(Axes &&... axes) {

  return histogram<Dynamic,
      typename detail::combine<default_axes, mpl::vector<Axes...>>::type
    >(std::forward<Axes>(axes)...);
}

template <typename Storage, typename... Axes>
inline histogram<Dynamic,
  typename detail::combine<default_axes, mpl::vector<Axes...>>::type,
  Storage>
make_dynamic_histogram_with(Axes &&... axes) {
  return histogram<Dynamic,
      typename detail::combine<default_axes, mpl::vector<Axes...>>::type,
      Storage
    >(std::forward<Axes>(axes)...);
}

} // namespace histogram
} // namespace boost

#endif
