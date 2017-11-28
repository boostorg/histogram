// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_DYNAMIC_IMPL_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_DYNAMIC_IMPL_HPP_

#include <algorithm>
#include <boost/config.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/axis/axis.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <boost/histogram/value_iterator.hpp>
#include <boost/mpl/count.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <vector>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
}
} // namespace boost

// forward declaration for python
namespace boost {
namespace python {
class access;
}
} // namespace boost

namespace boost {
namespace histogram {

template <typename Axes, typename Storage> class dynamic_histogram {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");

public:
  using any_axis_type = axis::any<Axes>;
  using value_type = typename Storage::value_type;
  using value_iterator = value_iterator<Storage>;

private:
  using axes_type = std::vector<any_axis_type>;

public:
  dynamic_histogram() = default;
  dynamic_histogram(const dynamic_histogram &) = default;
  dynamic_histogram(dynamic_histogram &&) = default;
  dynamic_histogram &operator=(const dynamic_histogram &) = default;
  dynamic_histogram &operator=(dynamic_histogram &&) = default;

  template <typename... Axes1>
  explicit dynamic_histogram(const Axes1 &... axes)
      : axes_({any_axis_type(axes)...}) {
    storage_ = Storage(bincount_from_axes());
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  dynamic_histogram(Iterator begin, Iterator end)
      : axes_(std::distance(begin, end)) {
    std::copy(begin, end, axes_.begin());
    storage_ = Storage(bincount_from_axes());
  }

  template <typename A, typename S>
  explicit dynamic_histogram(const static_histogram<A, S> &rhs)
      : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename A, typename S>
  explicit dynamic_histogram(const dynamic_histogram<A, S> &rhs)
      : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename A, typename S>
  dynamic_histogram &operator=(const static_histogram<A, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <typename A, typename S>
  dynamic_histogram &operator=(const dynamic_histogram<A, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <typename S>
  explicit dynamic_histogram(dynamic_histogram<Axes, S> &&rhs)
      : axes_(std::move(rhs.axes_)), storage_(std::move(rhs.storage_)) {}

  template <typename S>
  dynamic_histogram &operator=(dynamic_histogram<Axes, S> &&rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      axes_ = std::move(rhs.axes_);
      storage_ = std::move(rhs.storage_);
    }
    return *this;
  }

  template <typename A, typename S>
  bool operator==(const static_histogram<A, S> &rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename A, typename S>
  bool operator==(const dynamic_histogram<A, S> &rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename A, typename S>
  bool operator!=(const static_histogram<A, S> &rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename A, typename S>
  bool operator!=(const dynamic_histogram<A, S> &rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename A, typename S>
  dynamic_histogram &operator+=(const static_histogram<A, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::logic_error("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename A, typename S>
  dynamic_histogram &operator+=(const dynamic_histogram<A, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::logic_error("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  dynamic_histogram &operator*=(const value_type rhs) {
    storage_ *= rhs;
    return *this;
  }

  dynamic_histogram &operator/=(const value_type rhs) {
    storage_ *= 1.0 / rhs;
    return *this;
  }

  template <typename... Args> void fill(Args &&... args) {
    using n_count = typename mpl::count<mpl::vector<Args...>, count>;
    using n_weight = typename mpl::count<mpl::vector<Args...>, weight>;
    static_assert(
        (n_count::value + n_weight::value) <= 1,
        "arguments may contain at most one instance of type count or weight");
    if (dim() != sizeof...(args) - n_count::value - n_weight::value)
      throw std::invalid_argument(
          "fill arguments does not match histogram dimension");
    fill_impl(mpl::int_<(n_count::value + 2 * n_weight::value)>(),
              std::forward<Args>(args)...);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end) {
    if (dim() != std::distance(begin, end))
      throw std::invalid_argument(
          "fill iterator range does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_iter(idx, stride, begin);
    if (stride) {
      storage_.increase(idx);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end, const count n) {
    if (dim() != std::distance(begin, end))
      throw std::invalid_argument(
          "fill iterator range does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_iter(idx, stride, begin);
    if (stride) {
      storage_.add(idx, n.value);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end, const weight w) {
    if (dim() != std::distance(begin, end))
      throw std::invalid_argument(
          "fill iterator range does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_iter(idx, stride, begin);
    if (stride) {
      storage_.increase_by_weight(idx, w.value);
    }
  }

  template <typename... Indices> value_type value(Indices &&... indices) const {
    if (dim() != sizeof...(indices))
      throw std::invalid_argument(
          "value arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, indices...);
    if (stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(idx);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type value(Iterator begin, Iterator end) const {
    if (dim() != std::distance(begin, end))
      throw std::invalid_argument(
          "value iterator range does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_iter(idx, stride, begin);
    if (stride == 0)
      throw std::out_of_range("invalid index");
    return storage_.value(idx);
  }

  template <typename S = Storage, typename... Indices>
  detail::requires_variance_support<S> variance(Indices &&... indices) const {
    if (dim() != sizeof...(indices))
      throw std::invalid_argument(
          "variance arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, indices...);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.variance(idx);
  }

  template <typename S = Storage, typename Iterator,
            typename = detail::is_iterator<Iterator>>
  detail::requires_variance_support<S> variance(Iterator begin,
                                                Iterator end) const {
    if (dim() != std::distance(begin, end))
      throw std::invalid_argument(
          "variance iterator range does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_iter(idx, stride, begin);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.variance(idx);
  }

  /// Number of axes (dimensions) of histogram
  unsigned dim() const noexcept { return axes_.size(); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t bincount() const noexcept { return storage_.size(); }

  /// Sum of all counts in the histogram
  double sum() const noexcept {
    double result = 0.0;
    // don't use bincount() here, so sum() still works in a moved-from object
    for (std::size_t i = 0, n = storage_.size(); i < n; ++i) {
      result += storage_.value(i);
    }
    return result;
  }

  /// Reset bin counters to zero
  void reset() { storage_ = Storage(bincount_from_axes()); }

  /// Return axis \a i
  any_axis_type &axis(unsigned i = 0) {
    if (i >= dim())
      throw std::out_of_range("axis index out of range");
    return axes_[i];
  }

  /// Return axis \a i (const version)
  const any_axis_type &axis(unsigned i = 0) const {
    if (i >= dim())
      throw std::out_of_range("axis index out of range");
    return axes_[i];
  }

  /// Apply unary functor/function to each axis
  template <typename Unary> void for_each_axis(Unary &&unary) const {
    for (const auto &a : axes_) {
      apply_visitor(detail::unary_visitor<Unary>(unary), a);
    }
  }

  /// Return a lower dimensional histogram
  template <int N, typename... Rest>
  dynamic_histogram reduce_to(mpl::int_<N>, Rest...) const {
    const auto b =
        detail::bool_mask<mpl::vector<mpl::int_<N>, Rest...>>(dim(), true);
    return reduce_impl(b);
  }

  /// Return a lower dimensional histogram
  template <typename... Rest>
  dynamic_histogram reduce_to(int n, Rest... rest) const {
    std::vector<bool> b(dim(), false);
    for (const auto &i : {n, rest...})
      b[i] = true;
    return reduce_impl(b);
  }

  /// Return a lower dimensional histogram
  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  dynamic_histogram reduce_to(Iterator begin, Iterator end) const {
    std::vector<bool> b(dim(), false);
    for (; begin != end; ++begin)
      b[*begin] = true;
    return reduce_impl(b);
  }

  value_iterator begin() const noexcept {
    return value_iterator(*this, storage_);
  }

  value_iterator end() const noexcept {
    return value_iterator(storage_);
  }

private:
  axes_type axes_;
  Storage storage_;

  std::size_t bincount_from_axes() const noexcept {
    detail::field_count_visitor v;
    for_each_axis(v);
    return v.value;
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<0>, Args &&... args) {
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, std::forward<Args>(args)...);
    if (stride) {
      storage_.increase(idx);
    }
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<1>, Args &&... args) {
    std::size_t idx = 0, stride = 1;
    unsigned n = 0;
    xlin_n<0>(idx, stride, n, std::forward<Args>(args)...);
    if (stride) {
      storage_.add(idx, n);
    }
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<2>, Args &&... args) {
    std::size_t idx = 0, stride = 1;
    double w = 0.0;
    xlin_w<0>(idx, stride, w, std::forward<Args>(args)...);
    if (stride) {
      storage_.increase_by_weight(idx, w);
    }
  }

  template <typename Value> struct lin_visitor : public static_visitor<void> {
    std::size_t &idx;
    std::size_t &stride;
    const Value &val;
    lin_visitor(std::size_t &i, std::size_t &s, const Value &v)
        : idx(i), stride(s), val(v) {}
    template <typename A> void operator()(const A &a) const {
      detail::lin(idx, stride, a, val);
    }
  };

  template <typename Value> struct xlin_visitor : public static_visitor<void> {
    std::size_t &idx;
    std::size_t &stride;
    const Value &val;
    xlin_visitor(std::size_t &i, std::size_t &s, const Value &v)
        : idx(i), stride(s), val(v) {}
    template <typename Axis> void operator()(const Axis &a) const {
      impl(std::is_convertible<Value, typename Axis::value_type>(), a);
    }

    template <typename Axis> void impl(std::true_type, const Axis &a) const {
      detail::xlin(idx, stride, a, val);
    }

    template <typename Axis> void impl(std::false_type, const Axis &) const {
      throw std::runtime_error(
          "fill argument not convertible to axis value type");
    }
  };

  template <unsigned D> inline void lin(std::size_t &, std::size_t &) const {}

  template <unsigned D, typename First, typename... Rest>
  inline void lin(std::size_t &idx, std::size_t &stride, First &&x,
                  Rest &&... rest) const {
    apply_visitor(lin_visitor<First>{idx, stride, x}, axes_[D]);
    return lin<D + 1>(idx, stride, std::forward<Rest>(rest)...);
  }

  template <unsigned D> inline void xlin(std::size_t &, std::size_t &) const {}

  template <unsigned D, typename First, typename... Rest>
  inline void xlin(std::size_t &idx, std::size_t &stride, First &&x,
                   Rest &&... rest) const {
    apply_visitor(xlin_visitor<First>{idx, stride, x}, axes_[D]);
    return xlin<D + 1>(idx, stride, std::forward<Rest>(rest)...);
  }

  template <unsigned D>
  inline void xlin_w(std::size_t &, std::size_t &, double &) const {}

  template <unsigned D, typename First, typename... Rest>
  inline typename enable_if<is_same<First, weight>>::type
  xlin_w(std::size_t &idx, std::size_t &stride, double &x, First &&first,
         Rest &&... rest) const {
    x = first.value;
    return xlin_w<D>(idx, stride, x, std::forward<Rest>(rest)...);
  }

  template <unsigned D, typename First, typename... Rest>
  inline typename disable_if<is_same<First, weight>>::type
  xlin_w(std::size_t &idx, std::size_t &stride, double &x, First &&first,
         Rest &&... rest) const {
    apply_visitor(xlin_visitor<First>{idx, stride, std::forward<First>(first)},
                  axes_[D]);
    return xlin_w<D + 1>(idx, stride, x, std::forward<Rest>(rest)...);
  }

  template <unsigned D>
  inline void xlin_n(std::size_t &, std::size_t &, unsigned &) const {}

  template <unsigned D, typename First, typename... Rest>
  inline typename enable_if<is_same<First, count>>::type
  xlin_n(std::size_t &idx, std::size_t &stride, unsigned &x, First &&first,
         Rest &&... rest) const {
    x = first.value;
    return xlin_n<D>(idx, stride, x, std::forward<Rest>(rest)...);
  }

  template <unsigned D, typename First, typename... Rest>
  inline typename disable_if<is_same<First, count>>::type
  xlin_n(std::size_t &idx, std::size_t &stride, unsigned &x, First &&first,
         Rest &&... rest) const {
    apply_visitor(xlin_visitor<First>{idx, stride, std::forward<First>(first)},
                  axes_[D]);
    return xlin_n<D + 1>(idx, stride, x, std::forward<Rest>(rest)...);
  }

  template <typename Iterator>
  void lin_iter(std::size_t &idx, std::size_t &stride, Iterator iter) const {
    for (const auto &a : axes_) {
      apply_visitor(lin_visitor<decltype(*iter)>(idx, stride, *iter), a);
      ++iter;
    }
  }

  template <typename Iterator>
  void xlin_iter(std::size_t &idx, std::size_t &stride, Iterator iter) const {
    for (const auto &a : axes_) {
      apply_visitor(xlin_visitor<decltype(*iter)>(idx, stride, *iter), a);
      ++iter;
    }
  }

  dynamic_histogram reduce_impl(const std::vector<bool> &b) const {
    axes_type axes;
    std::vector<unsigned> n(b.size());
    auto axes_iter = axes_.begin();
    auto n_iter = n.begin();
    for (const auto &bi : b) {
      if (bi)
        axes.emplace_back(*axes_iter);
      *n_iter = axes_iter->shape();
      ++axes_iter;
      ++n_iter;
    }
    dynamic_histogram h(axes.begin(), axes.end());
    detail::index_mapper m(n, b);
    do {
      detail::storage_add(h.storage_, storage_, m.second, m.first);
    } while (m.next());
    return h;
  }

  template <typename A, typename S> friend class dynamic_histogram;
  template <typename A, typename S> friend class static_histogram;
  friend class ::boost::python::access;
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

template <typename... Axes>
dynamic_histogram<detail::combine_t<axis::builtins, mpl::vector<Axes...>>>
make_dynamic_histogram(Axes &&... axes) {

  return dynamic_histogram<
      detail::combine_t<axis::builtins, mpl::vector<Axes...>>>(
      std::forward<Axes>(axes)...);
}

template <typename Storage, typename... Axes>
dynamic_histogram<detail::combine_t<axis::builtins, mpl::vector<Axes...>>,
                  Storage>
make_dynamic_histogram_with(Axes &&... axes) {
  return dynamic_histogram<
      detail::combine_t<axis::builtins, mpl::vector<Axes...>>, Storage>(
      std::forward<Axes>(axes)...);
}

template <typename Iterator, typename = detail::is_iterator<Iterator>>
dynamic_histogram<
    detail::combine_t<axis::builtins, typename Iterator::value_type::types>>
make_dynamic_histogram(Iterator begin, Iterator end) {
  return dynamic_histogram<
      detail::combine_t<axis::builtins, typename Iterator::value_type::types>>(
      begin, end);
}

template <typename Storage, typename Iterator,
          typename = detail::is_iterator<Iterator>>
dynamic_histogram<
    detail::combine_t<axis::builtins, typename Iterator::value_type::types>,
    Storage>
make_dynamic_histogram_with(Iterator begin, Iterator end) {
  return dynamic_histogram<
      detail::combine_t<axis::builtins, typename Iterator::value_type::types>,
      Storage>(begin, end);
}

} // namespace histogram
} // namespace boost

#endif
