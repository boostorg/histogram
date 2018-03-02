// Copyright 2015-2017 Hans Dembinski
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
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/iterator.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/count_if.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/type_index.hpp>
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

template <typename Axes, typename Storage>
class histogram<dynamic_tag, Axes, Storage> {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");

public:
  using any_axis_type = axis::any<Axes>;
  using axes_type = std::vector<any_axis_type>;
  using bin_type = typename Storage::bin_type;
  using bin_iterator = bin_iterator_over<Storage>;

public:
  histogram() = default;
  histogram(const histogram &) = default;
  histogram(histogram &&) = default;
  histogram &operator=(const histogram &) = default;
  histogram &operator=(histogram &&) = default;

  template <typename... Axes1>
  explicit histogram(const Axes1 &... axes) : axes_({any_axis_type(axes)...}) {
    storage_ = Storage(bincount_from_axes());
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  histogram(Iterator begin, Iterator end) : axes_(std::distance(begin, end)) {
    std::copy(begin, end, axes_.begin());
    storage_ = Storage(bincount_from_axes());
  }

  template <typename T, typename A, typename S>
  explicit histogram(const histogram<T, A, S> &rhs) : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename T, typename A, typename S>
  histogram &operator=(const histogram<T, A, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <typename S>
  explicit histogram(histogram<dynamic_tag, Axes, S> &&rhs)
      : axes_(std::move(rhs.axes_)), storage_(std::move(rhs.storage_)) {}

  template <typename S>
  histogram &operator=(histogram<dynamic_tag, Axes, S> &&rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      axes_ = std::move(rhs.axes_);
      storage_ = std::move(rhs.storage_);
    }
    return *this;
  }

  template <typename T, typename A, typename S>
  bool operator==(const histogram<T, A, S> &rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename T, typename A, typename S>
  bool operator!=(const histogram<T, A, S> &rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename T, typename A, typename S>
  histogram &operator+=(const histogram<T, A, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::logic_error("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename T>
  histogram &operator*=(const T& rhs) {
    storage_ *= rhs;
    return *this;
  }

  template <typename T>
  histogram &operator/=(const T& rhs) {
    storage_ *= 1.0 / rhs;
    return *this;
  }

  template <typename... Args> void fill(const Args &... args) {
    using n_weight =
        typename mpl::count_if<mpl::vector<Args...>, detail::is_weight<mpl::_>>;
    using n_sample =
        typename mpl::count_if<mpl::vector<Args...>, detail::is_sample<mpl::_>>;
    static_assert(n_weight::value <= 1,
                  "more than one weight argument is not allowed");
    static_assert(n_sample::value <= 1,
                  "more than one sample argument is not allowed");
    if (dim() != sizeof...(args) - n_weight::value - n_sample::value)
      throw std::invalid_argument(
          "fill arguments does not match histogram dimension");
    fill_impl(mpl::bool_<n_weight::value>(), mpl::bool_<n_sample::value>(),
              args...);
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

  template <typename Iterator, typename T,
            typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end, const detail::weight_t<T> &w) {
    if (dim() != std::distance(begin, end))
      throw std::invalid_argument(
          "fill iterator range does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_iter(idx, stride, begin);
    if (stride) {
      storage_.add(idx, w);
    }
  }

  template <typename... Indices> bin_type bin(Indices &&... indices) const {
    if (dim() != sizeof...(indices))
      throw std::invalid_argument(
          "value arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, indices...);
    if (stride == 0)
      throw std::out_of_range("invalid index");
    return storage_[idx];
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  bin_type bin(Iterator begin, Iterator end) const {
    if (dim() != std::distance(begin, end))
      throw std::invalid_argument(
          "iterator range in bin(...) does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_iter(idx, stride, begin);
    if (stride == 0)
      throw std::out_of_range("invalid index");
    return storage_[idx];
  }

  /// Number of axes (dimensions) of histogram
  unsigned dim() const noexcept { return axes_.size(); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t bincount() const noexcept { return storage_.size(); }

  /// Sum of all counts in the histogram
  bin_type sum() const noexcept {
    bin_type result(0);
    // don't use bincount() here, so sum() still works in a moved-from object
    for (std::size_t i = 0, n = storage_.size(); i < n; ++i) {
      result += storage_[i];
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
  histogram reduce_to(mpl::int_<N>, Rest...) const {
    const auto b =
        detail::bool_mask<mpl::vector<mpl::int_<N>, Rest...>>(dim(), true);
    return reduce_impl(b);
  }

  /// Return a lower dimensional histogram
  template <typename... Rest> histogram reduce_to(int n, Rest... rest) const {
    std::vector<bool> b(dim(), false);
    for (const auto &i : {n, rest...})
      b[i] = true;
    return reduce_impl(b);
  }

  /// Return a lower dimensional histogram
  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  histogram reduce_to(Iterator begin, Iterator end) const {
    std::vector<bool> b(dim(), false);
    for (; begin != end; ++begin)
      b[*begin] = true;
    return reduce_impl(b);
  }

  bin_iterator begin() const noexcept {
    return bin_iterator(*this, storage_);
  }

  bin_iterator end() const noexcept { return bin_iterator(storage_); }

private:
  axes_type axes_;
  Storage storage_;

  std::size_t bincount_from_axes() const noexcept {
    detail::field_count_visitor v;
    for_each_axis(v);
    return v.value;
  }

  template <typename... Args>
  inline void fill_impl(mpl::false_, mpl::false_, const Args &... args) {
    std::size_t idx = 0, stride = 1;
    int dummy;
    xlin<0>(idx, stride, dummy, args...);
    if (stride) {
      storage_.increase(idx);
    }
  }

  template <typename... Args>
  inline void fill_impl(mpl::true_, mpl::false_, const Args &... args) {
    std::size_t idx = 0, stride = 1;
    typename mpl::deref<
        typename mpl::find_if<mpl::vector<Args...>, detail::is_weight<mpl::_>>::type
      >::type w;
    xlin<0>(idx, stride, w, args...);
    if (stride) {
      storage_.add(idx, w);
    }
  }

  template <typename... Args>
  inline void fill_impl(mpl::false_, mpl::true_, const Args &... args) {
    // not implemented
  }

  template <typename... Args>
  inline void fill_impl(mpl::true_, mpl::true_, const Args &... args) {
    // not implemented
  }

  struct lin_visitor : public static_visitor<void> {
    std::size_t &idx;
    std::size_t &stride;
    const int val;
    lin_visitor(std::size_t &i, std::size_t &s, const int v)
        : idx(i), stride(s), val(v) {}
    template <typename A> void operator()(const A &a) const {
      detail::lin(idx, stride, a, val);
    }
  };

  template <unsigned D>
  inline void lin(std::size_t &, std::size_t &) const noexcept {}

  template <unsigned D, typename First, typename... Rest>
  inline void lin(std::size_t &idx, std::size_t &stride, const First &x,
                  const Rest &... rest) const noexcept {
    apply_visitor(lin_visitor{idx, stride, x}, axes_[D]);
    return lin<D + 1>(idx, stride, rest...);
  }

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
      throw std::invalid_argument(
          detail::cat("fill argument not convertible to axis value type: ",
                      boost::typeindex::type_id<Axis>().pretty_name(), ", ",
                      boost::typeindex::type_id<Value>().pretty_name()));
    }
  };

  template <unsigned D, typename Weight>
  inline void xlin(std::size_t &, std::size_t &, Weight &) const {}

  template <unsigned D, typename Weight, typename First, typename... Rest>
  inline void xlin(std::size_t &idx, std::size_t &stride, Weight &w,
                   const First &first, const Rest &... rest) const {
    apply_visitor(xlin_visitor<First>{idx, stride, first}, axes_[D]);
    return xlin<D + 1>(idx, stride, w, rest...);
  }

  template <unsigned D, typename T, typename... Rest>
  inline void xlin(std::size_t &idx, std::size_t &stride, detail::weight_t<T> &w,
                   const detail::weight_t<T> &first, const Rest &... rest) const {
    w = first;
    return xlin<D>(idx, stride, w, rest...);
  }

  template <typename Iterator>
  inline void lin_iter(std::size_t &idx, std::size_t &stride,
                       Iterator iter) const {
    for (const auto &a : axes_) {
      apply_visitor(lin_visitor(idx, stride, *iter), a);
      ++iter;
    }
  }

  template <typename Iterator>
  void xlin_iter(std::size_t &idx, std::size_t &stride, Iterator iter) const {
    for (const auto &a : axes_) {
      apply_visitor(xlin_visitor<decltype(*iter)>{idx, stride, *iter}, a);
      ++iter;
    }
  }

  histogram reduce_impl(const std::vector<bool> &b) const {
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
    histogram h(axes.begin(), axes.end());
    detail::index_mapper m(n, b);
    do {
      h.storage_.add(m.second, storage_[m.first]);
    } while (m.next());
    return h;
  }

  template <typename T, typename A, typename S> friend class histogram;
  friend class ::boost::python::access;
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

template <typename... Axes>
histogram<dynamic_tag, detail::union_t<axis::builtins, mpl::vector<Axes...>>>
make_dynamic_histogram(Axes &&... axes) {
  return histogram<dynamic_tag,
                   detail::union_t<axis::builtins, mpl::vector<Axes...>>>(
      std::forward<Axes>(axes)...);
}

template <typename... Axes>
histogram<dynamic_tag, detail::union_t<axis::builtins, mpl::vector<Axes...>>,
array_storage<weight_counter<double>>>
make_dynamic_weighted_histogram(Axes &&... axes) {
  return histogram<dynamic_tag,
                   detail::union_t<axis::builtins, mpl::vector<Axes...>>,
                   array_storage<weight_counter<double>>
    >(std::forward<Axes>(axes)...);
}

template <typename Storage, typename... Axes>
histogram<dynamic_tag, detail::union_t<axis::builtins, mpl::vector<Axes...>>,
          Storage>
make_dynamic_histogram_with(Axes &&... axes) {
  return histogram<dynamic_tag,
                   detail::union_t<axis::builtins, mpl::vector<Axes...>>,
                   Storage>(std::forward<Axes>(axes)...);
}

template <typename Iterator, typename = detail::is_iterator<Iterator>>
histogram<dynamic_tag, detail::union_t<axis::builtins,
                                         typename Iterator::value_type::types>>
make_dynamic_histogram(Iterator begin, Iterator end) {
  return histogram<
      dynamic_tag,
      detail::union_t<axis::builtins, typename Iterator::value_type::types>>(
      begin, end);
}

template <typename Iterator, typename = detail::is_iterator<Iterator>>
histogram<dynamic_tag, detail::union_t<axis::builtins,
                                         typename Iterator::value_type::types>,
                                         array_storage<weight_counter<double>>>
make_dynamic_weighted_histogram(Iterator begin, Iterator end) {
  return histogram<
      dynamic_tag,
      detail::union_t<axis::builtins, typename Iterator::value_type::types>,
      array_storage<weight_counter<double>>>(
      begin, end);
}

template <typename Storage, typename Iterator,
          typename = detail::is_iterator<Iterator>>
histogram<
    dynamic_tag,
    detail::union_t<axis::builtins, typename Iterator::value_type::types>,
    Storage>
make_dynamic_histogram_with(Iterator begin, Iterator end) {
  return histogram<
      dynamic_tag,
      detail::union_t<axis::builtins, typename Iterator::value_type::types>,
      Storage>(begin, end);
}

} // namespace histogram
} // namespace boost

#endif
