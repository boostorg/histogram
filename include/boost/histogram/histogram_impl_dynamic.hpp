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
#include <boost/mpl/count.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/variant.hpp>
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
class histogram<Dynamic, Axes, Storage> {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");

public:
  using axis_type = typename make_variant_over<Axes>::type;
  using value_type = typename Storage::value_type;

private:
  using axes_type = std::vector<axis_type>;

public:
  histogram() = default;
  histogram(const histogram &) = default;
  histogram(histogram &&) = default;
  histogram &operator=(const histogram &) = default;
  histogram &operator=(histogram &&) = default;

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

  template <typename D, typename A, typename S>
  explicit histogram(const histogram<D, A, S> &rhs) : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename D, typename A, typename S>
  histogram &operator=(const histogram<D, A, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <typename S>
  explicit histogram(histogram<Dynamic, Axes, S> &&rhs)
      : axes_(std::move(rhs.axes_)), storage_(std::move(rhs.storage_)) {}

  template <typename S>
  histogram &operator=(histogram<Dynamic, Axes, S> &&rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      axes_ = std::move(rhs.axes_);
      storage_ = std::move(rhs.storage_);
    }
    return *this;
  }

  template <typename D, typename A, typename S>
  bool operator==(const histogram<D, A, S> &rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename D, typename A, typename S>
  bool operator!=(const histogram<D, A, S> &rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename D, typename A, typename S>
  histogram &operator+=(const histogram<D, A, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_)) {
      throw std::logic_error("axes of histograms differ");
    }
    for (std::size_t i = 0, n = storage_.size(); i < n; ++i)
      storage_.add(i, rhs.storage_.value(i), rhs.storage_.variance(i));
    return *this;
  }

  template <typename... Args> void fill(Args... args) noexcept {
    using n_count = typename mpl::count<mpl::vector<Args...>, count>;
    using n_weight = typename mpl::count<mpl::vector<Args...>, weight>;
    static_assert(
        (n_count::value + n_weight::value) <= 1,
        "arguments may contain at most one instance of type count or weight");
    BOOST_ASSERT_MSG(sizeof...(args) ==
                         (dim() + n_count::value + n_weight::value),
                     "number of arguments does not match histogram dimension");
    fill_impl(mpl::int_<(n_count::value + 2 * n_weight::value)>(), args...);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end) noexcept {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin_iter<detail::xlin>(idx, stride, begin);
    if (stride) {
      storage_.increase(idx);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end, const count &n) noexcept {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin_iter<detail::xlin>(idx, stride, begin);
    if (stride) {
      storage_.increase(idx, n);
    }
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  void fill(Iterator begin, Iterator end, const weight &w) noexcept {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin_iter<detail::xlin>(idx, stride, begin);
    if (stride) {
      storage_.weighted_increase(idx, static_cast<double>(w));
    }
  }

  template <typename... Indices> value_type value(Indices... indices) const {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin<detail::lin, 0, Indices...>(idx, stride, indices...);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(idx);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type value(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin_iter<detail::lin>(idx, stride, begin);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(idx);
  }

  template <typename... Indices> value_type variance(Indices... indices) const {
    BOOST_ASSERT_MSG(sizeof...(indices) == dim(),
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin<detail::lin, 0, Indices...>(idx, stride, indices...);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.variance(idx);
  }

  template <typename Iterator, typename = detail::is_iterator<Iterator>>
  value_type variance(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin_iter<detail::lin>(idx, stride, begin);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.variance(idx);
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

  /// Reset bin counters to zero
  void reset() { storage_ = std::move(Storage(storage_.size())); }

  /// Return axis \a i
  const axis_type &axis(unsigned i = 0) const {
    BOOST_ASSERT_MSG(i < dim(), "axis index out of range");
    return axes_[i];
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
    for_each_axis(fc);
    return fc.value;
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<0>, const Args &... args) {
    std::size_t idx = 0, stride = 1;
    apply_lin<detail::xlin, 0, Args...>(idx, stride, args...);
    if (stride) {
      storage_.increase(idx);
    }
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<1>, const Args &... args) {
    std::size_t idx = 0, stride = 1;
    unsigned n = 0;
    apply_lin_x<detail::xlin, 0, unsigned, Args...>(idx, stride, n, args...);
    if (stride) {
      storage_.increase(idx, n);
    }
  }

  template <typename... Args>
  inline void fill_impl(mpl::int_<2>, const Args &... args) {
    std::size_t idx = 0, stride = 1;
    double w = 0.0;
    apply_lin_x<detail::xlin, 0, double, Args...>(idx, stride, w, args...);
    if (stride) {
      storage_.weighted_increase(idx, w);
    }
  }

  template <template <class, class> class Lin, typename Value>
  struct lin_visitor : public static_visitor<void> {
    std::size_t &idx;
    std::size_t &stride;
    const Value &val;
    lin_visitor(std::size_t &i, std::size_t &s, const Value &v)
        : idx(i), stride(s), val(v) {}
    template <typename A> void operator()(const A &a) const {
      Lin<A, Value>::apply(idx, stride, a, val);
    }
  };

  template <template <class, class> class Lin, unsigned D>
  inline void apply_lin(std::size_t &, std::size_t &) const {}

  template <template <class, class> class Lin, unsigned D, typename First,
            typename... Rest>
  inline void apply_lin(std::size_t &idx, std::size_t &stride, const First &x,
                        const Rest &... rest) const {
    apply_visitor(lin_visitor<Lin, First>(idx, stride, x), axes_[D]);
    return apply_lin<Lin, D + 1, Rest...>(idx, stride, rest...);
  }

  template <template <class, class> class Lin, unsigned D, typename X>
  inline void apply_lin_x(std::size_t &, std::size_t &, X &) const {}

  template <template <class, class> class Lin, unsigned D, typename X,
            typename First, typename... Rest>
  inline typename std::enable_if<!(std::is_same<First, weight>::value ||
                                   std::is_same<First, count>::value)>::type
  apply_lin_x(std::size_t &idx, std::size_t &stride, X &x, const First &first,
              const Rest &... rest) const {
    apply_visitor(lin_visitor<Lin, First>(idx, stride, first), axes_[D]);
    return apply_lin_x<Lin, D + 1, X, Rest...>(idx, stride, x, rest...);
  }

  template <template <class, class> class Lin, unsigned D, typename X, typename,
            typename... Rest>
  inline void apply_lin_x(std::size_t &idx, std::size_t &stride, X &x,
                          const weight &first, const Rest &... rest) const {
    x = static_cast<X>(first);
    return apply_lin_x<Lin, D, X, Rest...>(idx, stride, x, rest...);
  }

  template <template <class, class> class Lin, unsigned D, typename X, typename,
            typename... Rest>
  inline void apply_lin_x(std::size_t &idx, std::size_t &stride, X &x,
                          const count &first, const Rest &... rest) const {
    x = static_cast<X>(first);
    return apply_lin_x<Lin, D, X, Rest...>(idx, stride, x, rest...);
  }

  template <template <class, class> class Lin, typename Iterator>
  void apply_lin_iter(std::size_t &idx, std::size_t &stride,
                      Iterator iter) const {
    for (const auto &a : axes_) {
      apply_visitor(lin_visitor<Lin, decltype(*iter)>(idx, stride, *iter), a);
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
      *n_iter = apply_visitor(detail::shape(), *axes_iter);
      ++axes_iter;
      ++n_iter;
    }
    histogram h(axes.begin(), axes.end());
    detail::index_mapper m(n, b);
    do {
      h.storage_.add(m.second, storage_.value(m.first),
                     storage_.variance(m.first));
    } while (m.next());
    return h;
  }

  template <typename Ns>
  friend histogram reduce(const histogram &h, const detail::keep_static<Ns> &) {
    const auto b = detail::bool_mask<Ns>(h.dim(), true);
    return h.reduce_impl(b);
  }

  friend histogram reduce(const histogram &h, const detail::keep_dynamic &k) {
    std::vector<bool> b(h.dim(), false);
    for (const auto &i : k)
      b[i] = true;
    return h.reduce_impl(b);
  }

  // friend histogram reduce(const histogram &h, const remove &r) {
  //   std::vector<bool> b(h.dim(), true);
  //   for (const auto &i : r)
  //     b[i] = false;
  //   return h.reduce_impl(std::move(b));
  // }

  template <typename D, typename A, typename S> friend class histogram;
  friend class ::boost::python::access;
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

template <typename... Axes>
inline histogram<
    Dynamic, typename detail::combine<builtin_axes, mpl::vector<Axes...>>::type>
make_dynamic_histogram(Axes &&... axes) {

  return histogram<Dynamic, typename detail::combine<
                                builtin_axes, mpl::vector<Axes...>>::type>(
      std::forward<Axes>(axes)...);
}

template <typename Storage, typename... Axes>
inline histogram<
    Dynamic, typename detail::combine<builtin_axes, mpl::vector<Axes...>>::type,
    Storage>
make_dynamic_histogram_with(Axes &&... axes) {
  return histogram<
      Dynamic,
      typename detail::combine<builtin_axes, mpl::vector<Axes...>>::type,
      Storage>(std::forward<Axes>(axes)...);
}

} // namespace histogram
} // namespace boost

#endif
