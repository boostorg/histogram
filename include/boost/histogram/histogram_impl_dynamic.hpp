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
  histogram(const histogram &rhs) = default;
  histogram(histogram &&rhs) = default;
  histogram &operator=(const histogram &rhs) = default;
  histogram &operator=(histogram &&rhs) = default;

  // template <typename... Axes1>
  // explicit histogram(Axes1 &&... axes) :
  // axes_({axis_type(std::move(axes))...}) {
  //   storage_ = Storage(field_count());
  // }

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
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename... Args> void fill(Args... args) noexcept {
    using n_weight = typename mpl::count<mpl::vector<Args...>, weight>;
    static_assert(n_weight::value <= 1,
                  "arguments may contain at most one instance of type weight");
    BOOST_ASSERT_MSG(sizeof...(args) == dim() + n_weight::value,
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    double w = 0.0;
    apply_lin<detail::xlin, 0, Args...>(idx, stride, w, args...);
    if (stride) {
      if (n_weight::value)
        storage_.increase(idx, w);
      else
        storage_.increase(idx);
    }
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
  void fill(Iterator begin, Iterator end, const weight &w) noexcept {
    BOOST_ASSERT_MSG(std::distance(begin, end) == dim(),
                     "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    apply_lin_iter<detail::xlin>(idx, stride, begin);
    if (stride) {
      storage_.increase(idx, static_cast<double>(w));
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
    static_assert(detail::has_variance<Storage>::value,
                  "Storage lacks variance support");
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
    static_assert(detail::has_variance<Storage>::value,
                  "Storage lacks variance support");
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
    for (const auto &a : axes_) {
      apply_visitor(fc, a);
    }
    return fc.value;
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

  template <template <class, class> class Lin, unsigned D, typename First,
            typename... Rest>
  void apply_lin(std::size_t &idx, std::size_t &stride, const First &x,
                 const Rest &... rest) const {
    apply_visitor(lin_visitor<Lin, First>(idx, stride, x), axes_[D]);
    return apply_lin<Lin, D + 1, Rest...>(idx, stride, rest...);
  }

  template <template <class, class> class Lin, unsigned D>
  void apply_lin(std::size_t &idx, std::size_t &stride) const {}

  template <template <class, class> class Lin, unsigned D, typename First,
            typename... Rest>
  void apply_lin(std::size_t &idx, std::size_t &stride, double &w,
                 const First &x, const Rest &... rest) const {
    apply_visitor(lin_visitor<Lin, First>(idx, stride, x), axes_[D]);
    return apply_lin<Lin, D + 1, Rest...>(idx, stride, w, rest...);
  }

  template <template <class, class> class Lin, unsigned D, typename,
            typename... Rest>
  void apply_lin(std::size_t &idx, std::size_t &stride, double &w,
                 const weight &x, const Rest &... rest) const {
    w = static_cast<double>(x);
    return apply_lin<Lin, D, Rest...>(idx, stride, w, rest...);
  }

  template <template <class, class> class Lin, unsigned D>
  void apply_lin(std::size_t &idx, std::size_t &stride, double &w) const {}

  template <template <class, class> class Lin, typename Iterator>
  void apply_lin_iter(std::size_t &idx, std::size_t &stride,
                      Iterator iter) const {
    for (const auto &a : axes_) {
      apply_visitor(lin_visitor<Lin, decltype(*iter)>(idx, stride, *iter), a);
      ++iter;
    }
  }

  template <typename D, typename A, typename S> friend class histogram;

  friend class ::boost::python::access;
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

template <typename... Axes>
inline histogram<
    Dynamic, typename detail::combine<default_axes, mpl::vector<Axes...>>::type>
make_dynamic_histogram(Axes &&... axes) {

  return histogram<Dynamic, typename detail::combine<
                                default_axes, mpl::vector<Axes...>>::type>(
      std::forward<Axes>(axes)...);
}

template <typename Storage, typename... Axes>
inline histogram<
    Dynamic, typename detail::combine<default_axes, mpl::vector<Axes...>>::type,
    Storage>
make_dynamic_histogram_with(Axes &&... axes) {
  return histogram<
      Dynamic,
      typename detail::combine<default_axes, mpl::vector<Axes...>>::type,
      Storage>(std::forward<Axes>(axes)...);
}

} // namespace histogram
} // namespace boost

#endif
