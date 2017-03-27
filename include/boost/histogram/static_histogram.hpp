// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STATIC_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_STATIC_HISTOGRAM_HPP_

#include <boost/config.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/algorithm.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/sequence.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/sequence/comparison.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/variance.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/vector.hpp>
#include <type_traits>

namespace boost {
namespace histogram {

template <typename Axes, typename Storage = adaptive_storage<>>
class static_histogram {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");
  using pairs = std::pair<std::size_t, std::size_t>;

public:
  using histogram_tag = detail::histogram_tag;
  using value_type = typename Storage::value_type;

private:
  using axes_type = typename fusion::result_of::as_vector<Axes>::type;

public:
  static_histogram() = default;

  template <typename... Axes1>
  explicit static_histogram(const Axes1 &... axes) : axes_(axes...) {
    storage_ = Storage(field_count());
  }

  static_histogram(const static_histogram &other) = default;
  static_histogram(static_histogram &&other) = default;
  static_histogram &operator=(const static_histogram &other) = default;
  static_histogram &operator=(static_histogram &&other) = default;

  template <typename OtherStorage>
  explicit static_histogram(const static_histogram<Axes, OtherStorage> &other)
      : axes_(other.axes_), storage_(other.storage_) {}

  template <typename OtherStorage>
  explicit static_histogram(static_histogram<Axes, OtherStorage> &&other)
      : axes_(std::move(other.axes_)), storage_(std::move(other.storage_)) {}

  template <typename OtherStorage>
  static_histogram &
  operator=(const static_histogram<Axes, OtherStorage> &other) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&other)) {
      axes_ = other.axes_;
      storage_ = other.storage_;
    }
    return *this;
  }

  template <typename OtherStorage>
  static_histogram &operator=(static_histogram<Axes, OtherStorage> &&other) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&other)) {
      axes_ = std::move(other.axes_);
      storage_ = std::move(other.storage_);
    }
    return *this;
  }

  template <typename OtherAxes, typename OtherStorage>
  bool
  operator==(const static_histogram<OtherAxes, OtherStorage> &other) const {
    if (!axes_equal_to(other.axes_)) {
      return false;
    }
    return storage_ == other.storage_;
  }

  template <typename OtherStorage>
  static_histogram &
  operator+=(const static_histogram<Axes, OtherStorage> &other) {
    if (!axes_equal_to(other.axes_)) {
      throw std::logic_error("axes of histograms differ");
    }
    storage_ += other.storage_;
    return *this;
  }

  template <typename... Values> void fill(Values... values) {
    static_assert(sizeof...(values) == dim(),
                  "number of arguments does not match histogram dimension");
    const auto p = apply_lin<detail::xlin, Values...>(pairs(0, 1), values...);
    if (p.second) {
      storage_.increase(p.first);
    }
  }

  template <
      bool has_weight_support = detail::has_weight_support<Storage>::value,
      typename... Values>
  typename std::enable_if<has_weight_support>::type wfill(value_type w,
                                                          Values... values) {
    static_assert(sizeof...(values) == dim(),
                  "number of arguments does not match histogram dimension");
    const auto p = apply_lin<detail::xlin, Values...>(pairs(0, 1), values...);
    if (p.second) {
      storage_.increase(p.first, w);
    }
  }

  template <typename... Indices> value_type value(Indices... indices) const {
    static_assert(sizeof...(indices) == dim(),
                  "number of arguments does not match histogram dimension");
    const auto p = apply_lin<detail::lin, Indices...>(pairs(0, 1), indices...);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(p.first);
  }

  template <typename... Indices> value_type variance(Indices... indices) const {
    static_assert(sizeof...(indices) == dim(),
                  "number of arguments does not match histogram dimension");
    const auto p = apply_lin<detail::lin, Indices...>(pairs(0, 1), indices...);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return detail::variance(storage_, p.first);
  }

  /// Number of axes (dimensions) of histogram
  static constexpr unsigned dim() {
    return fusion::result_of::size<Axes>::type::value;
  }

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

  template <unsigned N = 0>
  typename std::add_const<
      typename fusion::result_of::value_at_c<axes_type, N>::type>::type &
  axis() const {
    static_assert(N < fusion::result_of::size<axes_type>::value,
                  "axis index out of range");
    return fusion::at_c<N>(axes_);
  }

  /// Apply unary functor/function to each axis
  template <typename Unary> void for_each_axis(Unary &unary) const {
    fusion::for_each(axes_, unary);
  }

private:
  axes_type axes_;
  Storage storage_;

  std::size_t field_count() const {
    detail::field_count fc;
    fusion::for_each(axes_, fc);
    return fc.value;
  }

  template <typename OtherAxes>
  bool axes_equal_to(const OtherAxes & /*unused*/) const {
    return false;
  }

  bool axes_equal_to(const axes_type &other_axes) const {
    return axes_ == other_axes;
  }

  template <template <class, class> class Lin, typename First, typename... Rest>
  pairs apply_lin(pairs&& p, const First &x,
                 const Rest &... rest) const {
    Lin<typename fusion::result_of::value_at_c<
            axes_type, (dim() - 1 - sizeof...(Rest))>::type,
        First>::apply(p.first, p.second,
                      fusion::at_c<(dim() - 1 - sizeof...(Rest))>(axes_), x);
    return apply_lin<Lin, Rest...>(std::move(p), rest...);
  }

  template <template <class, class> class Lin>
  pairs apply_lin(pairs&& p) const { return p; }

  template <typename OtherAxes, typename OtherStorage>
  friend class static_histogram;

  template <class Archive, class OtherStorage, class OtherAxes>
  friend void serialize(Archive &, static_histogram<OtherStorage, OtherAxes> &,
                        unsigned);
};

/// default static type factory
template <typename... Axes>
inline static_histogram<mpl::vector<Axes...>>
make_static_histogram(const Axes &... axes) {
  return static_histogram<mpl::vector<Axes...>>(axes...);
}

/// static type factory with variable storage type
template <typename Storage, typename... Axes>
inline static_histogram<mpl::vector<Axes...>, Storage>
make_static_histogram_with(const Axes &... axes) {
  return static_histogram<mpl::vector<Axes...>, Storage>(axes...);
}
} // namespace histogram
} // namespace boost

#endif
