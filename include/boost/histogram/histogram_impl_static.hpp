// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_

#include <boost/config.hpp>
#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/algorithm.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/sequence.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/sequence/comparison.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/axis.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/vector.hpp>
#include <type_traits>

namespace boost {
namespace histogram {

template <typename Axes, typename Storage>
class histogram<Static, Axes, Storage> {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");
  using size_pair = std::pair<std::size_t, std::size_t>;
  using axes_size = typename fusion::result_of::size<Axes>::type;

public:
  using value_type = typename Storage::value_type;

private:
  using axes_type = typename fusion::result_of::as_vector<Axes>::type;

public:
  histogram() = default;

  template <typename... Axes1>
  explicit histogram(const Axes1 &... axes) : axes_(axes...) {
    storage_ = Storage(field_count());
  }

  histogram(const histogram &rhs) = default;
  histogram(histogram &&rhs) = default;
  histogram &operator=(const histogram &rhs) = default;
  histogram &operator=(histogram &&rhs) = default;

  template <type D, typename A, typename S>
  explicit histogram(const histogram<D, A, S> &rhs)
      : storage_(rhs.storage_)
  {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <type D, typename A, typename S>
  histogram &operator=(const histogram<D, A, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <type D, typename A, typename S>
  bool operator==(const histogram<D, A, S> &rhs) const {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <type D, typename A, typename S>
  bool operator!=(const histogram<D, A, S> &rhs) const {
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

  template <typename... Values> void fill(Values... values) {
    static_assert(sizeof...(values) == axes_size::value,
                  "number of arguments does not match histogram dimension");
    const auto p =
        apply_lin<detail::xlin, Values...>(size_pair(0, 1), values...);
    if (p.second) {
      storage_.increase(p.first);
    }
  }

  template <typename... Values> void wfill(value_type w, Values... values) {
    static_assert(sizeof...(values) == axes_size::value,
                  "number of arguments does not match histogram dimension");
    const auto p =
        apply_lin<detail::xlin, Values...>(size_pair(0, 1), values...);
    if (p.second) {
      storage_.increase(p.first, w);
    }
  }

  template <typename... Indices> value_type value(Indices... indices) const {
    static_assert(sizeof...(indices) == axes_size::value,
                  "number of arguments does not match histogram dimension");
    const auto p =
        apply_lin<detail::lin, Indices...>(size_pair(0, 1), indices...);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.value(p.first);
  }

  template <typename... Indices> value_type variance(Indices... indices) const {
    static_assert(detail::has_variance<Storage>::value,
                  "Storage lacks variance support");
    static_assert(sizeof...(indices) == axes_size::value,
                  "number of arguments does not match histogram dimension");
    const auto p =
        apply_lin<detail::lin, Indices...>(size_pair(0, 1), indices...);
    if (p.second == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_.variance(p.first);
  }

  /// Number of axes (dimensions) of histogram
  constexpr unsigned dim() const { return axes_size::value; }

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
    static_assert(N < axes_size::value, "axis index out of range");
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

  template <template <class, class> class Lin, typename First, typename... Rest>
  size_pair apply_lin(size_pair &&p, const First &x,
                      const Rest &... rest) const {
    Lin<typename fusion::result_of::value_at_c<
            axes_type, (axes_size::value - 1 - sizeof...(Rest))>::type,
        First>::apply(p.first, p.second,
                      fusion::at_c<(axes_size::value - 1 - sizeof...(Rest))>(
                          axes_),
                      x);
    return apply_lin<Lin, Rest...>(std::move(p), rest...);
  }

  template <template <class, class> class Lin>
  size_pair apply_lin(size_pair &&p) const {
    return p;
  }

  template <type D, typename A, typename S> friend class histogram;

  template <class Archive, class S, class A>
  friend void serialize(Archive &, histogram<Static, S, A> &, unsigned);
};

/// default static type factory
template <typename... Axes>
inline histogram<Static, mpl::vector<Axes...>>
make_static_histogram(const Axes &... axes) {
  return histogram<Static, mpl::vector<Axes...>>(axes...);
}

/// static type factory with variable storage type
template <typename Storage, typename... Axes>
inline histogram<Static, mpl::vector<Axes...>, Storage>
make_static_histogram_with(const Axes &... axes) {
  return histogram<Static, mpl::vector<Axes...>, Storage>(axes...);
}

} // namespace histogram
} // namespace boost

#endif
