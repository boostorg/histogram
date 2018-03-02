// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_

#include <boost/fusion/adapted/mpl.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/container/vector/convert.hpp>
#include <boost/fusion/include/algorithm.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/sequence.hpp>
#include <boost/fusion/sequence.hpp>
#include <boost/fusion/sequence/comparison.hpp>
#include <boost/histogram/axis/axis.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/iterator.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/count_if.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <type_traits>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
}
} // namespace boost

namespace boost {
namespace histogram {

template <typename Axes, typename Storage>
class histogram<static_tag, Axes, Storage> {
  static_assert(!mpl::empty<Axes>::value, "at least one axis required");
  using axes_size = typename fusion::result_of::size<Axes>::type;

public:
  using axes_type = typename fusion::result_of::as_vector<Axes>::type;
  using bin_type = typename Storage::bin_type;
  using bin_iterator = bin_iterator_over<Storage>;

  histogram() = default;
  histogram(const histogram &rhs) = default;
  histogram(histogram &&rhs) = default;
  histogram &operator=(const histogram &rhs) = default;
  histogram &operator=(histogram &&rhs) = default;

  template <typename... Axis>
  explicit histogram(const Axis &... axis) : axes_(axis...) {
    storage_ = Storage(bincount_from_axes());
  }

  explicit histogram(axes_type &&axes) : axes_(std::move(axes)) {
    storage_ = Storage(bincount_from_axes());
  }

  template <typename S>
  explicit histogram(const histogram<static_tag, Axes, S> &rhs)
      : storage_(rhs.storage_), axes_(rhs.axes_) {}

  template <typename S>
  histogram &operator=(const histogram<static_tag, Axes, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      axes_ = rhs.axes_;
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <typename A, typename S>
  explicit histogram(const histogram<dynamic_tag, A, S> &rhs)
      : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename A, typename S>
  histogram &operator=(const histogram<dynamic_tag, A, S> &rhs) {
    if (static_cast<const void *>(this) != static_cast<const void *>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
    }
    return *this;
  }

  template <typename A, typename S>
  bool operator==(const histogram<static_tag, A, S> &rhs) const noexcept {
    return false;
  }

  template <typename S>
  bool operator==(const histogram<static_tag, Axes, S> &rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename A, typename S>
  bool operator==(const histogram<dynamic_tag, A, S> &rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename T, typename A, typename S>
  bool operator!=(const histogram<T, A, S> &rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename S>
  histogram &operator+=(const histogram<static_tag, Axes, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::logic_error("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename T, typename A, typename S>
  histogram &operator+=(const histogram<T, A, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::logic_error("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename T> histogram &operator*=(const T &rhs) {
    storage_ *= rhs;
    return *this;
  }

  template <typename T> histogram &operator/=(const T &rhs) {
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
    static_assert(sizeof...(args) ==
                      (axes_size::value + n_weight::value + n_sample::value),
                  "number of arguments does not match histogram dimension");
    fill_impl(mpl::bool_<n_weight::value>(), mpl::bool_<n_sample::value>(),
              args...);
  }

  template <typename... Indices>
  bin_type bin(const Indices &... indices) const {
    static_assert(sizeof...(indices) == axes_size::value,
                  "number of arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, indices...);
    if (stride == 0) {
      throw std::out_of_range("invalid index");
    }
    return storage_[idx];
  }

  /// Number of axes (dimensions) of histogram
  constexpr unsigned dim() const noexcept { return axes_size::value; }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t bincount() const noexcept { return storage_.size(); }

  /// Sum of all counts in the histogram
  bin_type sum() const noexcept {
    bin_type result(0);
    for (std::size_t i = 0, n = storage_.size(); i < n; ++i)
      result += storage_[i];
    return result;
  }

  /// Reset bin counters to zero
  void reset() { storage_ = Storage(bincount_from_axes()); }

  /// Get N-th axis (const version)
  template <int N>
  typename std::add_const<
      typename fusion::result_of::value_at_c<axes_type, N>::type>::type &
  axis(mpl::int_<N>) const {
    static_assert(N < axes_size::value, "axis index out of range");
    return fusion::at_c<N>(axes_);
  }

  /// Get N-th axis
  template <int N>
  typename fusion::result_of::value_at_c<axes_type, N>::type &
  axis(mpl::int_<N>) {
    static_assert(N < axes_size::value, "axis index out of range");
    return fusion::at_c<N>(axes_);
  }

  // Get first axis (convenience for 1-d histograms, const version)
  constexpr typename std::add_const<
      typename fusion::result_of::value_at_c<axes_type, 0>::type>::type &
  axis() const {
    return fusion::at_c<0>(axes_);
  }

  // Get first axis (convenience for 1-d histograms)
  typename fusion::result_of::value_at_c<axes_type, 0>::type &axis() {
    return fusion::at_c<0>(axes_);
  }

  /// Apply unary functor/function to each axis
  template <typename Unary> void for_each_axis(Unary &&unary) const {
    fusion::for_each(axes_, unary);
  }

  /// Returns a lower-dimensional histogram
  template <int N, typename... Rest>
  auto reduce_to(mpl::int_<N>, Rest...) const -> histogram<
      static_tag, detail::axes_select_t<Axes, mpl::vector<mpl::int_<N>, Rest...>>,
      Storage> {
    using HR =
        histogram<static_tag,
                  detail::axes_select_t<Axes, mpl::vector<mpl::int_<N>, Rest...>>,
                  Storage>;
    typename HR::axes_type axes;
    detail::axes_assign_subset<mpl::vector<mpl::int_<N>, Rest...>>(axes, axes_);
    auto hr = HR(std::move(axes));
    const auto b =
        detail::bool_mask<mpl::vector<mpl::int_<N>, Rest...>>(dim(), true);
    reduce_impl(hr, b);
    return hr;
  }

  bin_iterator begin() const noexcept {
    return bin_iterator(*this, storage_);
  }

  bin_iterator end() const noexcept { return bin_iterator(storage_); }

private:
  axes_type axes_;
  Storage storage_;

  std::size_t bincount_from_axes() const noexcept {
    detail::field_count_visitor fc;
    fusion::for_each(axes_, fc);
    return fc.value;
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
    if (stride)
      storage_.add(idx, w);
  }

  template <typename... Args>
  inline void fill_impl(mpl::false_, mpl::true_, const Args &... args) {
    // not implemented
  }

  template <typename... Args>
  inline void fill_impl(mpl::true_, mpl::true_, const Args &... args) {
    // not implemented
  }

  template <unsigned D>
  inline void lin(std::size_t &, std::size_t &) const noexcept {}

  template <unsigned D, typename First, typename... Rest>
  inline void lin(std::size_t &idx, std::size_t &stride, const First &x,
                  const Rest &... rest) const noexcept {
    detail::lin(idx, stride, fusion::at_c<D>(axes_), x);
    return lin<D + 1>(idx, stride, rest...);
  }

  template <unsigned D, typename Weight>
  inline void xlin(std::size_t &, std::size_t &, Weight &) const {}

  template <unsigned D, typename Weight, typename First, typename... Rest>
  inline void xlin(std::size_t &idx, std::size_t &stride, Weight &w,
                   const First &first, const Rest &... rest) const {
    detail::xlin(idx, stride, fusion::at_c<D>(axes_), first);
    return xlin<D + 1>(idx, stride, w, rest...);
  }

  template <unsigned D, typename T, typename... Rest>
  inline void xlin(std::size_t &idx, std::size_t &stride, detail::weight_t<T> &w,
                   const detail::weight_t<T> &first, const Rest &... rest) const {
    w = first;
    return xlin<D>(idx, stride, w, rest...);
  }

  struct shape_assign_visitor {
    mutable std::vector<unsigned>::iterator ni;
    template <typename Axis> void operator()(const Axis &a) const {
      *ni = a.shape();
      ++ni;
    }
  };

  template <typename H>
  void reduce_impl(H &h, const std::vector<bool> &b) const {
    std::vector<unsigned> n(dim());
    for_each_axis(shape_assign_visitor{n.begin()});
    detail::index_mapper m(n, b);
    do {
      h.storage_.add(m.second, storage_[m.first]);
    } while (m.next());
  }

  template <typename T, typename A, typename S> friend class histogram;
  friend class ::boost::serialization::access;
  template <typename Archive> void serialize(Archive &, unsigned);
};

/// default static type factory
template <typename... Axis>
inline histogram<static_tag, mpl::vector<Axis...>>
make_static_histogram(Axis &&... axis) {
  using h = histogram<static_tag, mpl::vector<Axis...>>;
  return h(typename h::axes_type(std::forward<Axis>(axis)...));
}

template <typename... Axis>
inline histogram<static_tag, mpl::vector<Axis...>, array_storage<weight_counter<double>>>
make_static_weighted_histogram(Axis &&... axis) {
  using h = histogram<static_tag, mpl::vector<Axis...>, array_storage<weight_counter<double>>>;
  return h(typename h::axes_type(std::forward<Axis>(axis)...));
}


/// static type factory with variable storage type
template <typename Storage, typename... Axis>
inline histogram<static_tag, mpl::vector<Axis...>, Storage>
make_static_histogram_with(Axis &&... axis) {
  using h = histogram<static_tag, mpl::vector<Axis...>, Storage>;
  return h(typename h::axes_type(std::forward<Axis>(axis)...));
}

} // namespace histogram
} // namespace boost

#endif
