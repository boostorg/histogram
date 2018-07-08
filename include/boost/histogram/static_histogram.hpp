// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_

#include <boost/assert.hpp>
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
#include <boost/histogram/arithmetic_operators.hpp>
#include <boost/histogram/axis/axis.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/iterator.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/count_if.hpp>
#include <boost/mpl/deref.hpp>
#include <boost/mpl/empty.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <type_traits>
#include <utility>

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
  using element_type = typename Storage::element_type;
  using const_reference = typename Storage::const_reference;
  using const_iterator = iterator_over<histogram, Storage>;
  using iterator = const_iterator;

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
      : axes_(rhs.axes_), storage_(rhs.storage_) {}

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
  bool operator==(const histogram<static_tag, A, S> &) const noexcept {
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

  template <typename... Ts>
  bool operator!=(const histogram<Ts...> &rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename S>
  histogram &operator+=(const histogram<static_tag, Axes, S> &rhs) {
    BOOST_ASSERT_MSG(detail::axes_equal(axes_, rhs.axes_), "axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename A, typename S>
  histogram &operator+=(const histogram<dynamic_tag, A, S> &rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::invalid_argument("axes of histograms differ");
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

  template <typename... Ts> void operator()(Ts &&... ts) {
    // case with one argument is ambiguous, is specialized below
    static_assert(sizeof...(Ts) == axes_size::value,
                  "fill arguments do not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, ts...);
    if (stride)
      fill_storage_impl(idx);
  }

  template <typename T> void operator()(T && t) {
    // check whether we need to unpack argument
    fill_impl(detail::if_else<(axes_size::value == 1),
                detail::no_container_tag,
                detail::classify_container_t<T>>(), std::forward<T>(t));
  }

  template <typename W, typename... Ts> void operator()(detail::weight<W>&& w,
                                                        Ts &&... ts) {
    // case with one argument is ambiguous, is specialized below
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, ts...);
    if (stride)
      fill_storage_impl(idx, std::move(w));
  }

  template <typename W, typename T> void operator()(detail::weight<W>&& w,
                                                    T&&t) {
    // check whether we need to unpack argument
    fill_impl(detail::if_else<(axes_size::value == 1),
                detail::no_container_tag,
                detail::classify_container_t<T>>(), std::forward<T>(t), std::move(w));
  }

  template <typename... Indices>
  const_reference at(Indices &&... indices) const {
    // case with one argument is ambiguous, is specialized below
    static_assert(sizeof...(indices) == axes_size::value,
                  "bin arguments do not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, static_cast<int>(indices)...);
    if (stride == 0)
      throw std::out_of_range("bin index out of range");
    return storage_[idx];
  }

  template <typename T>
  const_reference operator[](T&&t) const {
    // check whether we need to unpack argument
    return at_impl(detail::classify_container_t<T>(), std::forward<T>(t));
  }

  template <typename T>
  const_reference at(T&&t) const {
    // check whether we need to unpack argument
    return at_impl(detail::classify_container_t<T>(), std::forward<T>(t));
  }

  /// Number of axes (dimensions) of histogram
  constexpr unsigned dim() const noexcept { return axes_size::value; }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t bincount() const noexcept { return storage_.size(); }

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
      static_tag,
      detail::axes_select_t<Axes, mpl::vector<mpl::int_<N>, Rest...>>,
      Storage> {
    using HR = histogram<
        static_tag,
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

  const_iterator begin() const noexcept {
    return const_iterator(*this, storage_, 0);
  }

  const_iterator end() const noexcept {
    return const_iterator(*this, storage_, storage_.size());
  }

private:
  axes_type axes_;
  Storage storage_;

  std::size_t bincount_from_axes() const noexcept {
    detail::field_count_visitor fc;
    fusion::for_each(axes_, fc);
    return fc.value;
  }

  template <typename T>
  void fill_storage_impl(std::size_t idx, detail::weight<T>&& w) {
    storage_.add(idx, w);
  }

  void fill_storage_impl(std::size_t idx) {
    storage_.increase(idx);
  }

  template <typename T, typename... Ts>
  void fill_impl(detail::dynamic_container_tag, T&&t, Ts&&...ts) {
    BOOST_ASSERT_MSG(std::distance(std::begin(t), std::end(t)) == axes_size::value,
                     "fill container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_iter(mpl::int_<axes_size::value>(), idx, stride, std::begin(t));
    if (stride) {
      fill_storage_impl(idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T, typename... Ts>
  void fill_impl(detail::static_container_tag, T&&t, Ts&&...ts) {
    static_assert(detail::size_of<T>::value == axes_size::value,
                  "fill container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_get(mpl::int_<axes_size::value>(), idx, stride, std::forward<T>(t));
    if (stride) {
      fill_storage_impl(idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T, typename... Ts>
  void fill_impl(detail::no_container_tag, T&&t, Ts&&... ts) {
    static_assert(axes_size::value == 1,
                  "fill argument does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, std::forward<T>(t));
    if (stride) {
      fill_storage_impl(idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T>
  const_reference at_impl(detail::dynamic_container_tag, T && t) const {
    BOOST_ASSERT_MSG(std::distance(std::begin(t), std::end(t)) == axes_size::value,
                     "bin container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_iter(mpl::int_<axes_size::value>(), idx, stride, std::begin(t));
    if (stride == 0)
      throw std::out_of_range("bin index out of range");
    return storage_[idx];
  }

  template <typename T>
  const_reference at_impl(detail::static_container_tag, T&&t) const {
    static_assert(detail::size_of<T>::value == axes_size::value,
                  "bin container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_get(mpl::int_<axes_size::value>(), idx, stride, std::forward<T>(t));
    if (stride == 0)
      throw std::out_of_range("bin index out of range");
    return storage_[idx];
  }

  template <typename T>
  const_reference at_impl(detail::no_container_tag, T&&t) const {
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, detail::indirect_int_cast(t));
    if (stride == 0)
      throw std::out_of_range("bin index out of range");
    return storage_[idx];
  }

  template <unsigned D> inline void xlin(std::size_t &, std::size_t &) const noexcept {}

  template <unsigned D, typename T, typename... Ts>
  inline void xlin(std::size_t &idx, std::size_t &stride, T&&t,
                   Ts&&... ts) const {
    const auto a_size = fusion::at_c<D>(axes_).size();
    const auto a_shape = fusion::at_c<D>(axes_).shape();
    const int j = fusion::at_c<D>(axes_).index(t);
    detail::lin(idx, stride, a_size, a_shape, j);
    xlin<D + 1>(idx, stride, std::forward<Ts>(ts)...);
  }

  template <typename Iterator>
  void xlin_iter(mpl::int_<0>, std::size_t &, std::size_t &, Iterator ) const noexcept {}

  template <int N, typename Iterator>
  void xlin_iter(mpl::int_<N>, std::size_t &idx, std::size_t &stride, Iterator iter) const {
    constexpr unsigned D = axes_size::value - N;
    const auto a_size = fusion::at_c<D>(axes_).size();
    const auto a_shape = fusion::at_c<D>(axes_).shape();
    const int j = fusion::at_c<D>(axes_).index(*iter);
    detail::lin(idx, stride, a_size, a_shape, j);
    xlin_iter(mpl::int_<N-1>(), idx, stride, ++iter);
  }

  template <unsigned D>
  inline void lin(std::size_t &, std::size_t &) const noexcept {}

  template <unsigned D, typename... Ts>
  inline void lin(std::size_t &idx, std::size_t &stride, int j,
                  Ts... ts) const noexcept {
    const auto a_size = fusion::at_c<D>(axes_).size();
    const auto a_shape = fusion::at_c<D>(axes_).shape();
    stride *= (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
    detail::lin(idx, stride, a_size, a_shape, j);
    lin<(D+1)>(idx, stride, ts...);
  }

  template <typename Iterator>
  void lin_iter(mpl::int_<0>, std::size_t &, std::size_t &, Iterator) const noexcept {}

  template <int N, typename Iterator>
  void lin_iter(mpl::int_<N>, std::size_t &idx, std::size_t &stride,
                Iterator iter) const noexcept {
    constexpr unsigned D = axes_size::value - N;
    const auto a_size = fusion::at_c<D>(axes_).size();
    const auto a_shape = fusion::at_c<D>(axes_).shape();
    const auto j = detail::indirect_int_cast(*iter);
    stride *= (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
    detail::lin(idx, stride, a_size, a_shape, j);
    lin_iter(mpl::int_<(N-1)>(), idx, stride, ++iter);
  }

  template <typename T>
  void xlin_get(mpl::int_<0>, std::size_t &, std::size_t &, T &&) const noexcept {}

  template <int N, typename T>
  void xlin_get(mpl::int_<N>, std::size_t &idx, std::size_t &stride, T && t) const {
    constexpr unsigned D = detail::size_of<T>::value - N;
    const auto a_size = fusion::at_c<D>(axes_).size();
    const auto a_shape = fusion::at_c<D>(axes_).shape();
    const auto j = fusion::at_c<D>(axes_).index(std::get<D>(t));
    detail::lin(idx, stride, a_size, a_shape, j);
    xlin_get(mpl::int_<(N-1)>(), idx, stride, std::forward<T>(t));
  }

  template <typename T>
  void lin_get(mpl::int_<0>, std::size_t &, std::size_t &, T&&) const noexcept {}

  template <int N, typename T>
  void lin_get(mpl::int_<N>, std::size_t &idx, std::size_t &stride, T&&t) const noexcept {
    constexpr unsigned D = detail::size_of<T>::value - N;
    const auto a_size = fusion::at_c<D>(axes_).size();
    const auto a_shape = fusion::at_c<D>(axes_).shape();
    const auto j = detail::indirect_int_cast(std::get<D>(t));
    stride *= (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
    detail::lin(idx, stride, a_size, a_shape, j);
    lin_get(mpl::int_<(N-1)>(), idx, stride, std::forward<T>(t));
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
inline histogram<static_tag, mpl::vector<Axis...>,
                 array_storage<weight_counter<double>>>
make_static_weighted_histogram(Axis &&... axis) {
  using h = histogram<static_tag, mpl::vector<Axis...>,
                      array_storage<weight_counter<double>>>;
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
