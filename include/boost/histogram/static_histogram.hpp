// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_IMPL_STATIC_HPP_

#include <boost/assert.hpp>
#include <boost/histogram/arithmetic_operators.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/index_cache.hpp>
#include <boost/histogram/detail/index_mapper.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/iterator.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/mp11.hpp>
#include <tuple>
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
  using axes_size = mp11::mp_size<Axes>;
  static_assert(axes_size::value > 0, "at least one axis required");

public:
  using axes_type = mp11::mp_rename<Axes, std::tuple>;
  using element_type = typename Storage::element_type;
  using const_reference = typename Storage::const_reference;
  using const_iterator = iterator_over<histogram>;
  using iterator = const_iterator;

  histogram() = default;
  histogram(const histogram& rhs) = default;
  histogram(histogram&& rhs) = default;
  histogram& operator=(const histogram& rhs) = default;
  histogram& operator=(histogram&& rhs) = default;

  template <typename Axis0, typename... Axis,
            typename = detail::requires_axis<Axis0>>
  explicit histogram(Axis0&& axis0, Axis&&... axis)
      : axes_(std::forward<Axis0>(axis0), std::forward<Axis>(axis)...) {
    storage_ = Storage(size_from_axes());
    index_cache_.reset(*this);
  }

  explicit histogram(axes_type&& axes) : axes_(std::move(axes)) {
    storage_ = Storage(size_from_axes());
    index_cache_.reset(*this);
  }

  template <typename S>
  explicit histogram(const static_histogram<Axes, S>& rhs)
      : axes_(rhs.axes_), storage_(rhs.storage_) {
    index_cache_.reset(*this);
  }

  template <typename S>
  histogram& operator=(const static_histogram<Axes, S>& rhs) {
    if (static_cast<const void*>(this) != static_cast<const void*>(&rhs)) {
      axes_ = rhs.axes_;
      storage_ = rhs.storage_;
      index_cache_.reset(*this);
    }
    return *this;
  }

  template <typename A, typename S>
  explicit histogram(const dynamic_histogram<A, S>& rhs)
      : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
    index_cache_.reset(*this);
  }

  template <typename A, typename S>
  histogram& operator=(const dynamic_histogram<A, S>& rhs) {
    if (static_cast<const void*>(this) != static_cast<const void*>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
      index_cache_.reset(*this);
    }
    return *this;
  }

  template <typename A, typename S>
  bool operator==(const static_histogram<A, S>&) const noexcept {
    return false;
  }

  template <typename S>
  bool operator==(const static_histogram<Axes, S>& rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename A, typename S>
  bool operator==(const dynamic_histogram<A, S>& rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename T, typename A, typename S>
  bool operator!=(const histogram<T, A, S>& rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename S>
  histogram& operator+=(const static_histogram<Axes, S>& rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::invalid_argument("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename A, typename S>
  histogram& operator+=(const dynamic_histogram<A, S>& rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::invalid_argument("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  template <typename T>
  histogram& operator*=(const T& rhs) {
    storage_ *= rhs;
    return *this;
  }

  template <typename T>
  histogram& operator/=(const T& rhs) {
    storage_ *= 1.0 / rhs;
    return *this;
  }

  template <typename... Ts>
  void operator()(Ts&&... ts) {
    // case with one argument is ambiguous, is specialized below
    static_assert(sizeof...(Ts) == axes_size::value,
                  "fill arguments do not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, ts...);
    if (stride) detail::fill_storage(storage_, idx);
  }

  template <typename T>
  void operator()(T&& t) {
    // check whether we need to unpack argument
    fill_impl(mp11::mp_if_c<(axes_size::value == 1), detail::no_container_tag,
                            detail::classify_container<T>>(),
              std::forward<T>(t));
  }

  // TODO: merge this with unpacking
  template <typename W, typename... Ts>
  void operator()(detail::weight<W>&& w, Ts&&... ts) {
    // case with one argument is ambiguous, is specialized below
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, ts...);
    if (stride) detail::fill_storage(storage_, idx, std::move(w));
  }

  // TODO: remove as obsolete
  template <typename W, typename T>
  void operator()(detail::weight<W>&& w, T&& t) {
    // check whether we need to unpack argument
    fill_impl(mp11::mp_if_c<(axes_size::value == 1), detail::no_container_tag,
                            detail::classify_container<T>>(),
              std::forward<T>(t), std::move(w));
  }

  template <typename... Ts>
  const_reference at(const Ts&... ts) const {
    // case with one argument is ambiguous, is specialized below
    static_assert(sizeof...(ts) == axes_size::value,
                  "bin arguments do not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, static_cast<int>(ts)...);
    return detail::storage_get(storage_, idx, stride == 0);
  }

  template <typename T>
  const_reference operator[](const T& t) const {
    return at(t);
  }

  template <typename T>
  const_reference at(const T& t) const {
    // check whether we need to unpack argument
    return at_impl(detail::classify_container<T>(), t);
  }

  /// Number of axes (dimensions) of histogram
  constexpr unsigned dim() const noexcept { return axes_size::value; }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const noexcept { return storage_.size(); }

  /// Reset bin counters to zero
  void reset() { storage_ = Storage(size_from_axes()); }

  /// Get N-th axis (const version)
  template <int N>
  typename std::add_const<typename std::tuple_element<N, axes_type>::type>::
      type& axis(mp11::mp_int<N>) const {
    static_assert(N < axes_size::value, "axis index out of range");
    return std::get<N>(axes_);
  }

  /// Get N-th axis
  template <int N>
  typename std::tuple_element<N, axes_type>::type& axis(mp11::mp_int<N>) {
    static_assert(N < axes_size::value, "axis index out of range");
    return std::get<N>(axes_);
  }

  // Get first axis (convenience for 1-d histograms, const version)
  constexpr typename std::add_const<
      typename std::tuple_element<0, axes_type>::type>::type&
  axis() const {
    return std::get<0>(axes_);
  }

  // Get first axis (convenience for 1-d histograms)
  typename std::tuple_element<0, axes_type>::type& axis() {
    return std::get<0>(axes_);
  }

  /// Apply unary functor/function to each axis
  template <typename Unary>
  void for_each_axis(Unary&& unary) const {
    mp11::tuple_for_each(axes_, std::forward<Unary>(unary));
  }

  /// Returns a lower-dimensional histogram
  template <int N, typename... Ns>
  auto reduce_to(mp11::mp_int<N>, Ns...) const
      -> static_histogram<detail::selection<Axes, mp11::mp_int<N>, Ns...>,
                          Storage> {
    using HR =
        static_histogram<detail::selection<Axes, mp11::mp_int<N>, Ns...>,
                         Storage>;
    auto hr =
        HR(detail::make_sub_tuple<axes_type, mp11::mp_int<N>, Ns...>(axes_));
    const auto b = detail::bool_mask<mp11::mp_int<N>, Ns...>(dim(), true);
    reduce_impl(hr, b);
    return hr;
  }

  const_iterator begin() const noexcept { return const_iterator(*this, 0); }

  const_iterator end() const noexcept {
    return const_iterator(*this, size());
  }

private:
  axes_type axes_;
  Storage storage_;
  mutable detail::index_cache index_cache_;

  std::size_t size_from_axes() const noexcept {
    detail::field_count_visitor v;
    for_each_axis(v);
    return v.value;
  }

  template <typename T, typename... Ts>
  void fill_impl(detail::dynamic_container_tag, T&& t, Ts&&... ts) {
    BOOST_ASSERT_MSG(
        std::distance(std::begin(t), std::end(t)) == axes_size::value,
        "fill container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_iter(axes_size(), idx, stride, std::begin(t));
    if (stride) {
      detail::fill_storage(storage_, idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T, typename... Ts>
  void fill_impl(detail::static_container_tag, T&& t, Ts&&... ts) {
    static_assert(detail::mp_size<T>::value == axes_size::value,
                  "fill container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_get(axes_size(), idx, stride, std::forward<T>(t));
    if (stride) {
      detail::fill_storage(storage_, idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T, typename... Ts>
  void fill_impl(detail::no_container_tag, T&& t, Ts&&... ts) {
    static_assert(axes_size::value == 1,
                  "fill argument does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, std::forward<T>(t));
    if (stride) {
      detail::fill_storage(storage_, idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T>
  const_reference at_impl(detail::dynamic_container_tag, const T& t) const {
    BOOST_ASSERT_MSG(
        std::distance(std::begin(t), std::end(t)) == axes_size::value,
        "bin container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_iter(axes_size(), idx, stride, std::begin(t));
    return detail::storage_get(storage_, idx, stride == 0);
  }

  template <typename T>
  const_reference at_impl(detail::static_container_tag, const T& t) const {
    static_assert(mp11::mp_size<T>::value == axes_size::value,
                  "bin container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_get(axes_size(), idx, stride, t);
    return detail::storage_get(storage_, idx, stride == 0);
  }

  template <typename T>
  const_reference at_impl(detail::no_container_tag, const T& t) const {
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, detail::indirect_int_cast(t));
    return detail::storage_get(storage_, idx, stride == 0);
  }

  template <unsigned D>
  void xlin(std::size_t&, std::size_t&) const noexcept {}

  template <unsigned D, typename T, typename... Ts>
  void xlin(std::size_t& idx, std::size_t& stride, T&& t, Ts&&... ts) const {
    const auto a_size = std::get<D>(axes_).size();
    const auto a_shape = std::get<D>(axes_).shape();
    const int j = std::get<D>(axes_).index(t);
    detail::lin(idx, stride, a_size, a_shape, j);
    xlin<D + 1>(idx, stride, std::forward<Ts>(ts)...);
  }

  template <typename Iterator>
  void xlin_iter(mp11::mp_size_t<0>, std::size_t&, std::size_t&,
                 Iterator) const noexcept {}

  template <long unsigned int N, typename Iterator>
  void xlin_iter(mp11::mp_size_t<N>, std::size_t& idx, std::size_t& stride,
                 Iterator iter) const {
    constexpr unsigned D = axes_size::value - N;
    const auto a_size = std::get<D>(axes_).size();
    const auto a_shape = std::get<D>(axes_).shape();
    const int j = std::get<D>(axes_).index(*iter);
    detail::lin(idx, stride, a_size, a_shape, j);
    xlin_iter(mp11::mp_size_t<N - 1>(), idx, stride, ++iter);
  }

  template <unsigned D>
  void lin(std::size_t&, std::size_t&) const noexcept {}

  template <unsigned D, typename... Ts>
  void lin(std::size_t& idx, std::size_t& stride, int j, Ts... ts) const
      noexcept {
    const auto a_size = std::get<D>(axes_).size();
    const auto a_shape = std::get<D>(axes_).shape();
    stride *= (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
    detail::lin(idx, stride, a_size, a_shape, j);
    lin<(D + 1)>(idx, stride, ts...);
  }

  template <typename Iterator>
  void lin_iter(mp11::mp_size_t<0>, std::size_t&, std::size_t&,
                Iterator) const noexcept {}

  template <long unsigned int N, typename Iterator>
  void lin_iter(mp11::mp_size_t<N>, std::size_t& idx, std::size_t& stride,
                Iterator iter) const noexcept {
    constexpr unsigned D = axes_size::value - N;
    const auto a_size = std::get<D>(axes_).size();
    const auto a_shape = std::get<D>(axes_).shape();
    const auto j = detail::indirect_int_cast(*iter);
    stride *= (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
    detail::lin(idx, stride, a_size, a_shape, j);
    lin_iter(mp11::mp_size_t<(N - 1)>(), idx, stride, ++iter);
  }

  template <typename T>
  void xlin_get(mp11::mp_size_t<0>, std::size_t&, std::size_t&, T&&) const
      noexcept {}

  template <long unsigned int N, typename T>
  void xlin_get(mp11::mp_size_t<N>, std::size_t& idx, std::size_t& stride,
                T&& t) const {
    constexpr unsigned D = detail::mp_size<T>::value - N;
    const auto a_size = std::get<D>(axes_).size();
    const auto a_shape = std::get<D>(axes_).shape();
    const auto j = std::get<D>(axes_).index(std::get<D>(t));
    detail::lin(idx, stride, a_size, a_shape, j);
    xlin_get(mp11::mp_size_t<(N - 1)>(), idx, stride, std::forward<T>(t));
  }

  template <typename T>
  void lin_get(mp11::mp_size_t<0>, std::size_t&, std::size_t&, T&&) const
      noexcept {}

  template <long unsigned int N, typename T>
  void lin_get(mp11::mp_size_t<N>, std::size_t& idx, std::size_t& stride,
               T&& t) const noexcept {
    constexpr unsigned D = detail::mp_size<T>::value - N;
    const auto a_size = std::get<D>(axes_).size();
    const auto a_shape = std::get<D>(axes_).shape();
    const auto j = detail::indirect_int_cast(std::get<D>(t));
    stride *= (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
    detail::lin(idx, stride, a_size, a_shape, j);
    lin_get(mp11::mp_size_t<(N - 1)>(), idx, stride, t);
  }

  template <typename H>
  void reduce_impl(H& h, const std::vector<bool>& b) const {
    detail::shape_vector_visitor v(dim());
    for_each_axis(v);
    detail::index_mapper m(v.shapes, b);
    do { h.storage_.add(m.second, storage_[m.first]); } while (m.next());
  }

  template <typename T, typename A, typename S>
  friend class histogram;
  template <typename H>
  friend class iterator_over;
  friend class ::boost::serialization::access;
  template <typename Archive>
  void serialize(Archive&, unsigned);
};

/// default static type factory
template <typename... Axis>
static_histogram<mp11::mp_list<detail::rm_cv_ref<Axis>...>>
make_static_histogram(Axis&&... axis) {
  return static_histogram<mp11::mp_list<detail::rm_cv_ref<Axis>...>>(
      std::forward<Axis>(axis)...);
}

/// static type factory with variable storage type
template <typename Storage, typename... Axis>
static_histogram<mp11::mp_list<detail::rm_cv_ref<Axis>...>, Storage>
make_static_histogram_with(Axis&&... axis) {
  return static_histogram<mp11::mp_list<detail::rm_cv_ref<Axis>...>, Storage>(
      std::forward<Axis>(axis)...);
}

} // namespace histogram
} // namespace boost

#endif
