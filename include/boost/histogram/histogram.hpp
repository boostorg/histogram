// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/histogram/arithmetic_operators.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/index_mapper.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/iterator.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/mp11.hpp>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

// forward declaration for serialization
namespace boost {
namespace serialization {
class access;
}
}

namespace boost {
namespace histogram {

template <typename Axes, typename Storage>
class histogram {
  static_assert(mp11::mp_size<Axes>::value > 0, "at least one axis required");

public:
  using axes_type = Axes;
  using storage_type = Storage;
  using element_type = typename storage_type::element_type;
  using scale_type = detail::arg_type<1, decltype(&Storage::operator*=)>;
  using const_reference = typename storage_type::const_reference;
  using const_iterator = iterator_over<histogram>;

  histogram() = default;
  histogram(const histogram& rhs) = default;
  histogram(histogram&& rhs) = default;
  histogram& operator=(const histogram& rhs) = default;
  histogram& operator=(histogram&& rhs) = default;

  template <typename A, typename S>
  explicit histogram(const histogram<A, S>& rhs) : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename A, typename S>
  histogram& operator=(const histogram<A, S>& rhs) {
    storage_ = rhs.storage_;
    detail::axes_assign(axes_, rhs.axes_);
    return *this;
  }

  explicit histogram(const axes_type& a, const storage_type& s = storage_type())
      : axes_(a), storage_(s) {
    storage_.reset(detail::bincount(axes_));
  }

  explicit histogram(axes_type&& a, storage_type&& s = storage_type())
      : axes_(std::move(a)), storage_(std::move(s)) {
    storage_.reset(detail::bincount(axes_));
  }

  template <typename A, typename S>
  bool operator==(const histogram<A, S>& rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename A, typename S>
  bool operator!=(const histogram<A, S>& rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename A, typename S>
  histogram& operator+=(const histogram<A, S>& rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::invalid_argument("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  histogram& operator*=(const scale_type rhs) {
    storage_ *= rhs;
    return *this;
  }

  histogram& operator/=(const scale_type rhs) {
    static_assert(std::is_floating_point<scale_type>::value,
                  "division requires a floating point type");
    storage_ *= scale_type(1) / rhs;
    return *this;
  }

  /// Number of axes (dimensions) of histogram
  std::size_t dim() const noexcept { return detail::axes_size(axes_); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const noexcept { return storage_.size(); }

  /// Reset bin counters to zero
  void reset() { storage_.reset(storage_.size()); }

  /// Get N-th axis (const version)
  template <std::size_t N>
  auto axis(mp11::mp_size_t<N>) const -> const detail::axis_at<N, axes_type>& {
    detail::range_check<N>(axes_);
    return detail::axis_get<N>(axes_);
  }

  /// Get N-th axis
  template <std::size_t N>
  auto axis(mp11::mp_size_t<N>) -> detail::axis_at<N, axes_type>& {
    detail::range_check<N>(axes_);
    return detail::axis_get<N>(axes_);
  }

  /// Get first axis (convenience for 1-d histograms, const version)
  const detail::axis_at<0, axes_type>& axis() const { return axis(mp11::mp_size_t<0>()); }

  /// Get first axis (convenience for 1-d histograms)
  detail::axis_at<0, axes_type>& axis() { return axis(mp11::mp_size_t<0>()); }

  /// Get N-th axis with runtime index (const version)
  template <typename U = axes_type, typename = detail::requires_dynamic_axes<U>>
  const detail::axis_at<0, U>& axis(std::size_t i) const {
    BOOST_ASSERT_MSG(i < axes_.size(), "index out of range");
    return axes_[i];
  }

  /// Get N-th axis with runtime index
  template <typename U = axes_type, typename = detail::requires_dynamic_axes<U>>
  detail::axis_at<0, U>& axis(std::size_t i) {
    BOOST_ASSERT_MSG(i < axes_.size(), "index out of range");
    return axes_[i];
  }

  /// Apply unary functor/function to each axis
  template <typename Unary>
  void for_each_axis(Unary&& unary) const {
    detail::for_each_axis(axes_, std::forward<Unary>(unary));
  }

  /// Fill histogram with a value tuple
  template <typename... Ts>
  void operator()(const Ts&... ts) {
    // case with one argument needs special treatment, specialized below
    const auto index = detail::call_impl(detail::no_container_tag(), axes_, ts...);
    if (index) storage_.increase(*index);
  }

  /// Fill histogram with a weight and a value tuple
  template <typename U, typename... Ts>
  void operator()(detail::weight_type<U>&& w, const Ts&... ts) {
    // case with one argument needs special treatment, specialized below
    const auto index = detail::call_impl(detail::no_container_tag(), axes_, ts...);
    if (index) storage_.add(*index, w);
  }

  /// Access bin counter at indices
  template <typename... Ts>
  const_reference at(const Ts&... ts) const {
    // case with one argument is ambiguous, is specialized below
    const auto index =
        detail::at_impl(detail::no_container_tag(), axes_, static_cast<int>(ts)...);
    return storage_[index];
  }

  template <typename T>
  void operator()(const T& t) {
    // check whether we need to unpack argument
    const auto index = detail::call_impl(detail::classify_container<T>(), axes_, t);
    if (index) storage_.increase(*index);
  }

  template <typename U, typename T>
  void operator()(detail::weight_type<U>&& w, const T& t) {
    // check whether we need to unpack argument
    const auto index = detail::call_impl(detail::classify_container<T>(), axes_, t);
    if (index) storage_.add(*index, w);
  }

  template <typename T>
  const_reference at(const T& t) const {
    // check whether we need to unpack argument;
    return storage_[detail::at_impl(detail::classify_container<T>(), axes_, t)];
  }

  /// Access bin counter at index
  template <typename T>
  const_reference operator[](const T& t) const {
    return at(t);
  }

  /// Returns a lower-dimensional histogram
  // precondition: argument sequence must be strictly ascending axis indices
  template <std::size_t I, typename... Ns>
  auto reduce_to(mp11::mp_size_t<I>, Ns...) const
      -> histogram<detail::sub_axes<axes_type, mp11::mp_size_t<I>, Ns...>, storage_type> {
    using N = mp11::mp_size_t<I>;
    using LN = mp11::mp_list<N, Ns...>;
    detail::range_check<detail::mp_last<LN>::value>(axes_);
    using sub_axes_type = detail::sub_axes<axes_type, N, Ns...>;
    using HR = histogram<sub_axes_type, storage_type>;
    auto sub_axes = detail::make_sub_axes(axes_, N(), Ns()...);
    auto hr = HR(std::move(sub_axes), storage_type(storage_.get_allocator()));
    const auto b = detail::bool_mask<N, Ns...>(dim(), true);
    std::vector<unsigned> shape(dim());
    for_each_axis(detail::shape_collector(shape.begin()));
    detail::index_mapper m(shape, b);
    do { hr.storage_.add(m.second, storage_[m.first]); } while (m.next());
    return hr;
  }

  /// Returns a lower-dimensional histogram
  // precondition: sequence must be strictly ascending axis indices
  template <typename Iterator, typename U = axes_type,
            typename = detail::requires_dynamic_axes<U>,
            typename = detail::requires_iterator<Iterator>>
  histogram reduce_to(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::is_sorted(begin, end, std::less_equal<decltype(*begin)>()),
                     "integer sequence must be strictly ascending");
    BOOST_ASSERT_MSG(begin == end || static_cast<unsigned>(*(end - 1)) < dim(),
                     "index out of range");
    auto sub_axes = histogram::axes_type(axes_.get_allocator());
    sub_axes.reserve(std::distance(begin, end));
    auto b = std::vector<bool>(dim(), false);
    for (auto it = begin; it != end; ++it) {
      sub_axes.push_back(axes_[*it]);
      b[*it] = true;
    }
    auto hr = histogram(std::move(sub_axes), storage_type(storage_.get_allocator()));
    std::vector<unsigned> shape(dim());
    for_each_axis(detail::shape_collector(shape.begin()));
    detail::index_mapper m(shape, b);
    do { hr.storage_.add(m.second, storage_[m.first]); } while (m.next());
    return hr;
  }

  const_iterator begin() const noexcept { return const_iterator(*this, 0); }

  const_iterator end() const noexcept { return const_iterator(*this, size()); }

private:
  axes_type axes_;
  Storage storage_;

  template <typename A, typename S>
  friend class histogram;
  template <typename H>
  friend class iterator_over;
  friend class python_access;
  friend class ::boost::serialization::access;
  template <typename Archive>
  void serialize(Archive&, unsigned);
};

/// static type factory
template <typename... Ts>
histogram<static_axes<detail::rm_cv_ref<Ts>...>> make_static_histogram(Ts&&... axis) {
  using histogram_type = histogram<static_axes<detail::rm_cv_ref<Ts>...>>;
  auto axes = typename histogram_type::axes_type(std::forward<Ts>(axis)...);
  return histogram_type(std::move(axes));
}

/// static type factory with custom storage type
template <typename Storage, typename... Ts>
histogram<static_axes<detail::rm_cv_ref<Ts>...>, detail::rm_cv_ref<Storage>>
make_static_histogram_with(Storage&& s, Ts&&... axis) {
  using histogram_type =
      histogram<static_axes<detail::rm_cv_ref<Ts>...>, detail::rm_cv_ref<Storage>>;
  auto axes = typename histogram_type::axes_type(std::forward<Ts>(axis)...);
  return histogram_type(std::move(axes), std::move(s));
}

/// dynamic type factory
template <typename T, typename... Ts>
histogram<mp11::mp_rename<
    mp11::mp_set_push_back<axis::any_std, detail::rm_cv_ref<T>, detail::rm_cv_ref<Ts>...>,
    dynamic_axes>>
make_dynamic_histogram(T&& axis0, Ts&&... axis) {
  using histogram_type = histogram<
      mp11::mp_rename<mp11::mp_set_push_back<axis::any_std, detail::rm_cv_ref<T>,
                                             detail::rm_cv_ref<Ts>...>,
                      dynamic_axes>>;
  using value_type = typename histogram_type::axes_type::value_type;
  auto axes = typename histogram_type::axes_type(
      {value_type(std::forward<T>(axis0)), value_type(std::forward<Ts>(axis))...});
  return histogram_type(std::move(axes));
}

/// dynamic type factory with custom storage type
template <typename Storage, typename T, typename... Ts>
histogram<mp11::mp_rename<mp11::mp_set_push_back<axis::any_std, detail::rm_cv_ref<T>,
                                                 detail::rm_cv_ref<Ts>...>,
                          dynamic_axes>,
          detail::rm_cv_ref<Storage>>
make_dynamic_histogram_with(Storage&& s, T&& axis0, Ts&&... axis) {
  using histogram_type = histogram<
      mp11::mp_rename<mp11::mp_set_push_back<axis::any_std, detail::rm_cv_ref<T>,
                                             detail::rm_cv_ref<Ts>...>,
                      dynamic_axes>,
      detail::rm_cv_ref<Storage>>;
  using axis_type = typename histogram_type::axes_type::value_type;
  auto a = axis0.get_allocator();
  auto axes = typename histogram_type::axes_type(
      {axis_type(std::forward<T>(axis0)), axis_type(std::forward<Ts>(axis))...}, a);
  return histogram_type(std::move(axes), std::move(s));
}

/// dynamic type factory from axis iterators
template <typename Iterator, typename = detail::requires_iterator<Iterator>>
histogram<mp11::mp_rename<
    detail::mp_set_union<axis::any_std, typename Iterator::value_type::types>,
    dynamic_axes>>
make_dynamic_histogram(Iterator begin, Iterator end) {
  BOOST_ASSERT_MSG(std::distance(begin, end) > 0, "at least one axis required");
  using histogram_type = histogram<mp11::mp_rename<
      detail::mp_set_union<axis::any_std, typename Iterator::value_type::types>,
      dynamic_axes>>;
  // auto axes = typename histogram_type::axes_type(begin, end, begin->get_allocator());
  auto axes = typename histogram_type::axes_type(begin, end);
  return histogram_type(std::move(axes));
}

/// dynamic type factory from axis iterators with custom storage type
template <typename Storage, typename Iterator,
          typename = detail::requires_iterator<Iterator>>
histogram<mp11::mp_rename<
              detail::mp_set_union<axis::any_std, typename Iterator::value_type::types>,
              dynamic_axes>,
          Storage>
make_dynamic_histogram_with(Storage&& s, Iterator begin, Iterator end) {
  BOOST_ASSERT_MSG(std::distance(begin, end) > 0, "at least one axis required");
  using histogram_type =
      histogram<mp11::mp_rename<detail::mp_set_union<
                                    axis::any_std, typename Iterator::value_type::types>,
                                dynamic_axes>,
                Storage>;
  // auto axes = typename histogram_type::axes_type(begin, end, begin->get_allocator());
  auto axes = typename histogram_type::axes_type(begin, end);
  return histogram_type(std::move(axes), std::move(s));
}

} // namespace histogram
} // namespace boost

#endif
