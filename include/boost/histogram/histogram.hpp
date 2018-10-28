// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_HISTOGRAM_HPP
#define BOOST_HISTOGRAM_HISTOGRAM_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/histogram/arithmetic_operators.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/index_mapper.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/iterator.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/mp11.hpp>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {

template <typename Axes, typename Container>
class histogram {
  static_assert(mp11::mp_size<Axes>::value > 0, "at least one axis required");

public:
  using axes_type = Axes;
  using container_type = Container;
  using storage_type = storage_adaptor<Container>;
  using value_type = typename storage_type::value_type;
  using const_reference = typename storage_type::const_reference;
  using const_iterator = iterator<histogram>;

  histogram() = default;
  histogram(const histogram& rhs) = default;
  histogram(histogram&& rhs) = default;
  histogram& operator=(const histogram& rhs) = default;
  histogram& operator=(histogram&& rhs) = default;

  template <typename A, typename C>
  explicit histogram(const histogram<A, C>& rhs) : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
  }

  template <typename A, typename C>
  histogram& operator=(const histogram<A, C>& rhs) {
    storage_ = rhs.storage_;
    detail::axes_assign(axes_, rhs.axes_);
    return *this;
  }

  explicit histogram(const axes_type& a, container_type c = container_type())
      : axes_(a), storage_(std::move(c)) {
    storage_.reset(detail::bincount(axes_));
  }

  explicit histogram(axes_type&& a, container_type c = container_type())
      : axes_(std::move(a)), storage_(std::move(c)) {
    storage_.reset(detail::bincount(axes_));
  }

  template <typename A, typename C>
  bool operator==(const histogram<A, C>& rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename A, typename C>
  bool operator!=(const histogram<A, C>& rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename A, typename C>
  histogram& operator+=(const histogram<A, C>& rhs) {
    if (!detail::axes_equal(axes_, rhs.axes_))
      throw std::invalid_argument("axes of histograms differ");
    storage_ += rhs.storage_;
    return *this;
  }

  histogram& operator*=(const double x) {
    storage_ *= x;
    return *this;
  }

  histogram& operator/=(const double x) {
    storage_ *= 1.0 / x;
    return *this;
  }

  /// Number of axes (dimensions) of histogram
  std::size_t rank() const noexcept { return detail::axes_size(axes_); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const noexcept { return storage_.size(); }

  /// Reset bin counters to zero
  void reset() { storage_.reset(storage_.size()); }

  /// Get N-th axis (const version)
  template <std::size_t N>
  decltype(auto) axis(mp11::mp_size_t<N>) const {
    detail::range_check<N>(axes_);
    return detail::axis_get<N>(axes_);
  }

  /// Get N-th axis
  template <std::size_t N>
  decltype(auto) axis(mp11::mp_size_t<N>) {
    detail::range_check<N>(axes_);
    return detail::axis_get<N>(axes_);
  }

  /// Get first axis (convenience for 1-d histograms, const version)
  decltype(auto) axis() const { return axis(mp11::mp_size_t<0>()); }

  /// Get first axis (convenience for 1-d histograms)
  decltype(auto) axis() { return axis(mp11::mp_size_t<0>()); }

  /// Get N-th axis with runtime index (const version)
  template <typename U = axes_type, typename = detail::requires_axis_vector<U>>
  decltype(auto) axis(std::size_t i) const {
    BOOST_ASSERT_MSG(i < axes_.size(), "index out of range");
    return axes_[i];
  }

  /// Get N-th axis with runtime index
  template <typename U = axes_type, typename = detail::requires_axis_vector<U>>
  decltype(auto) axis(std::size_t i) {
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
    if (index) storage_(*index);
  }

  template <typename T>
  void operator()(const T& t) {
    // check whether we need to unpack argument
    const auto index = detail::call_impl(detail::classify_container<T>(), axes_, t);
    if (index) storage_(*index);
  }

  /// Fill histogram with a weight and a value tuple
  template <typename U, typename... Ts>
  void operator()(const weight_type<U>& w, const Ts&... ts) {
    // case with one argument needs special treatment, specialized below
    const auto index = detail::call_impl(detail::no_container_tag(), axes_, ts...);
    if (index) storage_(*index, w);
  }

  template <typename U, typename T>
  void operator()(const weight_type<U>& w, const T& t) {
    // check whether we need to unpack argument
    const auto index = detail::call_impl(detail::classify_container<T>(), axes_, t);
    if (index) storage_(*index, w);
  }

  /// Access bin counter at indices
  template <typename... Ts>
  const_reference at(const Ts&... ts) const {
    // case with one argument is ambiguous, is specialized below
    const auto index =
        detail::at_impl(detail::no_container_tag(), axes_, static_cast<int>(ts)...);
    return storage_[index];
  }

  /// Access bin counter at index (specialization for 1D)
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
    auto hr = HR(std::move(sub_axes), storage_type());
    const auto b = detail::bool_mask<N, Ns...>(rank(), true);
    std::vector<unsigned> shape(rank());
    for_each_axis(detail::shape_collector(shape.begin()));
    detail::index_mapper m(shape, b);
    do { hr.storage_(m.second, storage_[m.first]); } while (m.next());
    return hr;
  }

  /// Returns a lower-dimensional histogram
  // precondition: sequence must be strictly ascending axis indices
  template <typename Iterator, typename U = axes_type,
            typename = detail::requires_axis_vector<U>,
            typename = detail::requires_iterator<Iterator>>
  histogram reduce_to(Iterator begin, Iterator end) const {
    BOOST_ASSERT_MSG(std::is_sorted(begin, end, std::less_equal<decltype(*begin)>()),
                     "integer sequence must be strictly ascending");
    BOOST_ASSERT_MSG(begin == end || static_cast<unsigned>(*(end - 1)) < rank(),
                     "index out of range");
    auto sub_axes = histogram::axes_type(axes_.get_allocator());
    sub_axes.reserve(std::distance(begin, end));
    auto b = std::vector<bool>(rank(), false);
    for (auto it = begin; it != end; ++it) {
      sub_axes.push_back(axes_[*it]);
      b[*it] = true;
    }
    auto hr = histogram(std::move(sub_axes), storage_type(storage_.get_allocator()));
    std::vector<unsigned> shape(rank());
    for_each_axis(detail::shape_collector(shape.begin()));
    detail::index_mapper m(shape, b);
    do { hr.storage_(m.second, storage_[m.first]); } while (m.next());
    return hr;
  }

  auto begin() const noexcept { return const_iterator(*this, 0); }

  auto end() const noexcept { return const_iterator(*this, size()); }

  template <typename Archive>
  void serialize(Archive&, unsigned);

private:
  axes_type axes_;
  storage_type storage_;

  template <typename A, typename S>
  friend class histogram;
  template <typename H>
  friend class iterator;
};

/// static type factory with custom storage type
template <typename S, typename T, typename... Ts, typename = detail::requires_axis<T>>
auto make_histogram_with(S&& s, T&& axis0, Ts&&... axis) {
  auto axes = std::make_tuple(std::forward<T>(axis0), std::forward<Ts>(axis)...);
  return histogram<decltype(axes), detail::unqual<S>>(std::move(axes),
                                                      std::forward<S>(s));
}

/// static type factory with standard storage type
template <typename T, typename... Ts, typename = detail::requires_axis<T>>
auto make_histogram(T&& axis0, Ts&&... axis) {
  return make_histogram_with(default_storage(), std::forward<T>(axis0),
                             std::forward<Ts>(axis)...);
}

/// dynamic type factory from vector-like with custom storage type
template <typename S, typename T, typename = detail::requires_axis_vector<T>>
auto make_histogram_with(S&& s, T&& t) {
  return histogram<detail::unqual<T>, detail::unqual<S>>(std::forward<T>(t),
                                                         std::forward<S>(s));
}

/// dynamic type factory from vector-like with standard storage type
template <typename T, typename = detail::requires_axis_vector<T>>
auto make_histogram(T&& t) {
  return make_histogram_with(default_storage(), std::forward<T>(t));
}

/// dynamic type factory from iterator range with custom storage type
template <typename Storage, typename Iterator,
          typename = detail::requires_iterator<Iterator>>
auto make_histogram_with(Storage&& s, Iterator begin, Iterator end) {
  using T = detail::iterator_value_type<Iterator>;
  auto axes = std::vector<T>(begin, end);
  return make_histogram_with(std::forward<Storage>(s), std::move(axes));
}

/// dynamic type factory from iterator range with standard storage type
template <typename Iterator, typename = detail::requires_iterator<Iterator>>
auto make_histogram(Iterator begin, Iterator end) {
  return make_histogram_with(default_storage(), begin, end);
}
} // namespace histogram
} // namespace boost

#endif
