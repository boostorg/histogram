// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_HISTOGRAM_HPP
#define BOOST_HISTOGRAM_HISTOGRAM_HPP

#include <boost/histogram/detail/at.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/common_type.hpp>
#include <boost/histogram/detail/compressed_pair.hpp>
#include <boost/histogram/detail/fill.hpp>
#include <boost/histogram/detail/fill_n.hpp>
#include <boost/histogram/detail/non_member_container_access.hpp>
#include <boost/histogram/detail/noop_mutex.hpp>
#include <boost/histogram/detail/static_if.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11/list.hpp>
#include <boost/throw_exception.hpp>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace boost {
namespace histogram {

/** Central class of the histogram library.

  Histogram uses the call operator to insert data, like the
  [Boost.Accumulators](https://www.boost.org/doc/libs/develop/doc/html/accumulators.html).

  Use factory functions (see
  [make_histogram.hpp](histogram/reference.html#header.boost.histogram.make_histogram_hpp)
  and
  [make_profile.hpp](histogram/reference.html#header.boost.histogram.make_profile_hpp)) to
  conveniently create histograms rather than calling the ctors directly.

  Use the [indexed](boost/histogram/indexed.html) range generator to iterate over filled
  histograms, which is convenient and faster than hand-written loops for multi-dimensional
  histograms.

  @tparam Axes std::tuple of axis types OR std::vector of an axis type or axis::variant
  @tparam Storage class that implements the storage interface
 */
template <class Axes, class Storage>
class histogram {
public:
  static_assert(mp11::mp_size<Axes>::value > 0, "at least one axis required");
  static_assert(std::is_same<std::decay_t<Storage>, Storage>::value,
                "Storage may not be a reference or const or volatile");

public:
  using axes_type = Axes;
  using storage_type = Storage;
  using value_type = typename storage_type::value_type;
  // typedefs for boost::range_iterator
  using iterator = typename storage_type::iterator;
  using const_iterator = typename storage_type::const_iterator;

  histogram() = default;
  histogram(const histogram& rhs)
      : axes_(rhs.axes_)
      , storage_and_mutex_(rhs.storage_and_mutex_.first())
      , offset_(rhs.offset_) {
    // no checks for too large axes needed
  }
  histogram(histogram&& rhs)
      : axes_(std::move(rhs.axes_))
      , storage_and_mutex_(std::move(rhs.storage_and_mutex_.first()))
      , offset_(std::exchange(rhs.offset_, 0)) {
    // no checks for too large axes needed
  }
  histogram& operator=(histogram&& rhs) {
    // no checks for too large axes needed
    if (this != &rhs) {
      axes_ = std::move(rhs.axes_);
      storage_and_mutex_.first() = std::move(rhs.storage_and_mutex_.first());
      offset_ = rhs.offset_;
    }
    return *this;
  }
  histogram& operator=(const histogram& rhs) {
    // no checks for too large axes needed
    if (this != &rhs) {
      axes_ = rhs.axes_;
      storage_and_mutex_.first() = rhs.storage_and_mutex_.first();
      offset_ = rhs.offset_;
    }
    return *this;
  }

  template <class A, class S>
  explicit histogram(histogram<A, S>&& rhs)
      : storage_and_mutex_(std::move(unsafe_access::storage(rhs)))
      , offset_(unsafe_access::offset(rhs)) {
    detail::axes_assign(axes_, std::move(unsafe_access::axes(rhs)));
    detail::throw_if_axes_is_too_large(axes_);
  }

  template <class A, class S>
  explicit histogram(const histogram<A, S>& rhs)
      : storage_and_mutex_(unsafe_access::storage(rhs))
      , offset_(unsafe_access::offset(rhs)) {
    detail::axes_assign(axes_, unsafe_access::axes(rhs));
    detail::throw_if_axes_is_too_large(axes_);
  }

  template <class A, class S>
  histogram& operator=(histogram<A, S>&& rhs) {
    detail::axes_assign(axes_, std::move(unsafe_access::axes(rhs)));
    detail::throw_if_axes_is_too_large(axes_);
    storage_and_mutex_.first() = std::move(unsafe_access::storage(rhs));
    offset_ = unsafe_access::offset(rhs);
    return *this;
  }

  template <class A, class S>
  histogram& operator=(const histogram<A, S>& rhs) {
    detail::axes_assign(axes_, unsafe_access::axes(rhs));
    detail::throw_if_axes_is_too_large(axes_);
    storage_and_mutex_.first() = unsafe_access::storage(rhs);
    offset_ = unsafe_access::offset(rhs);
    return *this;
  }

  template <class A, class S>
  histogram(A&& a, S&& s)
      : axes_(std::forward<A>(a))
      , storage_and_mutex_(std::forward<S>(s))
      , offset_(detail::offset(axes_)) {
    detail::throw_if_axes_is_too_large(axes_);
    storage_and_mutex_.first().reset(detail::bincount(axes_));
  }

  template <class A, class = detail::requires_axes<A>>
  explicit histogram(A&& a) : histogram(std::forward<A>(a), storage_type()) {}

  /// Number of axes (dimensions).
  constexpr unsigned rank() const noexcept { return detail::axes_rank(axes_); }

  /// Total number of bins (including underflow/overflow).
  std::size_t size() const noexcept { return storage_and_mutex_.first().size(); }

  /// Reset all bins to default initialized values.
  void reset() { storage_and_mutex_.first().reset(size()); }

  /// Get N-th axis using a compile-time number.
  /// This version is more efficient than the one accepting a run-time number.
  template <unsigned N = 0>
  decltype(auto) axis(std::integral_constant<unsigned, N> = {}) const {
    detail::axis_index_is_valid(axes_, N);
    return detail::axis_get<N>(axes_);
  }

  /// Get N-th axis with run-time number.
  /// Prefer the version that accepts a compile-time number, if you can use it.
  decltype(auto) axis(unsigned i) const {
    detail::axis_index_is_valid(axes_, i);
    return detail::axis_get(axes_, i);
  }

  /// Apply unary functor/function to each axis.
  template <class Unary>
  auto for_each_axis(Unary&& unary) const {
    return detail::for_each_axis(axes_, std::forward<Unary>(unary));
  }

  /** Fill histogram with values, an optional weight, and/or a sample.

    Arguments are passed in order to the axis objects. Passing an argument type that is
    not convertible to the value type accepted by the axis or passing the wrong number
    of arguments causes a throw of `std::invalid_argument`.

    __Optional weight__

    An optional weight can be passed as the first or last argument
    with the [weight](boost/histogram/weight.html) helper function. Compilation fails if
    the storage elements do not support weights.

    __Samples__

    If the storage elements accept samples, pass them with the sample helper function
    in addition to the axis arguments, which can be the first or last argument. The
    [sample](boost/histogram/sample.html) helper function can pass one or more arguments
    to the storage element. If samples and weights are used together, they can be passed
    in any order at the beginning or end of the argument list.

    __Axis with multiple arguments__

    If the histogram contains an axis which accepts a `std::tuple` of arguments, the
    arguments for that axis need to passed as a `std::tuple`, for example,
    `std::make_tuple(1.2, 2.3)`. If the histogram contains only this axis and no other,
    the arguments can be passed directly.
  */
  template <class... Args>
  iterator operator()(const Args&... args) {
    return operator()(std::forward_as_tuple(args...));
  }

  /// Fill histogram with values, an optional weight, and/or a sample from a `std::tuple`.
  template <class... Ts>
  iterator operator()(const std::tuple<Ts...>& args) {
    std::lock_guard<mutex_type> guard{storage_and_mutex_.second()};
    return detail::fill(offset_, storage_and_mutex_.first(), axes_, args);
  }

  /** Fill histogram with several values at once.

    The argument must be an iterable with a size that matches the
    rank of the histogram. The element of an iterable may be 1) a value or 2) an iterable
    with contiguous storage over values or 3) a variant of 1) and 2). Sub-iterables must
    have the same length.

    Values are passed to the corresponding histogram axis in order. If a single value is
    passed together with an iterable of values, the single value is treated like an
    iterable with matching length of copies of this value.

    If the histogram has only one axis, an iterable of values may be passed directly.

    @param args iterable as explained in the long description.
  */
  template <class Iterable, class = detail::requires_iterable<Iterable>>
  void fill(const Iterable& args) {
    std::lock_guard<mutex_type> guard{storage_and_mutex_.second()};
    detail::fill_n(offset_, storage_and_mutex_.first(), axes_, detail::make_span(args));
  }

  /** Fill histogram with several values and weights at once.

    @param args iterable of values.
    @param weights single weight or an iterable of weights.
  */
  template <class Iterable, class T, class = detail::requires_iterable<Iterable>>
  void fill(const Iterable& args, const weight_type<T>& weights) {
    std::lock_guard<mutex_type> guard{storage_and_mutex_.second()};
    detail::fill_n(offset_, storage_and_mutex_.first(), axes_, detail::make_span(args),
                   detail::to_ptr_size(weights.value));
  }

  /** Fill histogram with several values and weights at once.

    @param weights single weight or an iterable of weights.
    @param args iterable of values.
  */
  template <class Iterable, class T, class = detail::requires_iterable<Iterable>>
  void fill(const weight_type<T>& weights, const Iterable& args) {
    fill(args, weights);
  }

  /** Fill histogram with several values and samples at once.

    @param args iterable of values.
    @param samples single sample or an iterable of samples.
  */
  template <class Iterable, class T, class = detail::requires_iterable<Iterable>>
  void fill(const Iterable& args, const sample_type<T>& samples) {
    std::lock_guard<mutex_type> guard{storage_and_mutex_.second()};
    mp11::tuple_apply(
        [&](const auto&... sargs) {
          detail::fill_n(offset_, storage_and_mutex_.first(), axes_,
                         detail::make_span(args), detail::to_ptr_size(sargs)...);
        },
        samples.value);
  }

  /** Fill histogram with several values and samples at once.

    @param samples single sample or an iterable of samples.
    @param args iterable of values.
  */
  template <class Iterable, class T, class = detail::requires_iterable<Iterable>>
  void fill(const sample_type<T>& samples, const Iterable& args) {
    fill(args, samples);
  }

  template <class Iterable, class T, class U, class = detail::requires_iterable<Iterable>>
  void fill(const Iterable& args, const weight_type<T>& weights,
            const sample_type<U>& samples) {
    std::lock_guard<mutex_type> guard{storage_and_mutex_.second()};
    mp11::tuple_apply(
        [&](const auto&... sargs) {
          detail::fill_n(offset_, storage_and_mutex_.first(), axes_,
                         detail::make_span(args), detail::to_ptr_size(weights.value),
                         detail::to_ptr_size(sargs)...);
        },
        samples.value);
  }

  template <class Iterable, class T, class U, class = detail::requires_iterable<Iterable>>
  void fill(const sample_type<T>& samples, const weight_type<U>& weights,
            const Iterable& args) {
    fill(args, weights, samples);
  }

  template <class Iterable, class T, class U, class = detail::requires_iterable<Iterable>>
  void fill(const weight_type<T>& weights, const sample_type<U>& samples,
            const Iterable& args) {
    fill(args, weights, samples);
  }

  template <class Iterable, class T, class U, class = detail::requires_iterable<Iterable>>
  void fill(const Iterable& args, const sample_type<T>& samples,
            const weight_type<U>& weights) {
    fill(args, weights, samples);
  }

  /** Access cell value at integral indices.

    You can pass indices as individual arguments, as a std::tuple of integers, or as an
    interable range of integers. Passing the wrong number of arguments causes a throw of
    std::invalid_argument. Passing an index which is out of bounds causes a throw of
    std::out_of_range.

    @param i index of first axis.
    @param is indices of second, third, ... axes.
    @returns reference to cell value.
  */
  template <class... Indices>
  decltype(auto) at(axis::index_type i, Indices... is) {
    return at(std::forward_as_tuple(i, is...));
  }

  /// Access cell value at integral indices (read-only).
  template <class... Indices>
  decltype(auto) at(axis::index_type i, Indices... is) const {
    return at(std::forward_as_tuple(i, is...));
  }

  /// Access cell value at integral indices stored in `std::tuple`.
  template <class... Indices>
  decltype(auto) at(const std::tuple<Indices...>& is) {
    if (rank() != sizeof...(Indices))
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("number of arguments != histogram rank"));
    const auto idx = detail::at(axes_, is);
    if (!is_valid(idx))
      BOOST_THROW_EXCEPTION(std::out_of_range("at least one index out of bounds"));
    BOOST_ASSERT(idx < storage_and_mutex_.first().size());
    return storage_and_mutex_.first()[idx];
  }

  /// Access cell value at integral indices stored in `std::tuple` (read-only).
  template <typename... Indices>
  decltype(auto) at(const std::tuple<Indices...>& is) const {
    if (rank() != sizeof...(Indices))
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("number of arguments != histogram rank"));
    const auto idx = detail::at(axes_, is);
    if (!is_valid(idx))
      BOOST_THROW_EXCEPTION(std::out_of_range("at least one index out of bounds"));
    BOOST_ASSERT(idx < storage_and_mutex_.first().size());
    return storage_and_mutex_.first()[idx];
  }

  /// Access cell value at integral indices stored in iterable.
  template <class Iterable, class = detail::requires_iterable<Iterable>>
  decltype(auto) at(const Iterable& is) {
    if (rank() != detail::axes_rank(is))
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("number of arguments != histogram rank"));
    const auto idx = detail::at(axes_, is);
    if (!is_valid(idx))
      BOOST_THROW_EXCEPTION(std::out_of_range("at least one index out of bounds"));
    BOOST_ASSERT(idx < storage_and_mutex_.first().size());
    return storage_and_mutex_.first()[idx];
  }

  /// Access cell value at integral indices stored in iterable (read-only).
  template <class Iterable, class = detail::requires_iterable<Iterable>>
  decltype(auto) at(const Iterable& is) const {
    if (rank() != detail::axes_rank(is))
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("number of arguments != histogram rank"));
    const auto idx = detail::at(axes_, is);
    if (!is_valid(idx))
      BOOST_THROW_EXCEPTION(std::out_of_range("at least one index out of bounds"));
    BOOST_ASSERT(idx < storage_and_mutex_.first().size());
    return storage_and_mutex_.first()[idx];
  }

  /// Access value at index (number for rank = 1, else `std::tuple` or iterable).
  template <class Indices>
  decltype(auto) operator[](const Indices& is) {
    return at(is);
  }

  /// Access value at index (read-only).
  template <class Indices>
  decltype(auto) operator[](const Indices& is) const {
    return at(is);
  }

  /// Equality operator, tests equality for all axes and the storage.
  template <class A, class S>
  bool operator==(const histogram<A, S>& rhs) const noexcept {
    // testing offset is redundant, but offers fast return if it fails
    return offset_ == unsafe_access::offset(rhs) &&
           detail::axes_equal(axes_, unsafe_access::axes(rhs)) &&
           storage_and_mutex_.first() == unsafe_access::storage(rhs);
  }

  /// Negation of the equality operator.
  template <class A, class S>
  bool operator!=(const histogram<A, S>& rhs) const noexcept {
    return !operator==(rhs);
  }

  /// Add values of another histogram.
  template <class A, class S,
            class = std::enable_if_t<detail::has_operator_radd<
                value_type, typename histogram<A, S>::value_type>::value>>
  histogram& operator+=(const histogram<A, S>& rhs) {
    if (!detail::axes_equal(axes_, unsafe_access::axes(rhs)))
      BOOST_THROW_EXCEPTION(std::invalid_argument("axes of histograms differ"));
    auto rit = unsafe_access::storage(rhs).begin();
    auto& s = storage_and_mutex_.first();
    std::for_each(s.begin(), s.end(), [&rit](auto&& x) { x += *rit++; });
    return *this;
  }

  /// Subtract values of another histogram.
  template <class A, class S,
            class = std::enable_if_t<detail::has_operator_rsub<
                value_type, typename histogram<A, S>::value_type>::value>>
  histogram& operator-=(const histogram<A, S>& rhs) {
    if (!detail::axes_equal(axes_, unsafe_access::axes(rhs)))
      BOOST_THROW_EXCEPTION(std::invalid_argument("axes of histograms differ"));
    auto rit = unsafe_access::storage(rhs).begin();
    auto& s = storage_and_mutex_.first();
    std::for_each(s.begin(), s.end(), [&rit](auto&& x) { x -= *rit++; });
    return *this;
  }

  /// Multiply by values of another histogram.
  template <class A, class S,
            class = std::enable_if_t<detail::has_operator_rmul<
                value_type, typename histogram<A, S>::value_type>::value>>
  histogram& operator*=(const histogram<A, S>& rhs) {
    if (!detail::axes_equal(axes_, unsafe_access::axes(rhs)))
      BOOST_THROW_EXCEPTION(std::invalid_argument("axes of histograms differ"));
    auto rit = unsafe_access::storage(rhs).begin();
    auto& s = storage_and_mutex_.first();
    std::for_each(s.begin(), s.end(), [&rit](auto&& x) { x *= *rit++; });
    return *this;
  }

  /// Divide by values of another histogram.
  template <class A, class S,
            class = std::enable_if_t<detail::has_operator_rdiv<
                value_type, typename histogram<A, S>::value_type>::value>>
  histogram& operator/=(const histogram<A, S>& rhs) {
    if (!detail::axes_equal(axes_, unsafe_access::axes(rhs)))
      BOOST_THROW_EXCEPTION(std::invalid_argument("axes of histograms differ"));
    auto rit = unsafe_access::storage(rhs).begin();
    auto& s = storage_and_mutex_.first();
    std::for_each(s.begin(), s.end(), [&rit](auto&& x) { x /= *rit++; });
    return *this;
  }

  /// Multiply all values with a scalar.
  template <class V = value_type,
            class = std::enable_if_t<detail::has_operator_rmul<V, double>::value>>
  histogram& operator*=(const double x) {
    // use special implementation of scaling if available
    detail::static_if<detail::has_operator_rmul<storage_type, double>>(
        [](storage_type& s, auto x) { s *= x; },
        [](storage_type& s, auto x) {
          for (auto&& si : s) si *= x;
        },
        storage_and_mutex_.first(), x);
    return *this;
  }

  /// Divide all values by a scalar.
  template <class V = value_type,
            class = std::enable_if_t<detail::has_operator_rmul<V, double>::value>>
  histogram& operator/=(const double x) {
    return operator*=(1.0 / x);
  }

  /// Return value iterator to the beginning of the histogram.
  iterator begin() noexcept { return storage_and_mutex_.first().begin(); }

  /// Return value iterator to the end in the histogram.
  iterator end() noexcept { return storage_and_mutex_.first().end(); }

  /// Return value iterator to the beginning of the histogram (read-only).
  const_iterator begin() const noexcept { return storage_and_mutex_.first().begin(); }

  /// Return value iterator to the end in the histogram (read-only).
  const_iterator end() const noexcept { return storage_and_mutex_.first().end(); }

  /// Return value iterator to the beginning of the histogram (read-only).
  const_iterator cbegin() const noexcept { return begin(); }

  /// Return value iterator to the end in the histogram (read-only).
  const_iterator cend() const noexcept { return end(); }

private:
  axes_type axes_;

  using mutex_type = mp11::mp_if_c<(storage_type::has_threading_support &&
                                    detail::has_growing_axis<axes_type>::value),
                                   std::mutex, detail::noop_mutex>;

  detail::compressed_pair<storage_type, mutex_type> storage_and_mutex_;

  std::size_t offset_ = 0;

  friend struct unsafe_access;
};

/**
  Pairwise add cells of two histograms and return histogram with the sum.

  The returned histogram type is the most efficient and safest one constructible from the
  inputs, if they are not the same type. If one histogram has a tuple axis, the result has
  a tuple axis. The chosen storage is the one with the larger dynamic range.
*/
template <class A1, class S1, class A2, class S2>
auto operator+(const histogram<A1, S1>& a, const histogram<A2, S2>& b) {
  auto r = histogram<detail::common_axes<A1, A2>, detail::common_storage<S1, S2>>(a);
  return r += b;
}

/** Pairwise multiply cells of two histograms and return histogram with the product.

  For notes on the returned histogram type, see operator+.
*/
template <class A1, class S1, class A2, class S2>
auto operator*(const histogram<A1, S1>& a, const histogram<A2, S2>& b) {
  auto r = histogram<detail::common_axes<A1, A2>, detail::common_storage<S1, S2>>(a);
  return r *= b;
}

/** Pairwise subtract cells of two histograms and return histogram with the difference.

  For notes on the returned histogram type, see operator+.
*/
template <class A1, class S1, class A2, class S2>
auto operator-(const histogram<A1, S1>& a, const histogram<A2, S2>& b) {
  auto r = histogram<detail::common_axes<A1, A2>, detail::common_storage<S1, S2>>(a);
  return r -= b;
}

/** Pairwise divide cells of two histograms and return histogram with the quotient.

  For notes on the returned histogram type, see operator+.
*/
template <class A1, class S1, class A2, class S2>
auto operator/(const histogram<A1, S1>& a, const histogram<A2, S2>& b) {
  auto r = histogram<detail::common_axes<A1, A2>, detail::common_storage<S1, S2>>(a);
  return r /= b;
}

/** Multiply all cells of the histogram by a number and return a new histogram.

  If the original histogram has integer cells, the result has double cells.
*/
template <class A, class S>
auto operator*(const histogram<A, S>& h, double x) {
  auto r = histogram<A, detail::common_storage<S, dense_storage<double>>>(h);
  return r *= x;
}

/** Multiply all cells of the histogram by a number and return a new histogram.

  If the original histogram has integer cells, the result has double cells.
*/
template <class A, class S>
auto operator*(double x, const histogram<A, S>& h) {
  return h * x;
}

/** Divide all cells of the histogram by a number and return a new histogram.

  If the original histogram has integer cells, the result has double cells.
*/
template <class A, class S>
auto operator/(const histogram<A, S>& h, double x) {
  return h * (1.0 / x);
}

#if __cpp_deduction_guides >= 201606

template <class Axes>
histogram(Axes&& axes)->histogram<std::decay_t<Axes>, default_storage>;

template <class Axes, class Storage>
histogram(Axes&& axes, Storage&& storage)
    ->histogram<std::decay_t<Axes>, std::decay_t<Storage>>;

#endif

/** Helper function to mark argument as weight.

  @param t argument to be forward to the histogram.
*/
template <typename T>
auto weight(T&& t) noexcept {
  return weight_type<T>{std::forward<T>(t)};
}

/** Helper function to mark arguments as sample.

  @param ts arguments to be forwarded to the accumulator.
*/
template <typename... Ts>
auto sample(Ts&&... ts) noexcept {
  return sample_type<std::tuple<Ts...>>{std::forward_as_tuple(std::forward<Ts>(ts)...)};
}

template <class T>
struct weight_type {
  T value;
};

template <class T>
struct sample_type {
  T value;
};

} // namespace histogram
} // namespace boost

#endif
