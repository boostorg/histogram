// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_DYNAMIC_IMPL_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_DYNAMIC_IMPL_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/histogram/arithmetic_operators.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/axis/types.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/index_cache.hpp>
#include <boost/histogram/detail/index_mapper.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/iterator.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <boost/mp11.hpp>
#include <boost/type_index.hpp>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
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
class histogram<dynamic_tag, Axes, Storage> {
  static_assert(mp11::mp_size<Axes>::value > 0, "at least one axis required");

public:
  using any_axis_type = mp11::mp_rename<Axes, axis::any>;
  using axes_type = std::vector<any_axis_type>;
  using element_type = typename Storage::element_type;
  using const_reference = typename Storage::const_reference;
  using const_iterator = iterator_over<histogram>;
  using iterator = const_iterator;

public:
  histogram() = default;
  histogram(const histogram&) = default;
  histogram(histogram&&) = default;
  histogram& operator=(const histogram&) = default;
  histogram& operator=(histogram&&) = default;

  template <typename Axis0, typename... Axis,
            typename = detail::requires_axis<Axis0>>
  explicit histogram(Axis0&& axis0, Axis&&... axis)
      : axes_({any_axis_type(std::forward<Axis0>(axis0)),
               any_axis_type(std::forward<Axis>(axis))...}) {
    storage_ = Storage(size_from_axes());
    index_cache_.reset(*this);
  }

  explicit histogram(axes_type&& axes) : axes_(std::move(axes)) {
    storage_ = Storage(size_from_axes());
    index_cache_.reset(*this);
  }

  template <typename Iterator, typename = detail::requires_iterator<Iterator>>
  histogram(Iterator begin, Iterator end) : axes_(std::distance(begin, end)) {
    std::copy(begin, end, axes_.begin());
    storage_ = Storage(size_from_axes());
    index_cache_.reset(*this);
  }

  template <typename T, typename A, typename S>
  explicit histogram(const histogram<T, A, S>& rhs) : storage_(rhs.storage_) {
    detail::axes_assign(axes_, rhs.axes_);
    index_cache_.reset(*this);
  }

  template <typename T, typename A, typename S>
  histogram& operator=(const histogram<T, A, S>& rhs) {
    if (static_cast<const void*>(this) != static_cast<const void*>(&rhs)) {
      detail::axes_assign(axes_, rhs.axes_);
      storage_ = rhs.storage_;
      index_cache_.reset(*this);
    }
    return *this;
  }

  template <typename S>
  explicit histogram(dynamic_histogram<Axes, S>&& rhs)
      : axes_(std::move(rhs.axes_)), storage_(std::move(rhs.storage_)) {
    index_cache_.reset(*this);
  }

  template <typename S>
  histogram& operator=(dynamic_histogram<Axes, S>&& rhs) {
    if (static_cast<const void*>(this) != static_cast<const void*>(&rhs)) {
      axes_ = std::move(rhs.axes_);
      storage_ = std::move(rhs.storage_);
      index_cache_.reset(*this);
    }
    return *this;
  }

  template <typename T, typename A, typename S>
  bool operator==(const histogram<T, A, S>& rhs) const noexcept {
    return detail::axes_equal(axes_, rhs.axes_) && storage_ == rhs.storage_;
  }

  template <typename T, typename A, typename S>
  bool operator!=(const histogram<T, A, S>& rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename T, typename A, typename S>
  histogram& operator+=(const histogram<T, A, S>& rhs) {
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
    BOOST_ASSERT_MSG(dim() == sizeof...(Ts),
                     "fill arguments does not match histogram dimension "
                     "(did you use weight() in the wrong place?)");
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, std::forward<Ts>(ts)...);
    if (stride) { detail::fill_storage(storage_, idx); }
  }

  template <typename T>
  void operator()(T&& t) {
    // check whether T is unpackable
    if (dim() == 1) {
      fill_impl(detail::no_container_tag(), std::forward<T>(t));
    } else {
      fill_impl(detail::classify_container<T>(), std::forward<T>(t));
    }
  }

  template <typename W, typename... Ts>
  void operator()(detail::weight<W>&& w, Ts&&... ts) {
    // case with one argument is ambiguous, is specialized below
    BOOST_ASSERT_MSG(dim() == sizeof...(Ts),
                     "fill arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, std::forward<Ts>(ts)...);
    if (stride) { detail::fill_storage(storage_, idx, std::move(w)); }
  }

  template <typename W, typename T>
  void operator()(detail::weight<W>&& w, T&& t) {
    // check whether T is unpackable
    if (dim() == 1) {
      fill_impl(detail::no_container_tag(), std::forward<T>(t), std::move(w));
    } else {
      fill_impl(detail::classify_container<T>(), std::forward<T>(t),
                std::move(w));
    }
  }

  template <typename... Ts>
  const_reference at(const Ts&... ts) const {
    // case with one argument is ambiguous, is specialized below
    BOOST_ASSERT_MSG(dim() == sizeof...(Ts),
                     "bin arguments does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, static_cast<int>(ts)...);
    return detail::storage_get(storage_, idx, stride == 0);
  }

  template <typename T>
  const_reference at(const T& t) const {
    // check whether T is unpackable
    return at_impl(detail::classify_container<T>(), t);
  }

  template <typename T>
  const_reference operator[](const T& t) const {
    return at(t);
  }

  /// Number of axes (dimensions) of histogram
  unsigned dim() const noexcept { return axes_.size(); }

  /// Total number of bins in the histogram (including underflow/overflow)
  std::size_t size() const noexcept { return storage_.size(); }

  /// Reset bin counters to zero
  void reset() { storage_ = Storage(size_from_axes()); }

  /// Return axis \a i
  any_axis_type& axis(unsigned i = 0) {
    BOOST_ASSERT_MSG(i < dim(), "axis index out of range");
    return axes_[i];
  }

  /// Return axis \a i (const version)
  const any_axis_type& axis(unsigned i = 0) const {
    BOOST_ASSERT_MSG(i < dim(), "axis index out of range");
    return axes_[i];
  }

  /// Apply unary functor/function to each axis
  template <typename Unary>
  void for_each_axis(Unary&& unary) const {
    for (const auto& a : axes_) {
      apply_visitor(detail::unary_visitor<Unary>(unary), a);
    }
  }

  /// Return a lower dimensional histogram
  template <int N, typename... Ts>
  histogram reduce_to(mp11::mp_int<N>, Ts...) const {
    const auto b = detail::bool_mask<mp11::mp_int<N>, Ts...>(dim(), true);
    return reduce_impl(b);
  }

  /// Return a lower dimensional histogram
  template <typename... Ts>
  histogram reduce_to(int n, Ts... ts) const {
    std::vector<bool> b(dim(), false);
    for (const auto& i : {n, int(ts)...}) b[i] = true;
    return reduce_impl(b);
  }

  /// Return a lower dimensional histogram
  template <typename Iterator, typename = detail::requires_iterator<Iterator>>
  histogram reduce_to(Iterator begin, Iterator end) const {
    std::vector<bool> b(dim(), false);
    for (; begin != end; ++begin) b[*begin] = true;
    return reduce_impl(b);
  }

  const_iterator begin() const noexcept { return const_iterator(*this, 0); }

  const_iterator end() const noexcept {
    return const_iterator(*this, storage_.size());
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
    BOOST_ASSERT_MSG(dim() == std::distance(std::begin(t), std::end(t)),
                     "fill container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_iter(idx, stride, std::begin(t));
    if (stride) {
      detail::fill_storage(storage_, idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T, typename... Ts>
  void fill_impl(detail::static_container_tag, T&& t, Ts&&... ts) {
    BOOST_ASSERT_MSG(dim() == detail::mp_size<T>::value,
                     "fill container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin_get(mp11::mp_int<detail::mp_size<T>::value>(), idx, stride,
             std::forward<T>(t));
    if (stride) {
      detail::fill_storage(storage_, idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T, typename... Ts>
  void fill_impl(detail::no_container_tag, T&& t, Ts&&... ts) {
    BOOST_ASSERT_MSG(dim() == 1,
                     "fill argument does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    xlin<0>(idx, stride, t);
    if (stride) {
      detail::fill_storage(storage_, idx, std::forward<Ts>(ts)...);
    }
  }

  template <typename T>
  const_reference at_impl(detail::dynamic_container_tag, const T& t) const {
    BOOST_ASSERT_MSG(dim() == std::distance(std::begin(t), std::end(t)),
                     "bin container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_iter(idx, stride, std::begin(t));
    return detail::storage_get(storage_, idx, stride == 0);
  }

  template <typename T>
  const_reference at_impl(detail::static_container_tag, const T& t) const {
    BOOST_ASSERT_MSG(dim() == detail::mp_size<T>::value,
                     "bin container does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin_get(mp11::mp_size<T>(), idx, stride, t);
    return detail::storage_get(storage_, idx, stride == 0);
  }

  template <typename T>
  const_reference at_impl(detail::no_container_tag, const T& t) const {
    BOOST_ASSERT_MSG(dim() == 1,
                     "bin argument does not match histogram dimension");
    std::size_t idx = 0, stride = 1;
    lin<0>(idx, stride, detail::indirect_int_cast(t));
    return detail::storage_get(storage_, idx, stride == 0);
  }

  template <typename Value>
  struct xlin_visitor : public static_visitor<void> {
    std::size_t& idx;
    std::size_t& stride;
    const Value& val;
    xlin_visitor(std::size_t& i, std::size_t& s, const Value& v)
        : idx(i), stride(s), val(v) {}
    template <typename Axis>
    void operator()(const Axis& a) const {
      impl(std::is_convertible<Value, typename Axis::value_type>(), a);
    }

    template <typename Axis>
    void impl(std::true_type, const Axis& a) const {
      const auto a_size = a.size();
      const auto a_shape = a.shape();
      const auto j = a.index(static_cast<typename Axis::value_type>(val));
      detail::lin(idx, stride, a_size, a_shape, j);
    }

    template <typename Axis>
    void impl(std::false_type, const Axis&) const {
      throw std::invalid_argument(
          detail::cat("fill argument not convertible to axis value type: ",
                      boost::typeindex::type_id<Axis>().pretty_name(), ", ",
                      boost::typeindex::type_id<Value>().pretty_name()));
    }
  };

  template <unsigned D>
  void xlin(std::size_t&, std::size_t&) const {}

  template <unsigned D, typename T, typename... Ts>
  void xlin(std::size_t& idx, std::size_t& stride, T&& t, Ts&&... ts) const {
    apply_visitor(xlin_visitor<T>{idx, stride, t}, axes_[D]);
    xlin<(D + 1)>(idx, stride, std::forward<Ts>(ts)...);
  }

  template <typename Iterator>
  void xlin_iter(std::size_t& idx, std::size_t& stride, Iterator iter) const {
    for (const auto& a : axes_) {
      apply_visitor(xlin_visitor<decltype(*iter)>{idx, stride, *iter++}, a);
    }
  }

  template <typename T>
  void xlin_get(mp11::mp_int<0>, std::size_t&, std::size_t&, T&&) const
      noexcept {}

  template <int N, typename T>
  void xlin_get(mp11::mp_int<N>, std::size_t& idx, std::size_t& stride,
                T&& t) const {
    constexpr unsigned D = detail::mp_size<T>::value - N;
    apply_visitor(
        xlin_visitor<detail::mp_at_c<T, D>>{idx, stride, std::get<D>(t)},
        axes_[D]);
    xlin_get(mp11::mp_int<(N - 1)>(), idx, stride, std::forward<T>(t));
  }

  template <unsigned D>
  void lin(std::size_t&, std::size_t&) const noexcept {}

  template <unsigned D, typename... Ts>
  void lin(std::size_t& idx, std::size_t& stride, int j, Ts... ts) const
      noexcept {
    const auto& a = axes_[D];
    const auto a_size = a.size();
    const auto a_shape = a.shape();
    stride *= (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
    detail::lin(idx, stride, a_size, a_shape, j);
    lin<(D + 1)>(idx, stride, ts...);
  }

  template <typename Iterator>
  void lin_iter(std::size_t& idx, std::size_t& stride, Iterator iter) const
      noexcept {
    for (const auto& a : axes_) {
      const auto a_size = a.size();
      const auto a_shape = a.shape();
      const auto j = detail::indirect_int_cast(*iter++);
      stride *=
          (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
      detail::lin(idx, stride, a_size, a_shape, j);
    }
  }

  template <typename T>
  void lin_get(mp11::mp_size_t<0>, std::size_t&, std::size_t&, const T&) const
      noexcept {}

  template <long unsigned int N, typename T>
  void lin_get(mp11::mp_size_t<N>, std::size_t& idx, std::size_t& stride,
               const T& t) const noexcept {
    constexpr long unsigned int D = detail::mp_size<T>::value - N;
    const auto& a = axes_[D];
    const auto a_size = a.size();
    const auto a_shape = a.shape();
    const auto j = detail::indirect_int_cast(std::get<D>(t));
    stride *= (-1 <= j && j <= a_size); // set stride to zero, if j is invalid
    detail::lin(idx, stride, a_size, a_shape, j);
    lin_get(mp11::mp_size_t<(N - 1)>(), idx, stride, t);
  }

  histogram reduce_impl(const std::vector<bool>& b) const {
    axes_type axes;
    std::vector<unsigned> n(b.size());
    auto axes_iter = axes_.begin();
    auto n_iter = n.begin();
    for (const auto& bi : b) {
      if (bi) axes.emplace_back(*axes_iter);
      *n_iter = axes_iter->shape();
      ++axes_iter;
      ++n_iter;
    }
    histogram h(std::move(axes));
    detail::index_mapper m(n, b);
    do { h.storage_.add(m.second, storage_[m.first]); } while (m.next());
    return h;
  }

  template <typename T, typename A, typename S>
  friend class histogram;
  template <typename H>
  friend class iterator_over;
  friend class ::boost::python::access;
  friend class ::boost::serialization::access;
  template <typename Archive>
  void serialize(Archive&, unsigned);
};

template <typename... Axis>
dynamic_histogram<
    mp11::mp_set_push_back<axis::types, detail::rm_cv_ref<Axis>...>>
make_dynamic_histogram(Axis&&... axis) {
  return dynamic_histogram<
      mp11::mp_set_push_back<axis::types, detail::rm_cv_ref<Axis>...>>(
      std::forward<Axis>(axis)...);
}

template <typename Storage, typename... Axis>
dynamic_histogram<
    mp11::mp_set_push_back<axis::types, detail::rm_cv_ref<Axis>...>, Storage>
make_dynamic_histogram_with(Axis&&... axis) {
  return dynamic_histogram<
      mp11::mp_set_push_back<axis::types, detail::rm_cv_ref<Axis>...>,
      Storage>(std::forward<Axis>(axis)...);
}

template <typename Iterator, typename = detail::requires_iterator<Iterator>>
dynamic_histogram<
    detail::mp_set_union<axis::types, typename Iterator::value_type::types>>
make_dynamic_histogram(Iterator begin, Iterator end) {
  return dynamic_histogram<detail::mp_set_union<
      axis::types, typename Iterator::value_type::types>>(begin, end);
}

template <typename Storage, typename Iterator,
          typename = detail::requires_iterator<Iterator>>
dynamic_histogram<
    detail::mp_set_union<axis::types, typename Iterator::value_type::types>,
    Storage>
make_dynamic_histogram_with(Iterator begin, Iterator end) {
  return dynamic_histogram<
      detail::mp_set_union<axis::types, typename Iterator::value_type::types>,
      Storage>(begin, end);
}

} // namespace histogram
} // namespace boost

#endif
