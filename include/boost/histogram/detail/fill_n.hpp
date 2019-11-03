// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_FILL_N_HPP
#define BOOST_HISTOGRAM_DETAIL_FILL_N_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/detect.hpp>
#include <boost/histogram/detail/fill.hpp>
#include <boost/histogram/detail/linearize.hpp>
#include <boost/histogram/detail/non_member_container_access.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <boost/histogram/detail/span.hpp>
#include <boost/histogram/detail/static_if.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/bind.hpp>
#include <boost/mp11/utility.hpp>
#include <boost/throw_exception.hpp>
#include <boost/variant2/variant.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace boost {
namespace histogram {
namespace detail {

namespace dtl = boost::histogram::detail;

template <class... Ts>
void fold(Ts&&...) noexcept {} // helper to enable operator folding

template <class T>
auto to_ptr_size(const T& x) {
  return static_if<std::is_scalar<T>>(
      [](const auto& x) { return std::make_pair(&x, static_cast<std::size_t>(1)); },
      [](const auto& x) { return std::make_pair(dtl::data(x), dtl::size(x)); }, x);
}

template <class Index, class Axis, class IsGrowing>
struct index_visitor {
  using index_type = Index;
  using pointer = index_type*;

  using Opt = axis::traits::static_options<Axis>;

  Axis& axis_;
  const std::size_t stride_, start_, size_; // start and size of value collection
  const pointer begin_;
  axis::index_type* shift_;

  index_visitor(Axis& a, std::size_t& str, const std::size_t& sta, const std::size_t& si,
                const pointer it, axis::index_type* shift)
      : axis_(a), stride_(str), start_(sta), size_(si), begin_(it), shift_(shift) {}

  template <class T>
  void call_2(std::true_type, pointer it, const T& x) const {
    // must use this code for all axes if one of them is growing
    axis::index_type shift;
    linearize_growth(*it, shift, stride_, axis_, x);
    if (shift > 0) { // shift previous indices, because axis zero-point has changed
      while (it != begin_) *--it += static_cast<std::size_t>(shift) * stride_;
      *shift_ += shift;
    }
  }

  template <class T>
  void call_2(std::false_type, pointer it, const T& x) const {
    // no axis is growing
    linearize(*it, stride_, axis_, x);
  }

  template <class T>
  void call_1(std::true_type, const T& iterable) const {
    // T is iterable, fill N values
    auto* tp = dtl::data(iterable) + start_;
    for (auto it = begin_; it != begin_ + size_; ++it) call_2(IsGrowing{}, it, *tp++);
  }

  template <class T>
  void call_1(std::false_type, const T& value) const {
    // T is value, fill single value N times
    index_type idx{*begin_};
    call_2(IsGrowing{}, &idx, value);
    if (is_valid(idx)) {
      const auto delta =
          static_cast<std::intptr_t>(idx) - static_cast<std::intptr_t>(*begin_);
      for (auto&& i : make_span(begin_, size_)) i += delta;
    } else
      std::fill(begin_, begin_ + size_, invalid_index);
  }

  template <class T>
  void operator()(const T& iterable_or_value) const {
    call_1(is_iterable<T>{}, iterable_or_value);
  }
};

template <class Index, class S, class Axes, class T>
void fill_n_indices(Index* indices, const std::size_t start, const std::size_t size,
                    const std::size_t offset, S& storage, Axes& axes, const T* viter) {
  axis::index_type extents[buffer_size<Axes>::value];
  axis::index_type shifts[buffer_size<Axes>::value];
  for_each_axis(axes, [eit = extents, sit = shifts](const auto& a) mutable {
    *sit++ = 0;
    *eit++ = axis::traits::extent(a);
  }); // LCOV_EXCL_LINE: gcc-8 is missing this line for no reason

  // offset must be zero for growing axes
  using IsGrowing = has_growing_axis<Axes>;
  std::fill(indices, indices + size, IsGrowing::value ? 0 : offset);
  for_each_axis(axes, [&, stride = static_cast<std::size_t>(1),
                       pshift = shifts](auto& axis) mutable {
    using Axis = std::decay_t<decltype(axis)>;
    static_if<is_variant<T>>( // LCOV_EXCL_LINE: gcc-8 is missing this line for no reason
        [&](const auto& v) {
          variant2::visit(index_visitor<Index, Axis, IsGrowing>{axis, stride, start, size,
                                                                indices, pshift},
                          v);
        },
        [&](const auto& v) {
          index_visitor<Index, Axis, IsGrowing>{axis, stride,  start,
                                                size, indices, pshift}(v);
        },
        *viter++);
    stride *= static_cast<std::size_t>(axis::traits::extent(axis));
    ++pshift;
  });

  bool update_needed = false;
  for_each_axis(axes, [&update_needed, eit = extents](const auto& a) mutable {
    update_needed |= *eit++ != axis::traits::extent(a);
  });
  if (update_needed) {
    storage_grower<Axes> g(axes);
    g.from_extents(extents);
    g.apply(storage, shifts);
  }
}

template <class S, class Index, class... Ts>
void fill_n_storage(S& s, const Index idx, Ts&&... p) noexcept {
  if (is_valid(idx)) {
    BOOST_ASSERT(idx < s.size());
    fill_storage_3(s[idx], *p.first...);
  }
  fold((p.second > 1 ? ++p.first : 0)...);
}

template <class S, class Index, class T, class... Ts>
void fill_n_storage(S& s, const Index idx, weight_type<T>&& w, Ts&&... ps) noexcept {
  if (is_valid(idx)) {
    BOOST_ASSERT(idx < s.size());
    fill_storage_3(s[idx], weight_type<decltype(*w.value.first)>{*w.value.first},
                   *ps.first...);
  }
  if (w.value.second > 1) ++w.value.first;
  fold((ps.second > 1 ? ++ps.first : 0)...);
}

// general Nd treatment
template <class Index, class S, class A, class T, class... Ts>
void fill_n_nd(const std::size_t offset, S& storage, A& axes, const std::size_t vsize,
               const T* values, Ts&&... ts) {
  constexpr std::size_t buffer_size = 1ul << 14;
  Index indices[buffer_size];

  /*
    Parallelization options.

    A) Run the whole fill2 method in parallel, each thread fills its own buffer of
    indices, synchronization (atomics) are needed to synchronize the incrementing of
    the storage cells. This leads to a lot of congestion for small histograms.

    B) Run only fill_n_indices in parallel, subsections of the indices buffer
    can be filled by different threads. The final loop that fills the storage runs
    in the main thread, this requires no synchronization for the storage, cells do
    not need to support atomic operations.

    C) Like B), then sort the indices in the main thread and fill the
    storage in parallel, where each thread uses a disjunct set of indices. This
    should create less congestion and requires no synchronization for the storage.

    Note on C): Let's say we have an axis with 5 bins (with *flow to simplify).
    Then after filling 10 values, converting to indices and sorting, the index
    buffer may look like this: 0 0 0 1 2 2 2 4 4 5. Let's use two threads to fill
    the storage. Still in the main thread, we compute an iterator to the middle of
    the index buffer and move it to the right until the pointee changes. Now we have
    two ranges which contain disjunct sets of indices. We pass these ranges to the
    threads which then fill the storage. Since the threads by construction do not
    compete to increment the same cell, no further synchronization is required.

    In all cases, growing axes cannot be parallelized.
  */

  for (std::size_t start = 0; start < vsize; start += buffer_size) {
    const std::size_t n = std::min(buffer_size, vsize - start);
    // fill buffer of indices...
    fill_n_indices(indices, start, n, offset, storage, axes, values);
    // ...and fill corresponding storage cells
    for (auto&& idx : make_span(indices, n))
      fill_n_storage(storage, idx, std::forward<Ts>(ts)...);
  }
}

template <class S, class... As, class T, class... Us>
void fill_n_1(const std::size_t offset, S& storage, std::tuple<As...>& axes,
              const std::size_t vsize, const T* values, Us&&... us) {
  using index_type =
      mp11::mp_if<has_non_inclusive_axis<std::tuple<As...>>, optional_index, std::size_t>;
  fill_n_nd<index_type>(offset, storage, axes, vsize, values, std::forward<Us>(us)...);
}

template <class S, class A, class T, class... Us>
void fill_n_1(const std::size_t offset, S& storage, A& axes, const std::size_t vsize,
              const T* values, Us&&... us) {
  bool all_inclusive = true;
  for_each_axis(axes,
                [&](const auto& ax) { all_inclusive &= axis::traits::inclusive(ax); });
  if (axes_rank(axes) == 1) {
    axis::visit(
        [&](auto& ax) {
          std::tuple<decltype(ax)> axes{ax};
          fill_n_1(offset, storage, axes, vsize, values, std::forward<Us>(us)...);
        },
        axes[0]);
  } else {
    if (all_inclusive)
      fill_n_nd<std::size_t>(offset, storage, axes, vsize, values,
                             std::forward<Us>(us)...);
    else
      fill_n_nd<optional_index>(offset, storage, axes, vsize, values,
                                std::forward<Us>(us)...);
  }
}

template <class T, std::size_t N>
std::size_t get_total_size(const dtl::span<const T, N>& values) {
  std::size_t s = 1u;
  auto vis = [&s](const auto& v) {
    // cannot be replaced by std::decay_t
    using U = std::remove_cv_t<std::remove_reference_t<decltype(v)>>;
    const std::size_t n = static_if<is_iterable<U>>(
        [](const auto& v) { return dtl::size(v); },
        [](const auto&) { return static_cast<std::size_t>(1); }, v);
    if (s != 1u && n != 1u && s != n)
      BOOST_THROW_EXCEPTION(std::invalid_argument("spans must have compatible lengths"));
    s = std::max(s, n);
  };
  for (const auto& v : values)
    static_if<is_iterable<T>>([&vis](const auto& v) { vis(v); },
                              [&vis](const auto& v) { variant2::visit(vis, v); }, v);
  return s;
}

template <class... Ts>
void fill_n_check_extra_args(std::size_t n, Ts&&... ts) {
  // values of length 1 may not be combined with weights and samples of length > 1
  auto check = [n](auto&& x) {
    if (x.second != 1 && n != x.second)
      BOOST_THROW_EXCEPTION(std::invalid_argument("spans must have compatible lengths"));
    return 0;
  };
  fold(check(ts)...);
}

template <class T, class... Ts>
void fill_n_check_extra_args(std::size_t n, weight_type<T>&& w, Ts&&... ts) {
  fill_n_check_extra_args(n, w.value, std::forward<Ts>(ts)...);
}

inline void fill_n_check_extra_args(std::size_t) noexcept {}

template <class S, class A, class T, std::size_t N, class... Us>
void fill_n(std::true_type, const std::size_t offset, S& storage, A& axes,
            const dtl::span<const T, N> values, Us&&... us) {
  static_assert(!std::is_pointer<T>::value,
                "passing iterable of pointers not allowed (cannot determine lengths); "
                "pass iterable of iterables instead");
  using Vs = value_types<A>;
  static_if<mp11::mp_any_of_q<Vs, mp11::mp_bind_front<std::is_convertible, T>>>(
      [&](const auto& values, auto&&... us) {
        if (axes_rank(axes) != 1)
          BOOST_THROW_EXCEPTION(
              std::invalid_argument("number of arguments must match histogram rank"));
        fill_n_check_extra_args(values.size(), std::forward<Us>(us)...);
        fill_n_1(offset, storage, axes, values.size(), &values, std::forward<Us>(us)...);
      },
      [&](const auto& values, auto&&... us) {
        if (axes_rank(axes) != values.size())
          BOOST_THROW_EXCEPTION(
              std::invalid_argument("number of arguments must match histogram rank"));
        const auto vsize = get_total_size(values);
        fill_n_check_extra_args(vsize, std::forward<Us>(us)...);
        fill_n_1(offset, storage, axes, vsize, values.data(), std::forward<Us>(us)...);
      },
      values, std::forward<Us>(us)...);
}

// empty implementation for bad arguments to stop compiler from showing internals
template <class... Ts>
void fill_n(std::false_type, Ts...) {}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif // BOOST_HISTOGRAM_DETAIL_FILL_N_HPP
