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
#include <boost/mp11.hpp>
#include <boost/throw_exception.hpp>
#include <boost/variant2/variant.hpp>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

namespace dtl = boost::histogram::detail;

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
  });

  // offset must be zero for growing axes
  using IsGrowing = has_growing_axis<Axes>;
  std::fill(indices, indices + size, IsGrowing::value ? 0 : offset);
  for_each_axis(axes, [&, stride = static_cast<std::size_t>(1),
                       pshift = shifts](auto& axis) mutable {
    using Axis = std::decay_t<decltype(axis)>;
    static_if<is_variant<T>>( // LCOV_EXCL_LINE buggy in gcc-8, works in gcc-5
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

template <class... Ts>
void fold(Ts...) noexcept {} // helper to enable operator folding

template <class... Us>
void increment_pointers(const Us*&&... ptrs) noexcept {
  fold(++ptrs...);
}

template <class T, class... Us>
void increment_pointers(std::size_t wsize, const T*&& wptr,
                        const Us*&&... sptrs) noexcept {
  fold(++sptrs...);
  if (wsize > 1) ++wptr;
}

template <class S, class... Us>
void fill_n_storage_2(S& s, const std::size_t& idx, std::size_t,
                      const Us*&&... ptrs) noexcept {
  fill_storage_3(s[idx], *ptrs...);
}

template <class S, class... Ts>
void fill_n_storage_2(S& s, const std::size_t& idx, const Ts*&&... ptrs) noexcept {
  fill_storage_3(s[idx], *ptrs...);
}

template <class S, class Index, class... Ts>
void fill_n_storage(S& s, const Index idx, Ts&&... ts) noexcept {
  if (is_valid(idx)) {
    BOOST_ASSERT(idx < s.size());
    fill_n_storage_2(s, idx, std::forward<Ts>(ts)...);
  }
  increment_pointers(std::forward<Ts>(ts)...);
}

template <class T>
std::size_t get_total_size(const T* values, std::size_t vsize) {
  std::size_t s = 1u;
  auto vis = [&s](const auto& v) {
    // cannot be replaced by std::decay_t
    using U = std::remove_cv_t<std::remove_reference_t<decltype(v)>>;
    const std::size_t n = static_if<is_iterable<U>>(
        [](const auto& v) { return dtl::size(v); },
        [](const auto&) { return static_cast<std::size_t>(1); }, v);
    if (s == 1u)
      s = n;
    else if (n > 1u && s != n)
      throw_exception(std::invalid_argument("spans must have same length"));
  };
  for (const auto& v : make_span(values, vsize))
    static_if<is_iterable<T>>([&vis](const auto& v) { vis(v); },
                              [&vis](const auto& v) { variant2::visit(vis, v); }, v);
  return s;
}

// general Nd treatment
template <class S, class A, class T, class... Ts>
void fill_n_nd(const std::size_t offset, S& storage, A& axes, const std::size_t vsize,
               const T* values, Ts&&... rest) {
  constexpr std::size_t buffer_size = 1ul << 14;
  using index_type = mp11::mp_if<has_non_inclusive_axis<A>, optional_index, std::size_t>;
  index_type indices[buffer_size];

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
      fill_n_storage(storage, idx, std::forward<Ts>(rest)...);
  }
}

template <class S, class A, class T, class... Us, class L = mp11::mp_list<Us...>>
std::enable_if_t<(
    mp11::mp_count_if<L, is_weight>::value + mp11::mp_count_if<L, is_sample>::value == 0)>
fill_n_1(const std::size_t offset, S& storage, A& axes, const std::size_t vsize,
         const T* values, Us&&... rest) {
  fill_n_nd(offset, storage, axes, vsize, values, std::forward<Us>(rest)...);
}

// unpack weight argument, can be iterable or value
template <class S, class A, class T, class U, class... Us>
void fill_n_1(const std::size_t offset, S& storage, A& axes, const std::size_t vsize,
              const T* values, const weight_type<U>& weights, const Us&... rest) {
  static_if<is_iterable<std::remove_cv_t<std::remove_reference_t<U>>>>(
      [&](const auto& w, const auto&... rest) {
        const auto wsize = dtl::size(w);
        if (vsize != wsize)
          throw_exception(
              std::invalid_argument("number of arguments must match histogram rank"));
        fill_n_1(offset, storage, axes, vsize, values, wsize, dtl::data(w), rest...);
      },
      [&](const auto w, const auto&... rest) {
        fill_n_1(offset, storage, axes, vsize, values, static_cast<std::size_t>(1), &w,
                 rest...);
      },
      weights.value, rest...);
}

// unpack sample argument after weight was unpacked
template <class S, class A, class T, class U, class V>
void fill_n_1(std::size_t offset, S& storage, A& axes, const std::size_t vsize,
              const T* values, const std::size_t wsize, const U* wptr,
              const sample_type<V>& s) {
  mp11::tuple_apply(
      [&](const auto&... sargs) {
        fill_n_1(offset, storage, axes, vsize, values, wsize, std::move(wptr),
                 dtl::data(sargs)...);
      },
      s.value);
}

// unpack sample argument (no weight argument)
template <class S, class A, class T, class U>
void fill_n_1(const std::size_t offset, S& storage, A& axes, const std::size_t vsize,
              const T* values, const sample_type<U>& samples) {
  using namespace boost::mp11;
  tuple_apply(
      [&](const auto&... sargs) {
        fill_n_1(offset, storage, axes, vsize, values, dtl::data(sargs)...);
      },
      samples.value);
}

template <class S, class A, class T, class... Us>
void fill_n(const std::size_t offset, S& storage, A& axes, const T* values,
            const std::size_t vsize, const Us&... rest) {
  static_assert(!std::is_pointer<T>::value,
                "passing iterable of pointers not allowed (cannot determine lengths); "
                "pass iterable of iterables instead");
  using namespace boost::mp11;
  static_if<mp_or<is_iterable<T>, is_variant<T>>>(
      [&](const auto& values) {
        if (axes_rank(axes) != vsize)
          throw_exception(
              std::invalid_argument("number of arguments must match histogram rank"));
        fill_n_1(offset, storage, axes, get_total_size(values, vsize), values, rest...);
      },
      [&](const auto& values) {
        auto s = make_span(values, vsize);
        fill_n_1(offset, storage, axes, vsize, &s, rest...);
      },
      values);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif // BOOST_HISTOGRAM_DETAIL_FILL_N_HPP
