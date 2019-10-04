// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_FILL_HPP
#define BOOST_HISTOGRAM_DETAIL_FILL_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config/workaround.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/linearize.hpp>
#include <boost/histogram/detail/make_default.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <boost/histogram/detail/tuple_slice.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11.hpp>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <class A>
struct storage_grower {
  const A& axes_;
  struct {
    axis::index_type idx, old_extent;
    std::size_t new_stride;
  } data_[buffer_size<A>::value];
  std::size_t new_size_;

  storage_grower(const A& axes) noexcept : axes_(axes) {}

  void from_shifts(const axis::index_type* shifts) noexcept {
    auto dit = data_;
    std::size_t s = 1;
    for_each_axis(axes_, [&](const auto& a) {
      const auto n = axis::traits::extent(a);
      *dit++ = {0, n - std::abs(*shifts++), s};
      s *= n;
    });
    new_size_ = s;
  }

  // must be extents before any shifts were applied
  void from_extents(const axis::index_type* old_extents) noexcept {
    auto dit = data_;
    std::size_t s = 1;
    for_each_axis(axes_, [&](const auto& a) {
      const auto n = axis::traits::extent(a);
      *dit++ = {0, *old_extents++, s};
      s *= n;
    });
    new_size_ = s;
  }

  template <class S>
  void apply(S& storage, const axis::index_type* shifts) {
    auto new_storage = make_default(storage);
    new_storage.reset(new_size_);
    const auto dlast = data_ + axes_rank(axes_) - 1;
    for (const auto& x : storage) {
      auto ns = new_storage.begin();
      auto sit = shifts;
      auto dit = data_;
      for_each_axis(axes_, [&](const auto& a) {
        using opt = axis::traits::static_options<std::decay_t<decltype(a)>>;
        if (opt::test(axis::option::underflow)) {
          if (dit->idx == 0) {
            // axis has underflow and we are in the underflow bin:
            // keep storage pointer unchanged
            ++dit;
            ++sit;
            return;
          }
        }
        if (opt::test(axis::option::overflow)) {
          if (dit->idx == dit->old_extent - 1) {
            // axis has overflow and we are in the overflow bin:
            // move storage pointer to corresponding overflow bin position
            ns += (axis::traits::extent(a) - 1) * dit->new_stride;
            ++dit;
            ++sit;
            return;
          }
        }
        // we are in a normal bin:
        // move storage pointer to index position, apply positive shifts
        ns += (dit->idx + std::max(*sit, 0)) * dit->new_stride;
        ++dit;
        ++sit;
      });
      // assign old value to new location
      *ns = x;
      // advance multi-dimensional index
      dit = data_;
      ++dit->idx;
      while (dit != dlast && dit->idx == dit->old_extent) {
        dit->idx = 0;
        ++(++dit)->idx;
      }
    }
    storage = std::move(new_storage);
  }
};

template <class T, class... Us>
void fill_storage_4(mp11::mp_false, T&& t, Us&&... args) noexcept {
  t(std::forward<Us>(args)...);
}

template <class T>
void fill_storage_4(mp11::mp_true, T&& t) noexcept {
  ++t;
}

template <class T, class U>
void fill_storage_4(mp11::mp_true, T&& t, U&& w) noexcept {
  t += w;
}

template <class T, class... Us>
void fill_storage_3(T&& t, Us&&... args) noexcept {
  fill_storage_4(has_operator_preincrement<std::decay_t<T>>{}, std::forward<T>(t),
                 std::forward<Us>(args)...);
}

template <class IW, class IS, class T, class U>
void fill_storage_2(IW, IS, T&& t, U&& u) noexcept {
  mp11::tuple_apply(
      [&](auto&&... args) {
        fill_storage_3(std::forward<T>(t), std::get<IW::value>(u).value, args...);
      },
      std::get<IS::value>(u).value);
}

template <class IS, class T, class U>
void fill_storage_2(mp11::mp_int<-1>, IS, T&& t, const U& u) noexcept {
  mp11::tuple_apply(
      [&](const auto&... args) { fill_storage_3(std::forward<T>(t), args...); },
      std::get<IS::value>(u).value);
}

template <class IW, class T, class U>
void fill_storage_2(IW, mp11::mp_int<-1>, T&& t, const U& u) noexcept {
  fill_storage_3(std::forward<T>(t), std::get<IW::value>(u).value);
}

template <class T, class U>
void fill_storage_2(mp11::mp_int<-1>, mp11::mp_int<-1>, T&& t, const U&) noexcept {
  fill_storage_3(std::forward<T>(t));
}

template <class IW, class IS, class Storage, class Index, class Args>
auto fill_storage(IW, IS, Storage& s, const Index idx, const Args& a) noexcept {
  if (is_valid(idx)) {
    BOOST_ASSERT(idx < s.size());
    fill_storage_2(IW{}, IS{}, s[idx], a);
    return s.begin() + idx;
  }
  return s.end();
}

template <class L>
struct args_indices {
  static constexpr int _size = static_cast<int>(mp11::mp_size<L>::value);
  static constexpr int _weight = static_cast<int>(mp11::mp_find_if<L, is_weight>::value);
  static constexpr int _sample = static_cast<int>(mp11::mp_find_if<L, is_sample>::value);

  static constexpr unsigned nargs = _size - (_weight < _size) - (_sample < _size);
  static constexpr int start =
      _weight < _size && _sample < _size && (_weight + _sample < 2)
          ? 2
          : ((_weight == 0 || _sample == 0) ? 1 : 0);
  using weight = mp11::mp_int<(_weight < _size ? _weight : -1)>;
  using sample = mp11::mp_int<(_sample < _size ? _sample : -1)>;
};

template <int S, int N>
struct linearize_args {
  template <class Index, class A, class Args>
  static void impl(mp11::mp_int<N>, Index&, const std::size_t, A&, const Args&) {}

  template <int I, class Index, class A, class Args>
  static void impl(mp11::mp_int<I>, Index& o, const std::size_t s, A& ax,
                   const Args& args) {
    const auto e = linearize(o, s, axis_get<I>(ax), std::get<(S + I)>(args));
    impl(mp11::mp_int<(I + 1)>{}, o, s * e, ax, args);
  }

  template <class Index, class A, class Args>
  static void apply(Index& o, A& ax, const Args& args) {
    impl(mp11::mp_int<0>{}, o, 1, ax, args);
  }
};

template <int S>
struct linearize_args<S, 1> {
  template <class Index, class A, class Args>
  static void apply(Index& o, A& ax, const Args& args) {
    linearize(o, 1, axis_get<0>(ax), std::get<S>(args));
  }
};

template <class A>
constexpr unsigned min(const unsigned n) noexcept {
  constexpr unsigned a = static_cast<unsigned>(buffer_size<A>::value);
  return a < n ? a : n;
}

// not growing
template <class Storage, class Axes, class Args>
auto fill_2(mp11::mp_false, const std::size_t offset, Storage& st, const Axes& axes,
            const Args& args) {
  using pos = args_indices<mp11::mp_transform<std::decay_t, Args>>;
  mp11::mp_if<has_non_inclusive_axis<Axes>, optional_index, std::size_t> idx{offset};
  linearize_args<pos::start, min<Axes>(pos::nargs)>::apply(idx, axes, args);
  return fill_storage(typename pos::weight{}, typename pos::sample{}, st, idx, args);
}

// at least one axis is growing
template <class Storage, class A, class Args>
auto fill_2(mp11::mp_true, const std::size_t, Storage& st, A& axes, const Args& args) {
  using pos = args_indices<mp11::mp_transform<std::decay_t, Args>>;
  std::array<axis::index_type, pos::nargs> shifts;
  // offset must be zero for linearize_growth
  mp11::mp_if<has_non_inclusive_axis<A>, optional_index, std::size_t> idx{0};
  std::size_t stride = 1;
  bool update_needed = false;
  mp11::mp_for_each<mp11::mp_iota_c<min<A>(pos::nargs)>>([&](auto i) {
    auto& ax = axis_get<i>(axes);
    const auto extent =
        linearize_growth(idx, shifts[i], stride, ax, std::get<(pos::start + i)>(args));
    update_needed |= shifts[i] != 0;
    stride *= extent;
  });
  if (update_needed) {
    storage_grower<A> g(axes);
    g.from_shifts(shifts.data());
    g.apply(st, shifts.data());
  }
  return fill_storage(typename pos::weight{}, typename pos::sample{}, st, idx, args);
}

// pack original args tuple into another tuple (which is unpacked later)
template <int Start, int Size, class IW, class IS, class Args>
decltype(auto) pack_args(IW, IS, const Args& args) noexcept {
  return std::make_tuple(std::get<IW::value>(args), std::get<IS::value>(args),
                         tuple_slice<Start, Size>(args));
}

template <int Start, int Size, class IW, class Args>
decltype(auto) pack_args(IW, mp11::mp_int<-1>, const Args& args) noexcept {
  return std::make_tuple(std::get<IW::value>(args), tuple_slice<Start, Size>(args));
}

template <int Start, int Size, class IS, class Args>
decltype(auto) pack_args(mp11::mp_int<-1>, IS, const Args& args) noexcept {
  return std::make_tuple(std::get<IS::value>(args), tuple_slice<Start, Size>(args));
}

template <int Start, int Size, class Args>
decltype(auto) pack_args(mp11::mp_int<-1>, mp11::mp_int<-1>, const Args& args) noexcept {
  return std::make_tuple(args);
}

#if BOOST_WORKAROUND(BOOST_MSVC, >= 0)
#pragma warning(disable : 4702) // fixing warning would reduce code readability a lot
#endif

template <class S, class A, class Args>
auto fill(const std::size_t offset, S& storage, A& axes, const Args& args) {
  using pos = args_indices<mp11::mp_transform<std::decay_t, Args>>;
  using growing = has_growing_axis<A>;

  // Sometimes we need to pack the tuple into another tuple:
  // - histogram contains one axis which accepts tuple
  // - user passes tuple to fill(...)
  // Tuple is normally unpacked and arguments are processed, this causes pos::nargs > 1.
  // Now we pack tuple into another tuple so that original tuple is send to axis.
  // Notes:
  // - has nice side-effect of making histogram::operator(1, 2) work as well
  // - cannot detect call signature of axis at compile-time in all configurations
  //   (axis::variant provides generic call interface and hides concrete
  //   interface), so we throw at runtime if incompatible argument is passed (e.g.
  //   3d tuple)

  if (axes_rank(axes) == pos::nargs)
    return fill_2(growing{}, offset, storage, axes, args);
  else if (axes_rank(axes) == 1 && axis::traits::rank(axis_get<0>(axes)) == pos::nargs)
    return fill_2(growing{}, offset, storage, axes,
                  pack_args<pos::start, pos::nargs>(typename pos::weight{},
                                                    typename pos::sample{}, args));
  else
    return (BOOST_THROW_EXCEPTION(
                std::invalid_argument("number of arguments != histogram rank")),
            storage.end());
}

#if BOOST_WORKAROUND(BOOST_MSVC, >= 0)
#pragma warning(default : 4702)
#endif

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
