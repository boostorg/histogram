// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_FILL_HPP
#define BOOST_HISTOGRAM_DETAIL_FILL_HPP

#include <algorithm>
#include <boost/config/workaround.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/argument_traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/cell_map.hpp>
#include <boost/histogram/detail/linearize.hpp>
#include <boost/histogram/detail/make_default.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <boost/histogram/detail/priority.hpp>
#include <boost/histogram/detail/tuple_slice.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/integral.hpp>
#include <boost/mp11/tuple.hpp>
#include <boost/mp11/utility.hpp>
#include <cassert>
#include <mutex>
#include <tuple>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <class T, class U>
struct sample_args_passed_vs_expected;

template <class... Passed, class... Expected>
struct sample_args_passed_vs_expected<std::tuple<Passed...>, std::tuple<Expected...>> {
  static_assert(!(sizeof...(Expected) > 0 && sizeof...(Passed) == 0),
                "error: accumulator requires samples, but sample argument is missing");
  static_assert(
      !(sizeof...(Passed) > 0 && sizeof...(Expected) == 0),
      "error: accumulator does not accept samples, but sample argument is passed");
  static_assert(sizeof...(Passed) == sizeof...(Expected),
                "error: numbers of passed and expected sample arguments differ");
  static_assert(
      std::is_convertible<std::tuple<Passed...>, std::tuple<Expected...>>::value,
      "error: sample argument(s) not convertible to accumulator argument(s)");
};

template <class A>
struct storage_grower {
  const A& axes_;
  decltype(make_stack_buffer<axis::index_type>(std::declval<const A&>())) old_extents_;

  storage_grower(const A& axes) noexcept
      : axes_(axes), old_extents_(make_stack_buffer<axis::index_type>(axes)) {}

  void from_shifts(const axis::index_type* shifts) noexcept {
    auto eit = old_extents_.begin();
    for_each_axis(axes_, [&](const auto& a) {
      *eit++ = axis::traits::extent(a) - std::abs(*shifts++);
    });
  }

  // must be extents before any shifts were applied
  void from_extents(const axis::index_type* old_extents) noexcept {
    std::copy(old_extents, old_extents + old_extents_.size(), old_extents_.begin());
  }

  template <class S>
  void apply(S& storage, const axis::index_type* shifts) {
    cell_map<A> map(axes_);
    std::size_t out_stride = 1;
    auto eit = old_extents_.begin();
    for_each_axis(axes_, [&](const auto& a) {
      using opt = axis::traits::get_options<std::decay_t<decltype(a)>>;
      const auto n = axis::traits::extent(a);
      const auto old_extent = *eit++;
      // normal bins move by the positive shift, if any
      const auto shift = (std::max)(*shifts++, 0);
      std::size_t* l = map.add_axis(static_cast<std::size_t>(old_extent));
      for (axis::index_type i = 0; i < old_extent; ++i)
        l[i] = static_cast<std::size_t>(i + shift) * out_stride;
      if (opt::test(axis::option::overflow))
        l[old_extent - 1] = static_cast<std::size_t>(n - 1) * out_stride;
      if (opt::test(axis::option::underflow)) l[0] = 0;
      out_stride *= static_cast<std::size_t>(n);
    });
    assert(map.input_size() == storage.size());

    auto new_storage = make_default(storage);
    new_storage.reset(out_stride);
    map.for_each_line([&](std::size_t in_off, std::size_t out_off, const std::size_t* l,
                          std::size_t n) {
      for (std::size_t p = 0; p < n; ++p)
        new_storage[out_off + l[p]] = storage[in_off + p];
    });
    storage = std::move(new_storage);
  }
};

template <class T, class... Us>
auto fill_storage_element_impl(priority<2>, T&& t, const Us&... args) noexcept
    -> decltype(t(args...), void()) {
  t(args...);
}

template <class T, class U>
auto fill_storage_element_impl(priority<1>, T&& t, const weight_type<U>& w) noexcept
    -> decltype(t += w, void()) {
  t += w;
}

// fallback for arithmetic types and accumulators that do not handle the weight
template <class T, class U>
auto fill_storage_element_impl(priority<0>, T&& t, const weight_type<U>& w) noexcept
    -> decltype(t += w.value, void()) {
  t += w.value;
}

template <class T>
auto fill_storage_element_impl(priority<1>, T&& t) noexcept -> decltype(++t, void()) {
  ++t;
}

template <class T, class... Us>
void fill_storage_element(T&& t, const Us&... args) noexcept {
  fill_storage_element_impl(priority<2>{}, std::forward<T>(t), args...);
}

// t may be a proxy and then it is an rvalue reference, not an lvalue reference
template <class IW, class IS, class T, class U>
void fill_storage_2(IW, IS, T&& t, U&& u) noexcept {
  mp11::tuple_apply(
      [&](const auto&... args) {
        fill_storage_element(std::forward<T>(t), std::get<IW::value>(u), args...);
      },
      std::get<IS::value>(u).value);
}

// t may be a proxy and then it is an rvalue reference, not an lvalue reference
template <class IS, class T, class U>
void fill_storage_2(mp11::mp_int<-1>, IS, T&& t, const U& u) noexcept {
  mp11::tuple_apply(
      [&](const auto&... args) { fill_storage_element(std::forward<T>(t), args...); },
      std::get<IS::value>(u).value);
}

// t may be a proxy and then it is an rvalue reference, not an lvalue reference
template <class IW, class T, class U>
void fill_storage_2(IW, mp11::mp_int<-1>, T&& t, const U& u) noexcept {
  fill_storage_element(std::forward<T>(t), std::get<IW::value>(u));
}

// t may be a proxy and then it is an rvalue reference, not an lvalue reference
template <class T, class U>
void fill_storage_2(mp11::mp_int<-1>, mp11::mp_int<-1>, T&& t, const U&) noexcept {
  fill_storage_element(std::forward<T>(t));
}

template <class IW, class IS, class Storage, class Index, class Args>
auto fill_storage(IW, IS, Storage& s, const Index idx, const Args& a) noexcept {
  if (is_valid(idx)) {
    assert(idx < s.size());
    fill_storage_2(IW{}, IS{}, s[idx], a);
    return s.begin() + idx;
  }
  return s.end();
}

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
constexpr unsigned(min)(const unsigned n) noexcept {
  constexpr unsigned a = buffer_size<A>::value;
  return a < n ? a : n;
}

// not growing
template <class ArgTraits, class Storage, class Axes, class Args>
auto fill_2(ArgTraits, mp11::mp_false, const std::size_t offset, Storage& st,
            const Axes& axes, const Args& args) {
  mp11::mp_if<has_non_inclusive_axis<Axes>, optional_index, std::size_t> idx{offset};
  linearize_args<ArgTraits::start::value, min<Axes>(ArgTraits::nargs::value)>::apply(
      idx, axes, args);
  return fill_storage(typename ArgTraits::wpos{}, typename ArgTraits::spos{}, st, idx,
                      args);
}

// at least one axis is growing
template <class ArgTraits, class Storage, class Axes, class Args>
auto fill_2(ArgTraits, mp11::mp_true, const std::size_t, Storage& st, Axes& axes,
            const Args& args) {
  std::array<axis::index_type, ArgTraits::nargs::value> shifts;
  // offset must be zero for linearize_growth (value of offset argument is ignored)
  mp11::mp_if<has_non_inclusive_axis<Axes>, optional_index, std::size_t> idx{0};
  std::size_t stride = 1;
  bool update_needed = false;
  mp11::mp_for_each<mp11::mp_iota_c<min<Axes>(ArgTraits::nargs::value)>>([&](auto i) {
    auto& ax = axis_get<i>(axes);
    const auto extent = linearize_growth(idx, shifts[i], stride, ax,
                                         std::get<(ArgTraits::start::value + i)>(args));
    update_needed |= shifts[i] != 0;
    stride *= extent;
  });
  if (update_needed) {
    storage_grower<Axes> g(axes);
    g.from_shifts(shifts.data());
    g.apply(st, shifts.data());
  }
  return fill_storage(typename ArgTraits::wpos{}, typename ArgTraits::spos{}, st, idx,
                      args);
}

// pack original args tuple into another tuple (which is unpacked later)
template <int Start, int Size, class IW, class IS, class Args>
decltype(auto) pack_args(IW, IS, const Args& args) noexcept {
  return std::make_tuple(tuple_slice<Start, Size>(args), std::get<IW::value>(args),
                         std::get<IS::value>(args));
}

template <int Start, int Size, class IW, class Args>
decltype(auto) pack_args(IW, mp11::mp_int<-1>, const Args& args) noexcept {
  return std::make_tuple(tuple_slice<Start, Size>(args), std::get<IW::value>(args));
}

template <int Start, int Size, class IS, class Args>
decltype(auto) pack_args(mp11::mp_int<-1>, IS, const Args& args) noexcept {
  return std::make_tuple(tuple_slice<Start, Size>(args), std::get<IS::value>(args));
}

template <int Start, int Size, class Args>
decltype(auto) pack_args(mp11::mp_int<-1>, mp11::mp_int<-1>, const Args& args) noexcept {
  return std::make_tuple(args);
}

#if BOOST_WORKAROUND(BOOST_MSVC, >= 0)
#pragma warning(disable : 4702) // fixing warning would reduce code readability a lot
#endif

template <class ArgTraits, class S, class A, class Args>
auto fill(std::true_type, ArgTraits, const std::size_t offset, S& storage, A& axes,
          const Args& args) -> typename S::iterator {
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

  if (axes_rank(axes) == ArgTraits::nargs::value)
    return fill_2(ArgTraits{}, growing{}, offset, storage, axes, args);
  else if (axes_rank(axes) == 1 &&
           axis::traits::rank(axis_get<0>(axes)) == ArgTraits::nargs::value)
    return fill_2(
        argument_traits_holder<
            1, 0, (ArgTraits::wpos::value >= 0 ? 1 : -1),
            (ArgTraits::spos::value >= 0 ? (ArgTraits::wpos::value >= 0 ? 2 : 1) : -1),
            typename ArgTraits::sargs>{},
        growing{}, offset, storage, axes,
        pack_args<ArgTraits::start::value, ArgTraits::nargs::value>(
            typename ArgTraits::wpos{}, typename ArgTraits::spos{}, args));
  return BOOST_THROW_EXCEPTION(
             std::invalid_argument("number of arguments != histogram rank")),
         storage.end();
}

#if BOOST_WORKAROUND(BOOST_MSVC, >= 0)
#pragma warning(default : 4702)
#endif

// empty implementation for bad arguments to stop compiler from showing internals
template <class ArgTraits, class S, class A, class Args>
auto fill(std::false_type, ArgTraits, const std::size_t, S& storage, A&, const Args&) ->
    typename S::iterator {
  return storage.end();
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
