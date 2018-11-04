// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_AXES_HPP
#define BOOST_HISTOGRAM_DETAIL_AXES_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/weight.hpp>
#include <boost/mp11.hpp>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

template <int N, typename... Ts>
decltype(auto) axis_get(std::tuple<Ts...>& axes) {
  return std::get<N>(axes);
}

template <int N, typename... Ts>
decltype(auto) axis_get(const std::tuple<Ts...>& axes) {
  return std::get<N>(axes);
}

template <int N, typename T>
decltype(auto) axis_get(T& axes) {
  return axes[N];
}

template <int N, typename T>
decltype(auto) axis_get(const T& axes) {
  return axes[N];
}

template <typename... Ts>
decltype(auto) axis_get(std::tuple<Ts...>& axes, std::size_t i) {
  return mp11::mp_with_index<sizeof...(Ts)>(
      i, [&](auto I) { return axis::variant<Ts&...>(std::get<I>(axes)); });
}

template <typename... Ts>
decltype(auto) axis_get(const std::tuple<Ts...>& axes, std::size_t i) {
  return mp11::mp_with_index<sizeof...(Ts)>(
      i, [&](auto I) { return axis::variant<const Ts&...>(std::get<I>(axes)); });
}

template <typename T>
decltype(auto) axis_get(T& axes, std::size_t i) {
  return axes.at(i);
}

template <typename T>
decltype(auto) axis_get(const T& axes, std::size_t i) {
  return axes.at(i);
}

template <typename... Ts, typename... Us>
bool axes_equal(const std::tuple<Ts...>& t, const std::tuple<Us...>& u) {
  return static_if<std::is_same<mp11::mp_list<Ts...>, mp11::mp_list<Us...>>>(
      [](const auto& a, const auto& b) { return a == b; },
      [](const auto&, const auto&) { return false; }, t, u);
}

template <typename... Ts, typename U>
bool axes_equal(const std::tuple<Ts...>& t, const U& u) {
  if (sizeof...(Ts) != u.size()) return false;
  bool equal = true;
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>([&](auto I) {
    using T = mp11::mp_at<std::tuple<Ts...>, decltype(I)>;
    auto up = axis::get<T>(&u[I]);
    equal &= (up && std::get<I>(t) == *up);
  });
  return equal;
}

template <typename T, typename... Us>
bool axes_equal(const T& t, const std::tuple<Us...>& u) {
  return axes_equal(u, t);
}

template <typename T, typename U>
bool axes_equal(const T& t, const U& u) {
  if (t.size() != u.size()) return false;
  return std::equal(t.begin(), t.end(), u.begin());
}

template <typename... Ts, typename... Us>
void axes_assign(std::tuple<Ts...>& t, const std::tuple<Us...>& u) {
  static_if<std::is_same<mp11::mp_list<Ts...>, mp11::mp_list<Us...>>>(
      [](auto& a, const auto& b) { a = b; },
      [](auto&, const auto&) {
        throw std::invalid_argument("cannot assign axes, types do not match");
      },
      t, u);
}

template <typename... Ts, typename U>
void axes_assign(std::tuple<Ts...>& t, const U& u) {
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>([&](auto I) {
    using T = mp11::mp_at_c<std::tuple<Ts...>, I>;
    std::get<I>(t) = axis::get<T>(u[I]);
  });
}

template <typename T, typename... Us>
void axes_assign(T& t, const std::tuple<Us...>& u) {
  t.resize(sizeof...(Us));
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Us)>>(
      [&](auto I) { t[I] = std::get<I>(u); });
}

template <typename T, typename U>
void axes_assign(T& t, const U& u) {
  t.assign(u.begin(), u.end());
}

template <typename... Ts>
constexpr std::size_t axes_size(const std::tuple<Ts...>&) {
  return sizeof...(Ts);
}

template <typename T>
std::size_t axes_size(const T& axes) {
  return axes.size();
}

template <int N, typename... Ts>
void range_check(const std::tuple<Ts...>&) {
  static_assert(N < sizeof...(Ts), "index out of range");
}

template <int N, typename T>
void range_check(const T& axes) {
  BOOST_ASSERT_MSG(N < axes.size(), "index out of range");
}

template <typename F, typename... Ts>
void for_each_axis(const std::tuple<Ts...>& axes, F&& f) {
  mp11::tuple_for_each(axes, std::forward<F>(f));
}

template <typename F, typename T>
void for_each_axis(const T& axes, F&& f) {
  for (const auto& x : axes) { axis::visit(std::forward<F>(f), x); }
}

template <typename T>
std::size_t bincount(const T& axes) {
  std::size_t n = 1;
  for_each_axis(axes, [&n](const auto& a) { n *= axis::traits::extend(a); });
  return n;
}

template <typename... Ts, std::size_t N>
void dimension_check(const std::tuple<Ts...>&, mp11::mp_size_t<N>) {
  static_assert(sizeof...(Ts) == N, "number of arguments does not match");
}

template <typename... Ts>
void dimension_check(const std::tuple<Ts...>&, std::size_t n) {
  if (sizeof...(Ts) != n)
    throw std::invalid_argument("number of arguments does not match");
}

template <typename... Ts, std::size_t N>
void dimension_check(const std::vector<Ts...>& axes, mp11::mp_size_t<N>) {
  if (axes.size() != N) throw std::invalid_argument("number of arguments does not match");
}

template <typename... Ts>
void dimension_check(const std::vector<Ts...>& axes, std::size_t n) {
  if (axes.size() != n) throw std::invalid_argument("number of arguments does not match");
}

struct shape_collector {
  std::vector<unsigned>::iterator iter;
  shape_collector(std::vector<unsigned>::iterator i) : iter(i) {}
  template <typename T>
  void operator()(const T& t) {
    *iter++ = axis::traits::extend(t);
  }
};

namespace {

template <typename LN, typename T>
struct sub_axes_impl {};

template <typename LN, typename... Ts>
struct sub_axes_impl<LN, std::tuple<Ts...>> {
  static_assert(mp11::mp_is_set<LN>::value,
                "integer arguments must be strictly ascending");
  static_assert(mp_last<LN>::value < sizeof...(Ts), "index out of range");
  template <typename I>
  using at = mp11::mp_at<mp11::mp_list<Ts...>, I>;
  using L = mp11::mp_rename<LN, std::tuple>;
  using type = mp11::mp_transform<at, L>;
};

template <typename LN, typename... Ts>
struct sub_axes_impl<LN, std::vector<Ts...>> {
  static_assert(mp11::mp_is_set<LN>::value,
                "integer arguments must be strictly ascending");
  using type = std::vector<Ts...>;
};
} // namespace

template <typename T, typename... Ns>
using sub_axes = typename sub_axes_impl<mp11::mp_list<Ns...>, T>::type;

namespace {
template <typename Src, typename Dst>
struct sub_static_assign_impl {
  const Src& src;
  Dst& dst;
  template <typename I1, typename I2>
  void operator()(std::pair<I1, I2>) const {
    std::get<I1::value>(dst) = std::get<I2::value>(src);
  }
};
} // namespace

template <typename... Ts, typename... Ns>
sub_axes<std::tuple<Ts...>, Ns...> make_sub_axes(const std::tuple<Ts...>& t, Ns...) {
  using T = std::tuple<Ts...>;
  using U = sub_axes<std::tuple<Ts...>, Ns...>;
  U u;
  using N1 = mp11::mp_list<Ns...>;
  using N2 = mp11::mp_iota<mp11::mp_size<N1>>;
  using N3 = mp11::mp_transform<std::pair, N2, N1>;
  mp11::mp_for_each<N3>(sub_static_assign_impl<T, U>{t, u});
  return u;
}

template <typename... Ts, typename... Ns>
sub_axes<std::vector<Ts...>, Ns...> make_sub_axes(const std::vector<Ts...>& t, Ns...) {
  using T = std::vector<Ts...>;
  T u(t.get_allocator());
  u.reserve(sizeof...(Ns));
  using N = mp11::mp_list<Ns...>;
  mp11::mp_for_each<N>([&](auto I) { u.emplace_back(t[I]); });
  return u;
}

/// Index with an invalid state
struct optional_index {
  std::size_t idx = 0;
  std::size_t stride = 1;
  operator bool() const { return stride > 0; }
  std::size_t operator*() const { return idx; }
};

inline void linearize(optional_index& out, const int axis_size, const int axis_shape,
                      int j) noexcept {
  BOOST_ASSERT_MSG(out.stride == 0 || (-1 <= j && j <= axis_size),
                   "index must be in bounds for this algorithm");
  if (j < 0) j += (axis_size + 2); // wrap around if j < 0
  out.idx += j * out.stride;
  // set stride to 0, if j is invalid
  out.stride *= (j < axis_shape) * axis_shape;
}

template <typename... Ts, typename U>
void linearize1(optional_index& out, const axis::variant<Ts...>& axis, const U& u) {
  axis::visit([&](const auto& a) { linearize1(out, a, u); }, axis);
}

template <typename A, typename U>
void linearize1(optional_index& out, const A& axis, const U& u) {
  // protect against instantiation with wrong template argument
  using arg_type = axis::traits::arg<A>;
  static_if<std::is_convertible<U, arg_type>>(
      [&](const auto& u) {
        const auto a_size = axis.size();
        const auto a_shape = axis::traits::extend(axis);
        const auto j = axis(u);
        linearize(out, a_size, a_shape, j);
      },
      [&](const U&) {
        throw std::invalid_argument(
            detail::cat(boost::core::demangled_name(BOOST_CORE_TYPEID(A)),
                        ": cannot convert argument of type ",
                        boost::core::demangled_name(BOOST_CORE_TYPEID(U)), " to ",
                        boost::core::demangled_name(BOOST_CORE_TYPEID(arg_type))));
      },
      u);
}

template <typename T>
void linearize2(optional_index& out, const T& axis, const int j) {
  const auto a_size = static_cast<int>(axis.size());
  const auto a_shape = axis::traits::extend(axis);
  out.stride *= (-1 <= j && j <= a_size); // set stride to 0, if j is invalid
  linearize(out, a_size, a_shape, j);
}

template <typename S, typename A, typename... Us>
void fill_impl(mp11::mp_int<(sizeof...(Us) - 1)>, S& storage, const A& axes,
               const std::tuple<Us...>& args) {
  dimension_check(axes, sizeof...(Us) - 1);
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota_c<(sizeof...(Us) - 1)>>(
      [&](auto I) { linearize1(idx, axis_get<I>(axes), std::get<I>(args)); });
  if (idx) storage(*idx, std::get<(sizeof...(Us) - 1)>(args));
}

template <typename S, typename A, typename... Us>
void fill_impl(mp11::mp_int<0>, S& storage, const A& axes,
               const std::tuple<Us...>& args) {
  dimension_check(axes, sizeof...(Us) - 1);
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota_c<(sizeof...(Us) - 1)>>(
      [&](auto I) { linearize1(idx, axis_get<I>(axes), std::get<(I + 1)>(args)); });
  if (idx) storage(*idx, std::get<0>(args));
}

template <typename S, typename T1, typename T2, typename... Ts, typename... Us>
void fill_impl(mp11::mp_int<-1>, S& storage, const std::tuple<T1, T2, Ts...>& axes,
               const std::tuple<Us...>& args) {
  optional_index idx;
  dimension_check(axes, sizeof...(Us));
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Us)>>(
      [&](auto I) { linearize1(idx, axis_get<I>(axes), std::get<I>(args)); });
  if (idx) storage(*idx);
}

template <typename S, typename T, typename... Us>
void fill_impl(mp11::mp_int<-1>, S& storage, const std::tuple<T>& axes,
               const std::tuple<Us...>& args) {
  // special case that needs handling: 1d histogram, histogram::operator()
  // called with tuple(2, 1), while histogram may have axis that accepts 2d tuple
  // - normally would be interpret as two arguments passed, but here is one argument
  // - cannot check call signature of the axis at compile-time in all configurations
  //   (axis::variant provides generic call interface and hides concrete interface)
  // - solution: forward tuples of size > 1 directly to axis for 1d histograms
  optional_index idx;
  if (sizeof...(Us) > 1) {
    linearize1(idx, axis_get<0>(axes), args);
  } else {
    dimension_check(axes, sizeof...(Us));
    linearize1(idx, axis_get<0>(axes), std::get<0>(args));
  }
  if (idx) storage(*idx);
}

template <typename S, typename A, typename... Us>
void fill_impl(mp11::mp_int<-1>, S& storage, const A& axes,
               const std::tuple<Us...>& args) {
  // special case as above, but for dynamic axes
  optional_index idx;
  if (sizeof...(Us) > 1 && axes.size() == 1) {
    linearize1(idx, axis_get<0>(axes), args);
  } else {
    dimension_check(axes, sizeof...(Us));
    mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Us)>>(
        [&](auto I) { linearize1(idx, axis_get<I>(axes), std::get<I>(args)); });
  }
  if (idx) storage(*idx);
}

template <typename L>
constexpr int weight_index() {
  const int n = mp11::mp_size<L>::value - 1;
  if (is_weight<mp11::mp_first<L>>::value) return 0;
  if (is_weight<mp11::mp_at_c<L, n>>::value) return n;
  return -1;
}

// generic entry point which analyses args and calls specific
template <typename S, typename T, typename U>
void fill_impl(S& s, const T& axes, const U& args) {
  fill_impl(mp11::mp_int<weight_index<unqual<U>>()>(), s, axes, args);
}

/* In all at_impl, we throw instead of asserting when an index is out of
 * bounds, because wrapping code cannot check this condition without spending
 * a lot of extra cycles. For the wrapping code it is much easier to catch
 * the exception and do something sensible.
 */

template <typename A, typename... Us>
optional_index at_impl(const A& axes, const std::tuple<Us...>& args) {
  dimension_check(axes, sizeof...(Us));
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Us)>>([&](auto I) {
    linearize2(idx, axis_get<I>(axes), static_cast<int>(std::get<I>(args)));
  });
  return idx;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
