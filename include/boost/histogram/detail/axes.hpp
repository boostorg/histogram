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

template <typename T, std::size_t N = std::tuple_size<T>::value>
constexpr std::size_t axes_size(const T&) noexcept {
  return N;
}

// static to fix gcc warning about mangled names changing in C++17
template <typename T, typename = decltype(&T::size)>
static std::size_t axes_size(const T& axes) noexcept {
  return axes.size();
}

template <typename T>
void range_check(const T& axes, const unsigned N) {
  BOOST_ASSERT_MSG(N < axes_size(axes), "index out of range");
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

struct shape_collector {
  std::vector<unsigned>::iterator iter;
  shape_collector(std::vector<unsigned>::iterator i) : iter(i) {}
  template <typename T>
  void operator()(const T& t) {
    *iter++ = axis::traits::extend(t);
  }
};

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

template <typename T, typename... Ns>
using sub_axes = typename sub_axes_impl<mp11::mp_list<Ns...>, T>::type;

template <typename Src, typename Dst>
struct sub_static_assign_impl {
  const Src& src;
  Dst& dst;
  template <typename I1, typename I2>
  void operator()(std::pair<I1, I2>) const {
    std::get<I1::value>(dst) = std::get<I2::value>(src);
  }
};

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
  using arg_t = arg_type<A>;
  static_if<std::is_convertible<U, arg_t>>(
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
                        boost::core::demangled_name(BOOST_CORE_TYPEID(arg_t))));
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

// special case: histogram::operator(tuple(1, 2)) is called on 1d histogram with axis
// that accepts 2d tuple, this should work and not fail
// - solution is to forward tuples of size > 1 directly to axis for 1d histograms
// - has nice side-effect of making histogram::operator(1, 2) work as well
// - cannot detect call signature of axis at compile-time in all configurations
//   (axis::variant provides generic call interface and hides concrete interface),
//   so we throw at runtime if incompatible argument is passed (e.g. 3d tuple)
template <unsigned Offset, unsigned N, typename T, typename U>
optional_index args_to_index(const std::tuple<T>& axes, const U& args) {
  optional_index idx;
  if (N > 1) {
    linearize1(idx, std::get<0>(axes), sub_tuple<Offset, N>(args));
  } else {
    linearize1(idx, std::get<0>(axes), std::get<Offset>(args));
  }
  return idx;
}

template <unsigned Offset, unsigned N, typename T0, typename T1, typename... Ts,
          typename U>
optional_index args_to_index(const std::tuple<T0, T1, Ts...>& axes, const U& args) {
  static_assert(sizeof...(Ts) + 2 == N, "number of arguments != histogram rank");
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota_c<N>>(
      [&](auto I) { linearize1(idx, std::get<I>(axes), std::get<(Offset + I)>(args)); });
  return idx;
}

// overload for dynamic axes
template <unsigned Offset, unsigned N, typename T, typename U>
optional_index args_to_index(const T& axes, const U& args) {
  const unsigned m = axes.size();
  optional_index idx;
  if (m == 1 && N > 1)
    linearize1(idx, axes[0], sub_tuple<Offset, N>(args));
  else {
    if (m != N) throw std::invalid_argument("number of arguments != histogram rank");
    mp11::mp_for_each<mp11::mp_iota_c<N>>(
        [&](auto I) { linearize1(idx, axes[I], std::get<(Offset + I)>(args)); });
  }
  return idx;
}

template <typename... Us>
constexpr int weight_index() {
  const int n = sizeof...(Us) - 1;
  using L = mp11::mp_list<Us...>;
  if (is_weight<mp11::mp_at_c<L, 0>>::value) return 0;
  if (is_weight<mp11::mp_at_c<L, n>>::value) return n;
  return -1;
}

template <typename S, typename T>
void fill_storage_impl(mp11::mp_int<-1>, mp11::mp_int<-1>, S& storage, std::size_t i,
                       const T&) {
  storage(i);
}

template <int Iw, typename S, typename T>
void fill_storage_impl(mp11::mp_int<Iw>, mp11::mp_int<-1>, S& storage, std::size_t i,
                       const T& args) {
  storage(i, std::get<Iw>(args));
}

template <int Is, typename S, typename T>
void fill_storage_impl(mp11::mp_int<-1>, mp11::mp_int<Is>, S& storage, std::size_t i,
                       const T& args) {
  storage(i, std::get<Is>(args).value);
}

template <int Iw, int Is, typename S, typename T>
void fill_storage_impl(mp11::mp_int<Iw>, mp11::mp_int<Is>, S& storage, std::size_t i,
                       const T& args) {
  storage(i, std::get<Iw>(args), std::get<Is>(args).value);
}

template <typename S, typename T, typename... Us>
void fill_impl(S& storage, const T& axes, const std::tuple<Us...>& args) {
  constexpr int Iw = weight_index<Us...>();
  constexpr unsigned N = Iw >= 0 ? sizeof...(Us) - 1 : sizeof...(Us);
  optional_index idx = args_to_index<(Iw == 0 ? 1 : 0), N>(axes, args);
  if (idx) {
    fill_storage_impl(mp11::mp_int<Iw>(), mp11::mp_int<-1>(), storage, *idx, args);
  }
}

template <typename A, typename... Us>
optional_index at_impl(const A& axes, const std::tuple<Us...>& args) {
  if (axes_size(axes) != sizeof...(Us))
    throw std::invalid_argument("number of arguments != histogram rank");
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
