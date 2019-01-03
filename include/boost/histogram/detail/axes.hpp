// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_AXES_HPP
#define BOOST_HISTOGRAM_DETAIL_AXES_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

/* Most of the histogram code is generic and works for any number of axes. Buffers with a
 * fixed maximum capacity are used in some places, which have a size equal to the rank of
 * a histogram. The buffers are statically allocated to improve performance, which means
 * that they need a preset maximum capacity. 32 seems like a safe upper limit for the rank
 * (you can nevertheless increase it here if necessary): the simplest non-trivial axis has
 * 2 bins; even if counters are used which need only a byte of storage per bin, this still
 * corresponds to 4 GB of storage.
 */
#ifndef BOOST_HISTOGRAM_DETAIL_AXES_LIMIT
#define BOOST_HISTOGRAM_DETAIL_AXES_LIMIT 32
#endif

namespace boost {
namespace histogram {
namespace detail {

template <unsigned N, typename... Ts>
decltype(auto) axis_get(std::tuple<Ts...>& axes) {
  return std::get<N>(axes);
}

template <unsigned N, typename... Ts>
decltype(auto) axis_get(const std::tuple<Ts...>& axes) {
  return std::get<N>(axes);
}

template <unsigned N, typename T>
decltype(auto) axis_get(T& axes) {
  return axes[N];
}

template <unsigned N, typename T>
decltype(auto) axis_get(const T& axes) {
  return axes[N];
}

template <typename... Ts>
decltype(auto) axis_get(std::tuple<Ts...>& axes, unsigned i) {
  return mp11::mp_with_index<sizeof...(Ts)>(
      i, [&](auto I) { return axis::variant<Ts&...>(std::get<I>(axes)); });
}

template <typename... Ts>
decltype(auto) axis_get(const std::tuple<Ts...>& axes, unsigned i) {
  return mp11::mp_with_index<sizeof...(Ts)>(
      i, [&](auto I) { return axis::variant<const Ts&...>(std::get<I>(axes)); });
}

template <typename T>
decltype(auto) axis_get(T& axes, unsigned i) {
  return axes.at(i);
}

template <typename T>
decltype(auto) axis_get(const T& axes, unsigned i) {
  return axes.at(i);
}

template <typename... Ts, typename... Us>
bool axes_equal(const std::tuple<Ts...>& t, const std::tuple<Us...>& u) {
  return static_if<std::is_same<mp11::mp_list<Ts...>, mp11::mp_list<Us...>>>(
      [](const auto& a, const auto& b) { return relaxed_equal(a, b); },
      [](const auto&, const auto&) { return false; }, t, u);
}

template <typename... Ts, typename U>
bool axes_equal(const std::tuple<Ts...>& t, const U& u) {
  if (sizeof...(Ts) != u.size()) return false;
  bool equal = true;
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>([&](auto I) {
    using T = mp11::mp_at<std::tuple<Ts...>, decltype(I)>;
    auto up = axis::get<T>(&u[I]);
    equal &= (up && relaxed_equal(std::get<I>(t), *up));
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
        BOOST_THROW_EXCEPTION(
            std::invalid_argument("cannot assign axes, types do not match"));
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

template <typename T>
constexpr std::size_t axes_size(const T& axes) noexcept {
  return static_if<has_fixed_size<unqual<T>>>(
      [](const auto& a) {
        using U = unqual<decltype(a)>;
        return std::tuple_size<U>::value;
      },
      [&](const auto& a) { return a.size(); }, axes);
}

template <typename T>
void rank_check(const T& axes, const unsigned N) {
  BOOST_ASSERT_MSG(N < axes_size(axes), "index out of range");
}

template <typename F, typename T>
void for_each_axis_impl(std::true_type, const T& axes, F&& f) {
  for (const auto& x : axes) { axis::visit(std::forward<F>(f), x); }
}

template <typename F, typename T>
void for_each_axis_impl(std::false_type, const T& axes, F&& f) {
  for (const auto& x : axes) f(x);
}

template <typename F, typename T>
void for_each_axis(const T& axes, F&& f) {
  using U = mp11::mp_first<unqual<T>>;
  for_each_axis_impl(is_axis_variant<U>(), axes, std::forward<F>(f));
}

template <typename F, typename... Ts>
void for_each_axis(const std::tuple<Ts...>& axes, F&& f) {
  mp11::tuple_for_each(axes, std::forward<F>(f));
}

template <typename Axes, typename T>
using axes_buffer = boost::container::static_vector<
    T, mp11::mp_eval_if_c<!(has_fixed_size<Axes>::value),
                          mp11::mp_size_t<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>,
                          std::tuple_size, Axes>::value>;

template <typename T>
auto make_empty_axes(const T& t) {
  auto r = T(t.get_allocator());
  r.reserve(t.size());
  for_each_axis(t, [&r](const auto& a) {
    using U = unqual<decltype(a)>;
    r.emplace_back(U());
  });
  return r;
}

template <typename... Ts>
auto make_empty_axes(const std::tuple<Ts...>&) {
  return std::tuple<Ts...>();
}

template <typename T>
std::size_t bincount(const T& axes) {
  std::size_t n = 1;
  for_each_axis(axes, [&n](const auto& a) { n *= axis::traits::extend(a); });
  return n;
}

template <typename... Ns, typename... Ts>
auto make_sub_axes(const std::tuple<Ts...>& t, Ns... ns) {
  return std::make_tuple(std::get<ns>(t)...);
}

template <typename... Ns, typename T>
auto make_sub_axes(const T& t, Ns... ns) {
  return T({t[ns]...}, t.get_allocator());
}

/// Index with an invalid state
struct optional_index {
  std::size_t idx = 0;
  std::size_t stride = 1;
  operator bool() const { return stride > 0; }
  std::size_t operator*() const { return idx; }
};

inline void linearize(optional_index& out, const int axis_shape, int j) noexcept {
  // j is internal index, which is potentially shifted by +1 wrt external index
  out.idx += j * out.stride;
  // set stride to 0, if j is invalid
  out.stride *= (0 <= j && j < axis_shape) * axis_shape;
}

template <typename A, typename U>
void linearize_value(std::true_type, optional_index& out, const A& axis, const U& u) {
  const auto extend = axis::traits::extend(axis);
  const auto opt = axis::traits::options(axis);
  const auto j = axis(u) + (opt & axis::option_type::underflow);
  return linearize(out, extend, j);
}

template <typename A, typename U>
void linearize_value(std::false_type, optional_index&, const A&, const U&) {
  // protect against instantiation with wrong template argument
  using arg_t = arg_type<A>;
  BOOST_THROW_EXCEPTION(std::invalid_argument(
      detail::cat(boost::core::demangled_name(BOOST_CORE_TYPEID(A)),
                  ": cannot convert argument of type ",
                  boost::core::demangled_name(BOOST_CORE_TYPEID(U)), " to ",
                  boost::core::demangled_name(BOOST_CORE_TYPEID(arg_t)))));
}

template <typename A, typename U>
void linearize_value(optional_index& out, const A& axis, const U& u) {
  // protect against instantiation with wrong template argument
  using arg_t = arg_type<A>;
  return linearize_value(std::is_convertible<U, arg_t>(), out, axis, u);
}

template <typename... Ts, typename U>
void linearize_value(optional_index& out, const axis::variant<Ts...>& axis, const U& u) {
  axis::visit([&](const auto& a) { linearize_value(out, a, u); }, axis);
}

template <typename T>
void linearize_index(optional_index& out, const T& axis, const int j) {
  const auto extend = axis::traits::extend(axis);
  const auto opt = axis::traits::options(axis);
  linearize(out, extend, j + (opt & axis::option_type::underflow));
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
    linearize_value(idx, std::get<0>(axes), sub_tuple<Offset, N>(args));
  } else {
    linearize_value(idx, std::get<0>(axes), std::get<Offset>(args));
  }
  return idx;
}

template <unsigned Offset, unsigned N, typename T0, typename T1, typename... Ts,
          typename U>
optional_index args_to_index(const std::tuple<T0, T1, Ts...>& axes, const U& args) {
  static_assert(sizeof...(Ts) + 2 == N, "number of arguments != histogram rank");
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota_c<N>>([&](auto I) {
    linearize_value(idx, std::get<I>(axes), std::get<(Offset + I)>(args));
  });
  return idx;
}

// overload for dynamic axes
template <unsigned Offset, unsigned N, typename T, typename U>
optional_index args_to_index(const T& axes, const U& args) {
  const unsigned m = axes.size();
  optional_index idx;
  if (m == 1 && N > 1)
    linearize_value(idx, axes[0], sub_tuple<Offset, N>(args));
  else {
    if (m != N)
      BOOST_THROW_EXCEPTION(
          std::invalid_argument("number of arguments != histogram rank"));
    mp11::mp_for_each<mp11::mp_iota_c<N>>(
        [&](auto I) { linearize_value(idx, axes[I], std::get<(Offset + I)>(args)); });
  }
  return idx;
}

template <typename U>
constexpr std::pair<int, int> weight_sample_indices() {
  if (is_weight<U>::value) return std::make_pair(0, -1);
  if (is_sample<U>::value) return std::make_pair(-1, 0);
  return std::make_pair(-1, -1);
}

template <typename U0, typename U1, typename... Us>
constexpr std::pair<int, int> weight_sample_indices() {
  using L = mp11::mp_list<U0, U1, Us...>;
  const int n = sizeof...(Us) + 1;
  if (is_weight<mp11::mp_at_c<L, 0>>::value) {
    if (is_sample<mp11::mp_at_c<L, 1>>::value) return std::make_pair(0, 1);
    if (is_sample<mp11::mp_at_c<L, n>>::value) return std::make_pair(0, n);
    return std::make_pair(0, -1);
  }
  if (is_sample<mp11::mp_at_c<L, 0>>::value) {
    if (is_weight<mp11::mp_at_c<L, 1>>::value) return std::make_pair(1, 0);
    if (is_weight<mp11::mp_at_c<L, n>>::value) return std::make_pair(n, 0);
    return std::make_pair(-1, 0);
  }
  if (is_weight<mp11::mp_at_c<L, n>>::value) {
    // 0, n already covered
    if (is_sample<mp11::mp_at_c<L, (n - 1)>>::value) return std::make_pair(n, n - 1);
    return std::make_pair(n, -1);
  }
  if (is_sample<mp11::mp_at_c<L, n>>::value) {
    // n, 0 already covered
    if (is_weight<mp11::mp_at_c<L, (n - 1)>>::value) return std::make_pair(n - 1, n);
    return std::make_pair(-1, n);
  }
  return std::make_pair(-1, -1);
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
  mp11::tuple_apply([&](auto&&... sargs) { storage(i, sargs...); },
                    std::get<Is>(args).value);
}

template <int Iw, int Is, typename S, typename T>
void fill_storage_impl(mp11::mp_int<Iw>, mp11::mp_int<Is>, S& storage, std::size_t i,
                       const T& args) {
  mp11::tuple_apply([&](auto&&... sargs) { storage(i, std::get<Iw>(args), sargs...); },
                    std::get<Is>(args).value);
}

template <typename S, typename T, typename... Us>
void fill_impl(S& storage, const T& axes, const std::tuple<Us...>& args) {
  constexpr std::pair<int, int> iws = weight_sample_indices<Us...>();
  constexpr unsigned n = sizeof...(Us) - (iws.first > -1) - (iws.second > -1);
  constexpr unsigned offset = (iws.first == 0 || iws.second == 0)
                                  ? (iws.first == 1 || iws.second == 1 ? 2 : 1)
                                  : 0;
  optional_index idx = args_to_index<offset, n>(axes, args);
  if (idx) {
    fill_storage_impl(mp11::mp_int<iws.first>(), mp11::mp_int<iws.second>(), storage,
                      *idx, args);
  }
}

template <typename A, typename... Us>
optional_index at_impl(const A& axes, const std::tuple<Us...>& args) {
  if (axes_size(axes) != sizeof...(Us))
    BOOST_THROW_EXCEPTION(std::invalid_argument("number of arguments != histogram rank"));
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Us)>>([&](auto I) {
    linearize_index(idx, axis_get<I>(axes), static_cast<int>(std::get<I>(args)));
  });
  return idx;
}

template <typename A, typename U>
optional_index at_impl(const A& axes, const U& args) {
  if (axes_size(axes) != args.size())
    BOOST_THROW_EXCEPTION(std::invalid_argument("number of arguments != histogram rank"));
  optional_index idx;
  using std::begin;
  auto it = begin(args);
  for_each_axis(axes,
                [&](const auto& a) { linearize_index(idx, a, static_cast<int>(*it++)); });
  return idx;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
