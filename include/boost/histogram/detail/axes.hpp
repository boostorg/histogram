// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_AXES_HPP
#define BOOST_HISTOGRAM_DETAIL_AXES_HPP

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mp11.hpp>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

namespace {

template <typename Tuple, typename Vector>
struct axes_equal_static_dynamic_impl {
  bool& equal;
  const Tuple& t;
  const Vector& v;
  axes_equal_static_dynamic_impl(bool& eq, const Tuple& tt, const Vector& vv)
      : equal(eq), t(tt), v(vv) {}
  template <typename N>
  void operator()(N) const {
    using T = mp11::mp_at<Tuple, N>;
    auto tp = axis::get<T>(&v[N::value]);
    equal &= (tp && *tp == std::get<N::value>(t));
  }
};

template <typename Tuple>
bool axes_equal_static_static_impl(mp11::mp_true, const Tuple& t, const Tuple& u) {
  return t == u;
}

template <typename Tuple1, typename Tuple2>
bool axes_equal_static_static_impl(mp11::mp_false, const Tuple1&, const Tuple2&) {
  return false;
}

template <typename Tuple, typename Vector>
struct axes_assign_static_dynamic_impl {
  Tuple& t;
  const Vector& v;
  axes_assign_static_dynamic_impl(Tuple& tt, const Vector& vv) : t(tt), v(vv) {}
  template <typename N>
  void operator()(N) const {
    using T = mp11::mp_at<Tuple, N>;
    std::get<N::value>(t) = axis::get<T>(v[N::value]);
  }
};

template <typename Vector, typename Tuple>
struct axes_assign_dynamic_static_impl {
  Vector& v;
  const Tuple& t;
  axes_assign_dynamic_static_impl(Vector& vv, const Tuple& tt) : v(vv), t(tt) {}
  template <typename N>
  void operator()(N) const {
    v[N::value] = std::get<N::value>(t);
  }
};
} // namespace

template <typename... Ts, typename... Us>
bool axes_equal(const std::tuple<Ts...>& t, const std::tuple<Us...>& u) {
  return axes_equal_static_static_impl(
      mp11::mp_same<mp11::mp_list<Ts...>, mp11::mp_list<Us...>>(), t, u);
}

template <typename... Ts, typename... Us>
bool axes_equal(const std::tuple<Ts...>& t, const std::vector<Us...>& u) {
  if (sizeof...(Ts) != u.size()) return false;
  bool equal = true;
  auto fn =
      axes_equal_static_dynamic_impl<std::tuple<Ts...>, std::vector<Us...>>(equal, t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>(fn);
  return equal;
}

template <typename... Ts, typename... Us>
bool axes_equal(const std::vector<Ts...>& t, const std::tuple<Us...>& u) {
  return axes_equal(u, t);
}

template <typename... Ts, typename... Us>
bool axes_equal(const std::vector<Ts...>& t, const std::vector<Us...>& u) {
  if (t.size() != u.size()) return false;
  return std::equal(t.begin(), t.end(), u.begin());
}

template <typename... Ts, typename... Us>
void axes_assign(std::tuple<Ts...>& t, const std::tuple<Us...>& u) {
  static_assert(std::is_same<mp11::mp_list<Ts...>, mp11::mp_list<Us...>>::value,
                "cannot assign incompatible axes");
  t = u;
}

template <typename... Ts, typename... Us>
void axes_assign(std::tuple<Ts...>& t, const std::vector<Us...>& u) {
  auto fn = axes_assign_static_dynamic_impl<std::tuple<Ts...>, std::vector<Us...>>(t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>(fn);
}

template <typename... Ts, typename... Us>
void axes_assign(std::vector<Ts...>& t, const std::tuple<Us...>& u) {
  t.resize(sizeof...(Us));
  auto fn = axes_assign_dynamic_static_impl<std::vector<Ts...>, std::tuple<Us...>>(t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Us)>>(fn);
}

template <typename... Ts, typename... Us>
void axes_assign(std::vector<Ts...>& t, const std::vector<Us...>& u) {
  t.assign(u.begin(), u.end());
}

template <typename... Ts>
constexpr std::size_t axes_size(const std::tuple<Ts...>&) {
  return sizeof...(Ts);
}

template <typename... Ts>
std::size_t axes_size(const std::vector<Ts...>& axes) {
  return axes.size();
}

template <int N, typename... Ts>
void range_check(const std::vector<Ts...>& axes) {
  BOOST_ASSERT_MSG(N < axes.size(), "index out of range");
}

template <int N, typename... Ts>
void range_check(const std::tuple<Ts...>&) {
  static_assert(N < sizeof...(Ts), "index out of range");
}

template <int N, typename T, typename = requires_static_container<T>>
auto axis_get(T&& axes) -> decltype(std::get<N>(std::forward<T>(axes))) {
  return std::get<N>(std::forward<T>(axes));
}

template <int N, typename T, typename = requires_axis_vector<T>>
auto axis_get(T&& axes) -> decltype(std::forward<T>(axes)[N]) {
  return std::forward<T>(axes)[N];
}

template <typename F, typename... Ts>
void for_each_axis(const std::tuple<Ts...>& axes, F&& f) {
  mp11::tuple_for_each(axes, std::forward<F>(f));
}

template <typename F, typename... Ts>
void for_each_axis(const std::vector<Ts...>& axes, F&& f) {
  for (const auto& x : axes) { axis::visit(std::forward<F>(f), x); }
}

namespace {
struct field_counter {
  std::size_t value = 1;
  template <typename T>
  void operator()(const T& t) {
    value *= axis::traits::extend(t);
  }
};
} // namespace

template <typename T>
std::size_t bincount(const T& axes) {
  field_counter fc;
  for_each_axis(axes, fc);
  return fc.value;
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

namespace {
template <typename T>
struct sub_dynamic_assign_impl {
  const T& src;
  T& dst;
  template <typename I>
  void operator()(I) const {
    dst.emplace_back(src[I::value]);
  }
};
} // namespace

template <typename... Ts, typename... Ns>
sub_axes<std::vector<Ts...>, Ns...> make_sub_axes(const std::vector<Ts...>& t, Ns...) {
  using T = std::vector<Ts...>;
  T u(t.get_allocator());
  u.reserve(sizeof...(Ns));
  using N = mp11::mp_list<Ns...>;
  mp11::mp_for_each<N>(sub_dynamic_assign_impl<T>{t, u});
  return u;
}

struct optional_index {
  std::size_t idx = 0;
  std::size_t stride = 1;
  operator bool() const { return stride > 0; }
  std::size_t operator*() const { return idx; }
};

// the following is highly optimized code that runs in a hot loop;
// please measure the performance impact of changes
inline void linearize(optional_index& out, const int axis_size, const int axis_shape,
                      int j) noexcept {
  BOOST_ASSERT_MSG(out.stride == 0 || (-1 <= j && j <= axis_size),
                   "index must be in bounds for this algorithm");
  j += (j < 0) * (axis_size + 2); // wrap around if j < 0
  out.idx += j * out.stride;
  out.stride *= (j < axis_shape) * axis_shape; // set to 0, if j is invalid
}

template <typename T>
void linearize2(optional_index& out, const T& axis, int j) {
  const auto a_size = static_cast<int>(axis.size());
  const auto a_shape = axis::traits::extend(axis);
  out.stride *= (-1 <= j && j <= a_size); // set to 0, if j is invalid
  linearize(out, a_size, a_shape, j);
}

template <typename T, typename U>
void linearize1(optional_index& out, const T& axis, const U& u) {
  const auto a_size = axis.size();
  const auto a_shape = axis::traits::extend(axis);
  const auto j = axis(u);
  linearize(out, a_size, a_shape, j);
}

template <typename... Ts, typename U>
void linearize1(optional_index& out, const axis::variant<Ts...>& axis, const U& u) {
  axis::visit(
      [&](const auto& a) {
        using A = unqual<decltype(a)>;
        using arg_type = axis::traits::arg<A>;
        static_if<std::is_convertible<U, arg_type>>(
            [&](const auto& u) { linearize1(out, a, u); },
            [&](const U&) {
              throw std::invalid_argument(
                  detail::cat(boost::core::demangled_name(BOOST_CORE_TYPEID(A)),
                              ": cannot convert argument of type ",
                              boost::core::demangled_name(BOOST_CORE_TYPEID(U)), " to ",
                              boost::core::demangled_name(BOOST_CORE_TYPEID(arg_type))));
            },
            u);
      },
      axis);
}

// specialization for one-dimensional histograms
template <typename Tag, typename T, typename U>
optional_index call_impl(Tag, const std::tuple<T>& axes, const U& u) {
  dimension_check(axes, 1);
  optional_index idx;
  linearize1(idx, std::get<0>(axes), u);
  return idx;
}

template <typename T1, typename T2, typename... Ts, typename... Us>
optional_index call_impl(no_container_tag, const std::tuple<T1, T2, Ts...>& axes,
                         const Us&... us) {
  return call_impl(static_container_tag(), axes, std::forward_as_tuple(us...));
}

template <typename T1, typename T2, typename... Ts, typename U>
optional_index call_impl(static_container_tag, const std::tuple<T1, T2, Ts...>& axes,
                         const U& u) {
  dimension_check(axes, mp_size<U>());
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota_c<(2 + sizeof...(Ts))>>(
      [&](auto I) { linearize1(idx, std::get<I>(axes), std::get<I>(u)); });
  return idx;
}

template <typename T1, typename T2, typename... Ts, typename U>
optional_index call_impl(iterable_container_tag, const std::tuple<T1, T2, Ts...>& axes,
                         const U& u) {
  dimension_check(axes, u.size());
  optional_index idx;
  auto xit = std::begin(u);
  mp11::mp_for_each<mp11::mp_iota_c<(2 + sizeof...(Ts))>>(
      [&](auto I) { linearize1(idx, std::get<I>(axes), *xit++); });
  return idx;
}

template <typename... Ts, typename U>
optional_index call_impl(no_container_tag, const std::vector<Ts...>& axes, const U& u) {
  dimension_check(axes, 1);
  optional_index idx;
  linearize1(idx, axes[0], u);
  return idx;
}

template <typename... Ts, typename U>
optional_index call_impl(static_container_tag, const std::vector<Ts...>& axes,
                         const U& u) {
  if (axes.size() == 1) // do not unpack for 1d histograms, it is ambiguous
    return call_impl(no_container_tag(), axes, u);
  dimension_check(axes, mp11::mp_size<unqual<U>>());
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota<mp_size<U>>>([&](auto I) {
    linearize1(idx, axis_get<decltype(I)::value>(axes), std::get<I>(u));
  });
  return idx;
}

template <typename... Ts, typename U>
optional_index call_impl(iterable_container_tag, const std::vector<Ts...>& axes,
                         const U& u) {
  if (axes.size() == 1) // do not unpack for 1d histograms, it is ambiguous
    return call_impl(no_container_tag(), axes, u);
  dimension_check(axes, std::distance(std::begin(u), std::end(u)));
  optional_index idx;
  auto xit = std::begin(u);
  for (const auto& a : axes) { linearize1(idx, a, *xit++); }
  return idx;
}

/* In all at_impl, we throw instead of asserting when an index is out of
 * bounds, because wrapping code cannot check this condition without spending
 * a lot of extra cycles. For the wrapping code it is much easier to catch
 * the exception and do something sensible.
 */

template <typename A, typename U>
optional_index at_impl(no_container_tag, const A& axes, const U& u) {
  return at_impl(static_container_tag(), axes, std::forward_as_tuple(u));
}

template <typename A, typename U>
optional_index at_impl(static_container_tag, const A& axes, const U& u) {
  dimension_check(axes, mp11::mp_size<unqual<U>>());
  optional_index idx;
  mp11::mp_for_each<mp11::mp_iota<mp_size<U>>>([&](auto I) {
    linearize2(idx, axis_get<I>(axes), static_cast<int>(std::get<I>(u)));
  });
  return idx;
}

template <typename... Ts, typename U>
optional_index at_impl(iterable_container_tag, const std::tuple<Ts...>& axes,
                       const U& u) {
  dimension_check(axes, std::distance(std::begin(u), std::end(u)));
  optional_index idx;
  auto xit = std::begin(u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>(
      [&](auto I) { linearize2(idx, std::get<I>(axes), static_cast<int>(*xit++)); });
  return idx;
}

template <typename... Ts, typename U>
optional_index at_impl(iterable_container_tag, const std::vector<Ts...>& axes,
                       const U& u) {
  dimension_check(axes, std::distance(std::begin(u), std::end(u)));
  optional_index idx;
  auto xit = std::begin(u);
  for (const auto& a : axes) linearize2(idx, a, static_cast<int>(*xit++));
  return idx;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
