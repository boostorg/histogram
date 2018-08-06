// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_AXES_HPP_
#define _BOOST_HISTOGRAM_DETAIL_AXES_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/mp11.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/static_visitor.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

namespace {

template <typename StaticAxes, typename DynamicAxes>
struct axes_equal_static_dynamic_impl {
  bool& equal;
  const StaticAxes& t;
  const DynamicAxes& v;
  axes_equal_static_dynamic_impl(bool& eq, const StaticAxes& tt,
                                 const DynamicAxes& vv)
      : equal(eq), t(tt), v(vv) {}
  template <typename Int>
  void operator()(Int) const {
    using T = mp11::mp_at<StaticAxes, Int>;
    auto tp = boost::get<T>(&v[Int::value]);
    equal &= (tp && *tp == std::get<Int::value>(t));
  }
};

template <typename... Ts>
bool axes_equal_static_static_impl(mp11::mp_true, const static_axes<Ts...>& t,
                                   const static_axes<Ts...>& u) {
  return t == u;
}

template <typename... Ts, typename... Us>
bool axes_equal_static_static_impl(mp11::mp_false, const static_axes<Ts...>&,
                                   const static_axes<Us...>&) {
  return false;
}

template <typename StaticAxes, typename DynamicAxes>
struct axes_assign_static_dynamic_impl {
  StaticAxes& t;
  const DynamicAxes& v;
  axes_assign_static_dynamic_impl(StaticAxes& tt, const DynamicAxes& vv)
      : t(tt), v(vv) {}
  template <typename Int>
  void operator()(Int) const {
    using T = mp11::mp_at<StaticAxes, Int>;
    std::get<Int::value>(t) = boost::get<T>(v[Int::value]);
  }
};

template <typename DynamicAxes, typename StaticAxes>
struct axes_assign_dynamic_static_impl {
  DynamicAxes& v;
  const StaticAxes& t;
  axes_assign_dynamic_static_impl(DynamicAxes& vv, const StaticAxes& tt)
      : v(vv), t(tt) {}
  template <typename Int>
  void operator()(Int) const {
    v[Int::value] = std::get<Int::value>(t);
  }
};
} // namespace

template <typename... Ts, typename... Us>
bool axes_equal(const static_axes<Ts...>& t, const static_axes<Us...>& u) {
  return axes_equal_static_static_impl(
      mp11::mp_same<mp11::mp_list<Ts...>, mp11::mp_list<Us...>>(), t, u);
}

template <typename... Ts, typename... Us>
bool axes_equal(const static_axes<Ts...>& t, const dynamic_axes<Us...>& u) {
  if (sizeof...(Ts) != u.size()) return false;
  bool equal = true;
  auto fn =
      axes_equal_static_dynamic_impl<static_axes<Ts...>, dynamic_axes<Us...>>(
          equal, t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>(fn);
  return equal;
}

template <typename... Ts, typename... Us>
bool axes_equal(const dynamic_axes<Ts...>& t, const static_axes<Us...>& u) {
  return axes_equal(u, t);
}

template <typename... Ts, typename... Us>
bool axes_equal(const dynamic_axes<Ts...>& t, const dynamic_axes<Us...>& u) {
  if (t.size() != u.size()) return false;
  return std::equal(t.begin(), t.end(), u.begin());
}

template <typename... Ts, typename... Us>
void axes_assign(static_axes<Ts...>& t, const static_axes<Us...>& u) {
  static_assert(
      std::is_same<mp11::mp_list<Ts...>, mp11::mp_list<Us...>>::value,
      "cannot assign incompatible axes");
  t = u;
}

template <typename... Ts, typename... Us>
void axes_assign(static_axes<Ts...>& t, const dynamic_axes<Us...>& u) {
  auto fn = axes_assign_static_dynamic_impl<static_axes<Ts...>,
                                            dynamic_axes<Us...>>(t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>(fn);
}

template <typename... Ts, typename... Us>
void axes_assign(dynamic_axes<Ts...>& t, const static_axes<Us...>& u) {
  t.resize(sizeof...(Us));
  auto fn = axes_assign_dynamic_static_impl<dynamic_axes<Ts...>,
                                            static_axes<Us...>>(t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Us)>>(fn);
}

template <typename... Ts, typename... Us>
void axes_assign(dynamic_axes<Ts...>& t, const dynamic_axes<Us...>& u) {
  t.assign(u.begin(), u.end());
}

template <typename... Ts>
constexpr unsigned axes_size(const static_axes<Ts...>&) {
  return sizeof...(Ts);
}

template <typename... Ts>
unsigned axes_size(const dynamic_axes<Ts...>& axes) {
  return axes.size();
}

template <int N, typename... Ts>
void range_check(const dynamic_axes<Ts...>& axes) {
  BOOST_ASSERT_MSG(N < axes.size(), "index out of range");
}

template <int N, typename... Ts>
void range_check(const static_axes<Ts...>&) {
  static_assert(N < sizeof...(Ts), "index out of range");
}

namespace {
template <int N, typename T>
struct at_impl {};

template <int N, typename... Ts>
struct at_impl<N, static_axes<Ts...>> {
  using type = mp11::mp_at_c<static_axes<Ts...>, N>;
};

template <int N, typename... Ts>
struct at_impl<N, dynamic_axes<Ts...>> {
  using type = axis::any<Ts...>;
};
}

template <int N, typename T>
using at = typename at_impl<N, T>::type;

template <int N, typename... Ts>
auto get(static_axes<Ts...>& axes) -> at<N, static_axes<Ts...>>& {
  return std::get<N>(axes);
}

template <int N, typename... Ts>
auto get(const static_axes<Ts...>& axes) -> const at<N, static_axes<Ts...>>& {
  return std::get<N>(axes);
}

template <int N, typename... Ts>
axis::any<Ts...>& get(dynamic_axes<Ts...>& axes) {
  return axes[N];
}

template <int N, typename... Ts>
const axis::any<Ts...>& get(const dynamic_axes<Ts...>& axes) {
  return axes[N];
}

template <typename F, typename... Ts>
void for_each_axis(const static_axes<Ts...>& axes, F&& f) {
  mp11::tuple_for_each(axes, std::forward<F>(f));
}

namespace {
template <typename Unary>
struct unary_adaptor : public boost::static_visitor<void> {
  Unary&& unary;
  unary_adaptor(Unary&& u) : unary(std::forward<Unary>(u)) {}
  template <typename T>
  void operator()(const T& a) const {
    unary(a);
  }
};
}

template <typename F, typename... Ts>
void for_each_axis(const dynamic_axes<Ts...>& axes, F&& f) {
  for (const auto& x : axes) {
    boost::apply_visitor(unary_adaptor<F>(std::forward<F>(f)), x);
  }
}

namespace {
struct field_counter {
  std::size_t value = 1;
  template <typename T>
  void operator()(const T& t) {
    value *= t.shape();
  }
};
}

template <typename T>
std::size_t bincount(const T& axes) {
  field_counter fc;
  for_each_axis(axes, fc);
  return fc.value;
}

template <std::size_t N, typename... Ts>
void dimension_check(const static_axes<Ts...>&, mp11::mp_size_t<N>) {
  static_assert(sizeof...(Ts) == N, "number of arguments does not match");
}

template <std::size_t N, typename... Ts>
void dimension_check(const dynamic_axes<Ts...>& axes, mp11::mp_size_t<N>) {
  BOOST_ASSERT_MSG(axes.size() == N, "number of arguments does not match");
}

struct shape_collector {
  std::vector<unsigned>::iterator iter;
  shape_collector(std::vector<unsigned>::iterator i) : iter(i) {}
  template <typename T>
  void operator()(const T& a) {
    *iter++ = a.shape();
  }
};

namespace {
template <typename LN, typename T>
struct sub_axes_impl {};

template <typename LN, typename... Ts>
struct sub_axes_impl<LN, static_axes<Ts...>> {
  template <typename I>
  using at = mp11::mp_at<mp11::mp_list<Ts...>, I>;
  using L = mp11::mp_rename<unique_sorted<LN>, static_axes>;
  using type = mp11::mp_transform<at, L>;
};

template <typename LN, typename... Ts>
struct sub_axes_impl<LN, dynamic_axes<Ts...>> {
  using type = dynamic_axes<Ts...>;
};
}

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
}

template <typename... Ts, typename... Ns>
sub_axes<static_axes<Ts...>, Ns...> make_sub_axes(const static_axes<Ts...>& t,
                                                  Ns...) {
  using T = static_axes<Ts...>;
  using U = sub_axes<static_axes<Ts...>, Ns...>;
  U u;
  using N1 = unique_sorted<mp11::mp_list<Ns...>>;
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
}

template <typename... Ts, typename... Ns>
sub_axes<dynamic_axes<Ts...>, Ns...> make_sub_axes(
    const dynamic_axes<Ts...>& t, Ns...) {
  using T = dynamic_axes<Ts...>;
  T u;
  u.reserve(sizeof...(Ns));
  using N = unique_sorted<mp11::mp_list<Ns...>>;
  mp11::mp_for_each<N>(sub_dynamic_assign_impl<T>{t, u});
  return u;
}

struct optional_index {
  std::size_t idx = 0;
  std::size_t stride = 0;
  operator bool() const { return stride > 0; }
  std::size_t operator*() const { return idx; }
};

// the following is highly optimized code that runs in a hot loop;
// please measure the performance impact of changes
inline void linearize(optional_index& out, const int axis_size,
                      const int axis_shape, int j) noexcept {
  BOOST_ASSERT_MSG(out.stride == 0 || (-1 <= j && j <= axis_size),
                   "index must be in bounds for this algorithm");
  j += (j < 0) * (axis_size + 2); // wrap around if j < 0
  out.idx += j * out.stride;
  out.stride *= (j < axis_shape) * axis_shape; // set to 0, if j is invalid
}

template <unsigned D, typename Axes>
void indices_to_index(optional_index&, const Axes&) noexcept {}

template <unsigned D, typename Axes, typename... Us>
void indices_to_index(optional_index& idx, const Axes& axes, const int j,
                      const Us... us) {
  const auto& a = ::boost::histogram::detail::get<D>(axes);
  const auto a_size = a.size();
  const auto a_shape = a.shape();
  idx.stride *= (-1 <= j && j <= a_size); // set to 0, if j is invalid
  linearize(idx, a_size, a_shape, j);
  indices_to_index<(D + 1)>(idx, axes, us...);
}

template <unsigned D, typename... Ts, typename... Us>
void args_to_index(optional_index&, const static_axes<Ts...>&) noexcept {}

template <unsigned D, typename... Ts, typename U, typename... Us>
void args_to_index(optional_index& idx, const static_axes<Ts...>& axes,
                   const U& u, const Us&... us) {
  const auto a_size = std::get<D>(axes).size();
  const auto a_shape = std::get<D>(axes).shape();
  const int j = std::get<D>(axes).index(u);
  linearize(idx, a_size, a_shape, j);
  args_to_index<(D + 1)>(idx, axes, us...);
}

namespace {
template <typename T>
struct args_to_index_visitor : public boost::static_visitor<void> {
  optional_index& idx;
  const T& val;
  args_to_index_visitor(optional_index& i, const T& v) : idx(i), val(v) {}
  template <typename Axis>
  void operator()(const Axis& a) const {
    impl(std::is_convertible<T, typename Axis::value_type>(), a);
  }

  template <typename Axis>
  void impl(std::true_type, const Axis& a) const {
    const auto a_size = a.size();
    const auto a_shape = a.shape();
    const auto j = a.index(static_cast<typename Axis::value_type>(val));
    linearize(idx, a_size, a_shape, j);
  }

  template <typename Axis>
  void impl(std::false_type, const Axis&) const {
    throw std::invalid_argument(detail::cat(
        "axis ", boost::typeindex::type_id<Axis>().pretty_name(),
        ": argument ", boost::typeindex::type_id<T>().pretty_name(),
        " not convertible to value_type ",
        boost::typeindex::type_id<typename Axis::value_type>()
            .pretty_name()));
  }
};
}

template <unsigned D, typename... Ts>
void args_to_index(optional_index&, const dynamic_axes<Ts...>&) {}

template <unsigned D, typename... Ts, typename U, typename... Us>
void args_to_index(optional_index& idx, const dynamic_axes<Ts...>& axes,
                   const U& u, const Us&... us) {
  boost::apply_visitor(args_to_index_visitor<rm_cv_ref<U>>(idx, u), axes[D]);
  args_to_index<(D + 1)>(idx, axes, us...);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
