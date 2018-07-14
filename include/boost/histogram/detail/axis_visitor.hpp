// Copyright 2015-2017 Hans Demsizeki
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGARM_AXIS_VISITOR_HPP_
#define _BOOST_HISTOGARM_AXIS_VISITOR_HPP_

#include <tuple>
#include <vector>
#include <boost/mp11.hpp>
#include <boost/variant/get.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/variant.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <type_traits>


namespace boost {
namespace histogram {
namespace detail {

namespace {
template <typename Variant> struct axes_equal_visitor : public static_visitor<bool> {
  const Variant &lhs;
  axes_equal_visitor(const Variant &v) : lhs(v) {}
  template <typename T> bool operator()(const T &rhs) const {
    return impl(mp11::mp_contains<typename Variant::types, T>(), rhs);
  }

  template <typename T> bool impl(mp11::mp_true, const T &rhs) const {
    auto tp = boost::get<T>(&lhs);
    return tp && *tp == rhs;
  }

  template <typename T> bool impl(mp11::mp_false, const T &) const {
    return false;
  }
};

template <typename V> struct axes_assign_visitor : public static_visitor<void> {
  V &lhs;
  axes_assign_visitor(V &v) : lhs(v) {}

  template <typename U> void operator()(const U &rhs) const {
    impl(mp11::mp_contains<typename V::types, U>(), rhs);
  }

  template <typename U> void impl(mp11::mp_true, const U &rhs) const { lhs = rhs; }

  template <typename U> void impl(mp11::mp_false, const U &) const {
    BOOST_ASSERT_MSG(false, "cannot assign U to variant if it is not a bounded type");
  }
};

template <typename Tuple, typename VecVar>
struct axes_equal_tuple_vecvar {
  bool& equal; 
  const Tuple& t;
  const VecVar& v;
  axes_equal_tuple_vecvar(bool& eq, const Tuple& tt, const VecVar& vv) :
    equal(eq), t(tt), v(vv) {}
  template <typename Int> void operator()(Int) const {
    using T = mp11::mp_at<Tuple, Int>;
    auto tp = ::boost::get<T>(&v[Int::value]);
    equal &= (tp && *tp == std::get<Int::value>(t));
  }
};

template <typename Tuple, typename VecVar>
struct axes_assign_tuple_vecvar {
  Tuple& t;
  const VecVar& v;
  axes_assign_tuple_vecvar(Tuple& tt, const VecVar& vv) : t(tt), v(vv) {}
  template <typename Int> void operator()(Int) const {
    using T = mp11::mp_at<Tuple, Int>;
    std::get<Int::value>(t) = ::boost::get<T>(v[Int::value]);
  }
};

template <typename VecVar, typename Tuple>
struct axes_assign_vecvar_tuple {
  VecVar& v;
  const Tuple& t;
  axes_assign_vecvar_tuple(VecVar& vv, const Tuple& tt) : v(vv), t(tt) {}
  template <typename Int> void operator()(Int) const {
    v[Int::value] = std::get<Int::value>(t);
  }
};

}

struct field_count_visitor : public static_visitor<void> {
  mutable std::size_t value = 1;
  template <typename T> void operator()(const T &t) const {
    value *= t.shape();
  }
};

template <typename Unary> struct unary_visitor : public static_visitor<void> {
  Unary &unary;
  unary_visitor(Unary &u) : unary(u) {}
  template <typename Axis> void operator()(const Axis &a) const { unary(a); }
};

namespace {

template <typename... Ts>
bool axes_equal_impl(mp11::mp_true, const std::tuple<Ts...>& t,
                               const std::tuple<Ts...>& u) {
  return t == u;
}

template <typename... Ts, typename... Us>
bool axes_equal_impl(mp11::mp_false, const std::tuple<Ts...>&,
                                const std::tuple<Us...>&) {
  return false;
}

}

template <typename... Ts, typename... Us>
bool axes_equal(const std::tuple<Ts...>& t,
                const std::tuple<Us...>& u)
{
  return axes_equal_impl(mp11::mp_same<mp11::mp_list<Ts...>, mp11::mp_list<Us...>>(), t, u);
}

template <typename... Ts, typename... Us>
void axes_assign(std::tuple<Ts...>& t,
                 const std::tuple<Us...>& u) {
  static_assert(std::is_same<mp11::mp_list<Ts...>,
                             mp11::mp_list<Us...>>::value,
                "cannot assign incompatible axes");
  t = u;
}

template <typename... Ts, typename... Us>
bool axes_equal(const std::tuple<Ts...>& t,
                const std::vector<axis::any<Us...>>& u)
{
  if (sizeof...(Ts) != u.size())
    return false;
  bool equal = true;
  auto fn = axes_equal_tuple_vecvar<
    std::tuple<Ts...>,
    std::vector<axis::any<Us...>>
  >(equal, t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>(fn);
  return equal;
}

template <typename... Ts, typename... Us>
void axes_assign(std::tuple<Ts...>& t,
                 const std::vector<axis::any<Us...>>& u) {
  auto fn = axes_assign_tuple_vecvar<
    std::tuple<Ts...>,
    std::vector<axis::any<Us...>>
  >(t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Ts)>>(fn);
}

template <typename... Ts, typename... Us>
bool axes_equal(const std::vector<axis::any<Ts...>>& t,
                const std::tuple<Us...>& u)
{
  return axes_equal(u, t);
}

template <typename... Ts, typename... Us>
void axes_assign(std::vector<axis::any<Ts...>>& t,
                 const std::tuple<Us...>& u) {
  t.resize(sizeof...(Us));
  auto fn = axes_assign_vecvar_tuple<
    std::vector<axis::any<Ts...>>,
    std::tuple<Us...>
  >(t, u);
  mp11::mp_for_each<mp11::mp_iota_c<sizeof...(Us)>>(fn);
}

template <typename... Ts, typename... Us>
bool axes_equal(const std::vector<axis::any<Ts...>>& t,
                const std::vector<axis::any<Us...>>& u)
{
  if (t.size() != u.size())
    return false;
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (!apply_visitor(axes_equal_visitor<axis::any<Ts...>>(t[i]), u[i]))
      return false;
  }
  return true;
}

template <typename... Ts, typename... Us>
void axes_assign(std::vector<axis::any<Ts...>>& t,
                 const std::vector<axis::any<Us...>>& u) {
  for (std::size_t i = 0; i < t.size(); ++i) {
    apply_visitor(axes_assign_visitor<axis::any<Ts...>>(t[i]), u[i]);
  }
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
