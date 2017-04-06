// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGARM_AXIS_VISITOR_HPP_
#define _BOOST_HISTOGARM_AXIS_VISITOR_HPP_

#include <boost/histogram/detail/utility.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/bool.hpp>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

struct bins : public static_visitor<int> {
  template <typename A> int operator()(const A &a) const { return a.bins(); }
};

struct shape : public static_visitor<int> {
  template <typename A> int operator()(const A &a) const { return a.shape(); }
};

struct uoflow : public static_visitor<bool> {
  template <typename A> bool operator()(const A &a) const { return a.uoflow(); }
};

template <typename V> struct index : public static_visitor<int> {
  const V v;
  explicit index(const V x) : v(x) {}
  template <typename A> int operator()(const A &a) const { return a.index(v); }
};

struct left : public static_visitor<double> {
  const int i;
  explicit left(const int x) : i(x) {}
  template <typename A> double operator()(const A &a) const { return a[i]; }
};

struct right : public static_visitor<double> {
  const int i;
  explicit right(const int x) : i(x) {}
  template <typename A> double operator()(const A &a) const { return a[i + 1]; }
};

struct center : public static_visitor<double> {
  const int i;
  explicit center(const int x) : i(x) {}
  template <typename A> double operator()(const A &a) const {
    return 0.5 * (a[i] + a[i + 1]);
  }
};

template <typename V>
struct cmp_axis : public static_visitor<bool> {
  const V& lhs;
  cmp_axis(const V& v) : lhs(v) {}
  template <typename T>
  bool operator()(const T& rhs) const {
    return impl(typename mpl::contains<typename V::types, T>::type(), rhs);
  }

  template <typename U>
  bool impl(mpl::true_, const U& rhs) const {
    auto pt = boost::get<U>(&lhs);
    return pt && *pt == rhs;
  }

  template <typename U>
  bool impl(mpl::false_, const U&) const {
    return false;
  }
};

template <typename V>
struct assign_axis : public static_visitor<void> {
  V& lhs;
  assign_axis(V& v) : lhs(v) {}

  template <typename U>
  void operator()(const U& rhs) const {
    impl(typename mpl::contains<typename V::types, U>::type(), rhs);
  }

  template <typename U>
  void impl(mpl::true_, const U& rhs) const {
    lhs = rhs;
  }

  template <typename U>
  void impl(mpl::false_, const U&) const {}

};

template <typename Iterator>
struct fusion_cmp_axis {
  mutable bool is_equal;
  mutable Iterator iter;
  fusion_cmp_axis(Iterator it) : is_equal(true), iter(it) {}
  template <typename T>
  void operator()(const T& t) const {
    auto pt = boost::get<T>(&(*iter));
    is_equal &= (pt && *pt == t);
    ++iter;
  }
};

template <typename Iterator>
struct fusion_assign_axis {
  mutable Iterator iter;
  fusion_assign_axis(Iterator it) : iter(it) {}
  template <typename T>
  void operator()(T& t) const {
    t = boost::get<const T&>(*(iter++));
  }
};

template <typename Iterator>
struct fusion_assign_axis2 {
  mutable Iterator iter;
  fusion_assign_axis2(Iterator it) : iter(it) {}
  template <typename T>
  void operator()(const T& t) const {
    *(iter++) == t;
  }
};

struct field_count : public static_visitor<void> {
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

template <typename... A, typename... B>
inline bool axes_equal(
  const std::vector<boost::variant<A...>> &a,
  const std::vector<boost::variant<B...>> &b)
{
  auto n = b.size();
  if (a.size() != n) {
    return false;
  }
  for (decltype(n) i = 0; i < n; ++i) {
    if (!apply_visitor(cmp_axis<boost::variant<A...>>(a[i]), b[i])) {
      return false;
    }
  }
  return true;
}

template <typename... A>
inline bool axes_equal(const fusion::vector<A...>& a, const fusion::vector<A...>& b) {
  return a == b;
}

template <typename... A, typename... B>
constexpr bool axes_equal(const fusion::vector<A...>& a, const fusion::vector<B...>& b) {
  return false;
}

template <typename... A, typename... B>
inline bool axes_equal(const std::vector<boost::variant<A...>>& a,
                const fusion::vector<B...>& b)
{
  if (a.size() != fusion::size(b))
    return false;
  fusion_cmp_axis<typename std::vector<boost::variant<A...>>::const_iterator> cmp(a.begin());
  fusion::for_each(b, cmp);
  return cmp.is_equal;
}

template <typename... A, typename... B>
inline bool axes_equal(const fusion::vector<B...>& a,
                const std::vector<boost::variant<A...>>& b)
{
  return axes_equal(b, a);
}

template <typename... A, typename... B>
inline void axes_assign(std::vector<boost::variant<A...>> &a, const std::vector<boost::variant<B...>> &b) {
  const unsigned n = b.size();
  a.resize(n);
  for (unsigned i = 0; i < n; ++i) {
    apply_visitor(assign_axis<boost::variant<A...>>(a[i]), b[i]);
  }
}

template <typename... A>
inline void axes_assign(fusion::vector<A...>& a, const fusion::vector<A...>& b)
{
  a = b;
}

template <typename... A, typename... B>
inline void axes_assign(fusion::vector<A...>& a, const std::vector<boost::variant<B...>>& b)
{
  const unsigned n = b.size();
  BOOST_ASSERT_MSG(fusion::size(a) == n, "cannot assign to static axes vector: number of axes does not match");
  fusion::for_each(a, fusion_assign_axis<typename std::vector<boost::variant<B...>>::const_iterator>(b.begin()));
}

template <typename... A, typename... B>
inline void axes_assign(std::vector<boost::variant<B...>>& a, const fusion::vector<A...>& b)
{
  a.resize(fusion::size(b));
  fusion::for_each(b, fusion_assign_axis2<typename std::vector<boost::variant<B...>>::iterator>(b.begin()));
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
