// Copyright 2015-2017 Hans Demsizeki
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGARM_AXIS_VISITOR_HPP_
#define _BOOST_HISTOGARM_AXIS_VISITOR_HPP_

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/comparison.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/fusion/include/is_sequence.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/fusion/support/is_sequence.hpp>
#include <boost/histogram/interval.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/variant/get.hpp>
#include <boost/variant/static_visitor.hpp>
#include <boost/variant/variant.hpp>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <typename V> struct cmp_axis : public static_visitor<bool> {
  const V &lhs;
  cmp_axis(const V &v) : lhs(v) {}
  template <typename T> bool operator()(const T &rhs) const {
    return impl(typename mpl::contains<typename V::types, T>::type(), rhs);
  }

  template <typename U> bool impl(mpl::true_, const U &rhs) const {
    auto pt = boost::get<U>(&lhs);
    return pt && *pt == rhs;
  }

  template <typename U> bool impl(mpl::false_, const U &) const {
    return false;
  }
};

template <typename V> struct assign_axis : public static_visitor<void> {
  V &lhs;
  assign_axis(V &v) : lhs(v) {}

  template <typename U> void operator()(const U &rhs) const {
    impl(typename mpl::contains<typename V::types, U>::type(), rhs);
  }

  template <typename U> void impl(mpl::true_, const U &rhs) const { lhs = rhs; }

  template <typename U> void impl(mpl::false_, const U &) const {
    // never called: cannot assign to variant if U is not a bounded type
  }
};

template <typename Iterator> struct fusion_cmp_axis {
  mutable bool is_equal;
  mutable Iterator iter;
  fusion_cmp_axis(Iterator it) : is_equal(true), iter(it) {}
  template <typename T> void operator()(const T &t) const {
    auto pt = boost::get<T>(&(*iter));
    is_equal &= (pt && *pt == t);
    ++iter;
  }
};

template <typename Iterator> struct fusion_assign_axis {
  mutable Iterator iter;
  fusion_assign_axis(Iterator it) : iter(it) {}
  template <typename T> void operator()(T &t) const {
    t = boost::get<const T &>(*(iter++));
  }
};

template <typename Iterator> struct fusion_assign_axis2 {
  mutable Iterator iter;
  fusion_assign_axis2(Iterator it) : iter(it) {}
  template <typename T> void operator()(const T &t) const { *(iter++) = t; }
};

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

template <typename A, typename B>
inline bool axes_equal_impl(mpl::false_, mpl::false_, const A &a, const B &b) {
  auto n = b.size();
  if (a.size() != n) {
    return false;
  }
  for (auto i = 0u; i < n; ++i) {
    if (!apply_visitor(cmp_axis<typename A::value_type>(a[i]), b[i])) {
      return false;
    }
  }
  return true;
}

template <typename A, typename B>
inline bool axes_equal_impl(mpl::false_, mpl::true_, const A &a, const B &b) {
  if (a.size() != fusion::size(b))
    return false;
  fusion_cmp_axis<typename A::const_iterator> cmp(a.begin());
  fusion::for_each(b, cmp);
  return cmp.is_equal;
}

template <typename A, typename B>
inline bool axes_equal_impl(mpl::true_, mpl::false_, const A &a, const B &b) {
  return axes_equal_impl(mpl::false_(), mpl::true_(), b, a);
}

template <typename A>
inline bool axes_equal_impl(mpl::true_, mpl::true_, const A &a, const A &b) {
  return a == b;
}

template <typename A, typename B>
inline bool axes_equal_impl(mpl::true_, mpl::true_, const A &, const B &) {
  return false;
}

template <typename A, typename B>
inline bool axes_equal(const A &a, const B &b) {
  return axes_equal_impl(typename fusion::traits::is_sequence<A>::type(),
                         typename fusion::traits::is_sequence<B>::type(), a, b);
}

template <typename A, typename B>
inline void axes_assign_impl(mpl::false_, mpl::false_, A &a, const B &b) {
  auto n = b.size();
  a.resize(n);
  for (decltype(n) i = 0; i < n; ++i) {
    apply_visitor(assign_axis<typename A::value_type>(a[i]), b[i]);
  }
}

template <typename A, typename B>
inline void axes_assign_impl(mpl::false_, mpl::true_, A &a, const B &b) {
  a.resize(fusion::size(b));
  fusion::for_each(b, fusion_assign_axis2<typename A::iterator>(a.begin()));
}

template <typename A, typename B>
inline void axes_assign_impl(mpl::true_, mpl::false_, A &a, const B &b) {
  BOOST_ASSERT_MSG(
      static_cast<int>(fusion::size(a)) == b.size(),
      "cannot assign to static axes vector: number of axes does not match");
  fusion::for_each(a,
                   fusion_assign_axis<typename B::const_iterator>(b.begin()));
}

template <typename A>
inline void axes_assign_impl(mpl::true_, mpl::true_, A &a, const A &b) {
  a = b;
}

template <typename A, typename B>
inline void axes_assign_impl(mpl::true_, mpl::true_, A &a, const B &b) {
  static_assert(is_same<A, B>::type::value,
                "cannot assign different static axes vectors");
}

template <typename A, typename B> inline void axes_assign(A &a, const B &b) {
  return axes_assign_impl(typename fusion::traits::is_sequence<A>::type(),
                          typename fusion::traits::is_sequence<B>::type(), a,
                          b);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
