// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGARM_AXIS_VISITOR_HPP_
#define _BOOST_HISTOGARM_AXIS_VISITOR_HPP_

#include <boost/histogram/detail/utility.hpp>
#include <boost/variant/static_visitor.hpp>

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

struct bicmp_axis : public static_visitor<bool> {
  template <typename T> bool operator()(const T &a, const T &b) const {
    return a == b;
  }

  template <typename T, typename U>
  bool operator()(const T & /*unused*/, const U & /*unused*/) const {
    return false;
  }
};

template <typename Variant>
struct assign_axis : public static_visitor<void> {
  Variant& variant;
  assign_axis(Variant& v) : variant(v) {}
  template <typename T> void operator()(const T& a) const {
    variant = a;
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

template <typename A, typename B> bool axes_equal(const A &a, const B& b) {
  const unsigned dim = b.size();
  if (a.size() != dim) {
    return false;
  }
  for (unsigned i = 0; i < dim; ++i) {
    if (!apply_visitor(bicmp_axis(), a[i], b[i])) {
      return false;
    }
  }
  return true;
}

template <typename A, typename B> void axes_assign(A &a, const B& b) {
  const unsigned dim = b.size();
  a.resize(dim);
  for (unsigned i = 0; i < dim; ++i) {
    apply_visitor(assign_axis<typename A::value_type>(a[i]), b[i]);
  }
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
