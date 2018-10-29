// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/adaptive_storage.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

namespace bh = boost::histogram;
using adaptive_storage_type = bh::adaptive_storage<>;
template <typename T>
using array_storage = bh::storage_adaptor<std::vector<T>>;
using bh::weight;

template <typename T>
adaptive_storage_type prepare(std::size_t n, const T x) {
  std::unique_ptr<T[]> v(new T[n]);
  std::fill(v.get(), v.get() + n, static_cast<T>(0));
  v.get()[0] = x;
  return adaptive_storage_type(n, v.get());
}

template <typename T>
adaptive_storage_type prepare(std::size_t n) {
  return adaptive_storage_type(n, static_cast<T*>(nullptr));
}

template <typename T>
void copy_impl() {
  const auto b = prepare<T>(1);
  auto a(b);
  BOOST_TEST(a == b);
  a(0);
  BOOST_TEST(!(a == b));
  a = b;
  BOOST_TEST(a == b);
  a(0);
  BOOST_TEST(!(a == b));
  a = b;
  a = prepare<T>(2);
  BOOST_TEST(!(a == b));
  a = b;
  BOOST_TEST(a == b);
}

template <typename T>
void equal_1_impl() {
  auto a = prepare<void>(1);
  auto b = prepare(1, T(0));
  BOOST_TEST_EQ(a[0], 0.0);
  BOOST_TEST(a == b);
  b(0);
  BOOST_TEST(!(a == b));
}

template <>
void equal_1_impl<void>() {
  auto a = prepare<void>(1);
  auto b = prepare<uint8_t>(1, 0);
  auto c = prepare<uint8_t>(2, 0);
  auto d = prepare<unsigned>(1);
  BOOST_TEST_EQ(a[0], 0.0);
  BOOST_TEST(a == b);
  BOOST_TEST(b == a);
  BOOST_TEST(a == d);
  BOOST_TEST(d == a);
  BOOST_TEST(!(a == c));
  BOOST_TEST(!(c == a));
  b(0);
  BOOST_TEST(!(a == b));
  BOOST_TEST(!(b == a));
  d(0);
  BOOST_TEST(!(a == d));
  BOOST_TEST(!(d == a));
}

template <typename T, typename U>
void equal_2_impl() {
  auto a = prepare<T>(1);
  array_storage<U> b;
  b.reset(1);
  BOOST_TEST(a == b);
  b(0);
  BOOST_TEST(!(a == b));
}

template <typename T>
void increase_and_grow_impl() {
  auto tmax = std::numeric_limits<T>::max();
  auto s = prepare(2, tmax);
  auto n = s;
  auto n2 = s;

  n(0);

  auto x = prepare<void>(2);
  x(0);
  n2(0, x[0]);

  double v = tmax;
  ++v;
  BOOST_TEST_EQ(n[0], v);
  BOOST_TEST_EQ(n2[0], v);
  BOOST_TEST_EQ(n[1], 0.0);
  BOOST_TEST_EQ(n2[1], 0.0);
}

template <>
void increase_and_grow_impl<void>() {
  auto s = prepare<void>(2);
  BOOST_TEST_EQ(s[0], 0);
  BOOST_TEST_EQ(s[1], 0);
  s(0);
  BOOST_TEST_EQ(s[0], 1);
  BOOST_TEST_EQ(s[1], 0);
}

template <typename T>
void convert_array_storage_impl() {
  const auto aref = prepare(1, T(0));
  array_storage<uint8_t> s;
  s.reset(1);
  s(0);

  auto a = aref;
  a = s;
  BOOST_TEST_EQ(a[0], 1.0);
  BOOST_TEST(a == s);
  a(0);
  BOOST_TEST(!(a == s));

  adaptive_storage_type b(s);
  BOOST_TEST_EQ(b[0], 1.0);
  BOOST_TEST(b == s);
  b(0);
  BOOST_TEST(!(b == s));

  auto c = aref;
  c(0, s[0]);
  BOOST_TEST_EQ(c[0], 1.0);
  BOOST_TEST(c == s);
  BOOST_TEST(s == c);

  array_storage<float> t;
  t.reset(1);
  t(0);
  while (t[0] < 1e20) t(0, t[0]);
  auto d = aref;
  d = t;
  BOOST_TEST(d == t);

  auto e = aref;
  e = s;
  BOOST_TEST_EQ(e[0], 1.0);
  BOOST_TEST(e == s);
  e(0);
  BOOST_TEST(!(e == s));

  adaptive_storage_type f(s);
  BOOST_TEST_EQ(f[0], 1.0);
  BOOST_TEST(f == s);
  f(0);
  BOOST_TEST(!(f == s));

  auto g = aref;
  g(0, s[0]);
  BOOST_TEST_EQ(g[0], 1.0);
  BOOST_TEST(g == s);
  BOOST_TEST(s == g);

  array_storage<uint8_t> u;
  u.reset(2);
  u(0);
  auto h = aref;
  BOOST_TEST(!(h == u));
  h = u;
  BOOST_TEST(h == u);
}

template <>
void convert_array_storage_impl<void>() {
  const auto aref = prepare<void>(1);
  BOOST_TEST_EQ(aref[0], 0.0);
  array_storage<uint8_t> s;
  s.reset(1);
  s(0);

  auto a = aref;
  a = s;
  BOOST_TEST_EQ(a[0], 1.0);
  BOOST_TEST(a == s);
  a(0);
  BOOST_TEST(!(a == s));

  auto c = aref;
  c(0, s[0]);
  BOOST_TEST_EQ(c[0], 1.0);
  BOOST_TEST(c == s);
  BOOST_TEST(s == c);

  array_storage<uint8_t> t;
  t.reset(2);
  t(0);
  auto d = aref;
  BOOST_TEST(!(d == t));
}

template <typename LHS, typename RHS>
void add_impl() {
  auto a = prepare<LHS>(2);
  auto b = prepare<RHS>(2);
  if (std::is_same<RHS, void>::value) {
    a += b;
    BOOST_TEST_EQ(a[0], 0);
    BOOST_TEST_EQ(a[1], 0);
  } else {
    b(0);
    b(0);
    a += b;
    BOOST_TEST_EQ(a[0], 2);
    BOOST_TEST_EQ(a[1], 0);
  }
}

template <typename LHS>
void add_impl_all_rhs() {
  add_impl<LHS, void>();
  add_impl<LHS, uint8_t>();
  add_impl<LHS, uint16_t>();
  add_impl<LHS, uint32_t>();
  add_impl<LHS, uint64_t>();
  add_impl<LHS, adaptive_storage_type::mp_int>();
  add_impl<LHS, double>();
}

int main() {
  // low-level tools
  {
    uint8_t c = 0;
    BOOST_TEST_EQ(bh::detail::safe_increment(c), true);
    BOOST_TEST_EQ(c, 1);
    c = 255;
    BOOST_TEST_EQ(bh::detail::safe_increment(c), false);
    BOOST_TEST_EQ(c, 255);
    BOOST_TEST_EQ(bh::detail::safe_assign(c, 255), true);
    BOOST_TEST_EQ(bh::detail::safe_assign(c, 256), false);
    BOOST_TEST_EQ(c, 255);
    c = 0;
    BOOST_TEST_EQ(bh::detail::safe_radd(c, 255), true);
    BOOST_TEST_EQ(c, 255);
    c = 1;
    BOOST_TEST_EQ(bh::detail::safe_radd(c, 255), false);
    BOOST_TEST_EQ(c, 1);
    c = 255;
    BOOST_TEST_EQ(bh::detail::safe_radd(c, 1), false);
    BOOST_TEST_EQ(c, 255);
  }

  // empty state
  {
    adaptive_storage_type a;
    BOOST_TEST_EQ(a.size(), 0);
  }

  // copy
  {
    copy_impl<double>();
    copy_impl<void>();
    copy_impl<uint8_t>();
    copy_impl<uint16_t>();
    copy_impl<uint32_t>();
    copy_impl<uint64_t>();
    copy_impl<adaptive_storage_type::mp_int>();
  }

  // equal_operator
  {
    equal_1_impl<void>();
    equal_1_impl<uint8_t>();
    equal_1_impl<uint16_t>();
    equal_1_impl<uint32_t>();
    equal_1_impl<uint64_t>();
    equal_1_impl<adaptive_storage_type::mp_int>();
    equal_1_impl<double>();

    equal_2_impl<void, unsigned>();
    equal_2_impl<uint8_t, unsigned>();
    equal_2_impl<uint16_t, unsigned>();
    equal_2_impl<uint32_t, unsigned>();
    equal_2_impl<uint64_t, unsigned>();
    equal_2_impl<adaptive_storage_type::mp_int, unsigned>();
    equal_2_impl<double, unsigned>();

    equal_2_impl<adaptive_storage_type::mp_int, double>();

    auto a = prepare<double>(1);
    auto b = prepare<adaptive_storage_type::mp_int>(1);
    BOOST_TEST(a == b);
    a(0);
    BOOST_TEST_NOT(a == b);
  }

  // increase_and_grow
  {
    increase_and_grow_impl<void>();
    increase_and_grow_impl<uint8_t>();
    increase_and_grow_impl<uint16_t>();
    increase_and_grow_impl<uint32_t>();
    increase_and_grow_impl<uint64_t>();

    // only increase for mp_int
    auto a = prepare<adaptive_storage_type::mp_int>(2, 1);
    BOOST_TEST_EQ(a[0], 1);
    BOOST_TEST_EQ(a[1], 0);
    a(0);
    BOOST_TEST_EQ(a[0], 2);
    BOOST_TEST_EQ(a[1], 0);
  }

  // add
  {
    add_impl_all_rhs<void>();
    add_impl_all_rhs<uint8_t>();
    add_impl_all_rhs<uint16_t>();
    add_impl_all_rhs<uint32_t>();
    add_impl_all_rhs<uint64_t>();
    add_impl_all_rhs<adaptive_storage_type::mp_int>();
    add_impl_all_rhs<double>();
  }

  // add_and_grow
  {
    auto a = prepare<void>(1);
    a += a;
    BOOST_TEST_EQ(a[0], 0);
    a(0);
    double x = 1;
    auto b = prepare<void>(1);
    b(0);
    BOOST_TEST_EQ(b[0], x);
    for (unsigned i = 0; i < 80; ++i) {
      x += x;
      a(0, a[0]);
      b += b;
      BOOST_TEST_EQ(a[0], x);
      BOOST_TEST_EQ(b[0], x);
      auto c = prepare<void>(1);
      c(0, a[0]);
      BOOST_TEST_EQ(c[0], x);
      c(0, weight(0));
      BOOST_TEST_EQ(c[0], x);
      auto d = prepare<void>(1);
      d(0, weight(x));
      BOOST_TEST_EQ(d[0], x);
    }
  }

  // multiply
  {
    auto a = prepare<void>(2);
    a *= 2;
    BOOST_TEST_EQ(a[0], 0);
    BOOST_TEST_EQ(a[1], 0);
    a(0);
    a *= 3;
    BOOST_TEST_EQ(a[0], 3);
    BOOST_TEST_EQ(a[1], 0);
    a(1, 2);
    BOOST_TEST_EQ(a[0], 3);
    BOOST_TEST_EQ(a[1], 2);
    a *= 3;
    BOOST_TEST_EQ(a[0], 9);
    BOOST_TEST_EQ(a[1], 6);
  }

  // convert_array_storage
  {
    convert_array_storage_impl<void>();
    convert_array_storage_impl<uint8_t>();
    convert_array_storage_impl<uint16_t>();
    convert_array_storage_impl<uint32_t>();
    convert_array_storage_impl<uint64_t>();
    convert_array_storage_impl<adaptive_storage_type::mp_int>();
    convert_array_storage_impl<double>();
  }

  return boost::report_errors();
}
