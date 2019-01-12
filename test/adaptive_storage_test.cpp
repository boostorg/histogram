// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
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
using vector_storage = bh::storage_adaptor<std::vector<T>>;

template <typename T = std::uint8_t>
adaptive_storage_type prepare(std::size_t n, T x = T()) {
  std::unique_ptr<T[]> v(new T[n]);
  std::fill(v.get(), v.get() + n, static_cast<T>(0));
  v.get()[0] = x;
  return adaptive_storage_type(n, v.get());
}

template <typename T>
void copy() {
  const auto b = prepare<T>(1);
  auto a(b);
  BOOST_TEST(a == b);
  ++a[0];
  BOOST_TEST(!(a == b));
  a = b;
  BOOST_TEST(a == b);
  ++a[0];
  BOOST_TEST(!(a == b));
  a = prepare<T>(2);
  BOOST_TEST(!(a == b));
  a = b;
  BOOST_TEST(a == b);
}

template <typename T>
void equal_1() {
  auto a = prepare(1);
  auto b = prepare(1, T(0));
  BOOST_TEST_EQ(a[0], 0.0);
  BOOST_TEST(a == b);
  ++b[0];
  BOOST_TEST(!(a == b));
}

template <typename T, typename U>
void equal_2() {
  auto a = prepare<T>(1);
  vector_storage<U> b;
  b.reset(1);
  BOOST_TEST(a == b);
  ++b[0];
  BOOST_TEST(!(a == b));
}

template <typename T>
void increase_and_grow() {
  auto tmax = std::numeric_limits<T>::max();
  auto s = prepare(2, tmax);
  auto n = s;
  auto n2 = s;

  ++n[0];

  auto x = prepare(2);
  ++x[0];
  n2[0] += x[0];

  double v = tmax;
  ++v;
  BOOST_TEST_EQ(n[0], v);
  BOOST_TEST_EQ(n2[0], v);
  BOOST_TEST_EQ(n[1], 0.0);
  BOOST_TEST_EQ(n2[1], 0.0);
}

template <typename T>
void convert_array_storage() {
  const auto aref = prepare<T>(1);
  vector_storage<uint8_t> s;
  s.reset(1);
  ++s[0];

  auto a(aref);
  a = s;
  BOOST_TEST_EQ(a[0], 1.0);
  BOOST_TEST(a == s);
  ++a[0];
  BOOST_TEST(!(a == s));

  adaptive_storage_type b(s);
  BOOST_TEST_EQ(b[0], 1.0);
  BOOST_TEST(b == s);
  ++b[0];
  BOOST_TEST(!(b == s));

  auto c = aref;
  c[0] += s[0];
  BOOST_TEST_EQ(c[0], 1.0);
  BOOST_TEST(c == s);

  vector_storage<float> t;
  t.reset(1);
  ++t[0];
  while (t[0] < 1e20) t[0] += t[0];
  auto d(aref);
  d = t;
  BOOST_TEST(d == t);

  auto e(aref);
  e = s;
  BOOST_TEST_EQ(e[0], 1.0);
  BOOST_TEST(e == s);
  ++e[0];
  BOOST_TEST(!(e == s));

  adaptive_storage_type f(s);
  BOOST_TEST_EQ(f[0], 1.0);
  BOOST_TEST(f == s);
  ++f[0];
  BOOST_TEST(!(f == s));

  auto g = aref;
  g[0] += s[0];
  BOOST_TEST_EQ(g[0], 1.0);
  BOOST_TEST(g == s);

  vector_storage<uint8_t> u;
  u.reset(2);
  ++u[0];
  auto h = aref;
  BOOST_TEST(!(h == u));
  h = u;
  BOOST_TEST(h == u);
}

template <typename LHS, typename RHS>
void add() {
  auto a = prepare<LHS>(2);
  auto b = prepare<RHS>(2);
  b[0] += 2;
  a += b;
  BOOST_TEST_EQ(a[0], 2);
  BOOST_TEST_EQ(a[1], 0);
}

template <typename LHS>
void add_all_rhs() {
  add<LHS, uint8_t>();
  add<LHS, uint16_t>();
  add<LHS, uint32_t>();
  add<LHS, uint64_t>();
  add<LHS, adaptive_storage_type::mp_int>();
  add<LHS, double>();
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
    copy<uint8_t>();
    copy<uint16_t>();
    copy<uint32_t>();
    copy<uint64_t>();
    copy<adaptive_storage_type::mp_int>();
    copy<double>();
  }

  // equal_operator
  {
    equal_1<uint8_t>();
    equal_1<uint16_t>();
    equal_1<uint32_t>();
    equal_1<uint64_t>();
    equal_1<adaptive_storage_type::mp_int>();
    equal_1<double>();

    equal_2<uint8_t, unsigned>();
    equal_2<uint16_t, unsigned>();
    equal_2<uint32_t, unsigned>();
    equal_2<uint64_t, unsigned>();
    equal_2<adaptive_storage_type::mp_int, unsigned>();
    equal_2<double, unsigned>();

    equal_2<adaptive_storage_type::mp_int, double>();

    auto a = prepare<double>(1);
    auto b = prepare<adaptive_storage_type::mp_int>(1);
    BOOST_TEST(a == b);
    ++a[0];
    BOOST_TEST_NOT(a == b);
  }

  // increase_and_grow
  {
    increase_and_grow<uint8_t>();
    increase_and_grow<uint16_t>();
    increase_and_grow<uint32_t>();
    increase_and_grow<uint64_t>();

    // only increase for mp_int
    auto a = prepare<adaptive_storage_type::mp_int>(2, 1);
    BOOST_TEST_EQ(a[0], 1);
    BOOST_TEST_EQ(a[1], 0);
    ++a[0];
    BOOST_TEST_EQ(a[0], 2);
    BOOST_TEST_EQ(a[1], 0);
  }

  // add
  {
    add_all_rhs<uint8_t>();
    add_all_rhs<uint16_t>();
    add_all_rhs<uint32_t>();
    add_all_rhs<uint64_t>();
    add_all_rhs<adaptive_storage_type::mp_int>();
    add_all_rhs<double>();
  }

  // add_and_grow
  {
    auto a = prepare(1);
    a += a;
    BOOST_TEST_EQ(a[0], 0);
    ++a[0];
    double x = 1;
    auto b = prepare(1);
    ++b[0];
    BOOST_TEST_EQ(b[0], x);
    for (unsigned i = 0; i < 80; ++i) {
      x += x;
      a[0] += a[0];
      b += b;
      BOOST_TEST_EQ(a[0], x);
      BOOST_TEST_EQ(b[0], x);
      auto c = prepare(1);
      c[0] += a[0];
      BOOST_TEST_EQ(c[0], x);
      c[0] += 0;
      BOOST_TEST_EQ(c[0], x);
      auto d = prepare(1);
      d[0] += x;
      BOOST_TEST_EQ(d[0], x);
    }
  }

  // multiply
  {
    auto a = prepare(2);
    ++a[0];
    a *= 3;
    BOOST_TEST_EQ(a[0], 3);
    BOOST_TEST_EQ(a[1], 0);
    a[1] += 2;
    a *= 3;
    BOOST_TEST_EQ(a[0], 9);
    BOOST_TEST_EQ(a[1], 6);
  }

  // convert_array_storage
  {
    convert_array_storage<uint8_t>();
    convert_array_storage<uint16_t>();
    convert_array_storage<uint32_t>();
    convert_array_storage<uint64_t>();
    convert_array_storage<adaptive_storage_type::mp_int>();
    convert_array_storage<double>();
  }

  // iterators
  {
    auto a = prepare(2);
    for (auto&& x : a) BOOST_TEST_EQ(x, 0);

    std::vector<double> b(2, 1);
    std::copy(b.begin(), b.end(), a.begin());

    const auto aconst = a;
    BOOST_TEST(std::equal(aconst.begin(), aconst.end(), b.begin(), b.end()));

    adaptive_storage_type::iterator it1 = a.begin();
    *it1 = 3;
    adaptive_storage_type::const_iterator it2 = a.begin();
    BOOST_TEST_EQ(*it2, 3);
    adaptive_storage_type::const_iterator it3 = aconst.begin();
    BOOST_TEST_EQ(*it3, 1);
    // adaptive_storage_type::iterator it3 = aconst.begin();
  }

  // compare reference
  {
    auto a = prepare(1);
    auto b = prepare<uint32_t>(1);
    BOOST_TEST_EQ(a[0], b[0]);
    a[0] = 1;
    BOOST_TEST_NE(a[0], b[0]);
    b[0] = 1;
    BOOST_TEST_EQ(a[0], b[0]);
  }

  return boost::report_errors();
}
