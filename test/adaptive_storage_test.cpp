// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#ifndef BOOST_HISTOGRAM_NO_SERIALIZATION
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/histogram/serialization.hpp>
#endif
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/operators.hpp>
#include <limits>
#include <memory>
#include <sstream>

using adaptive_storage_type = boost::histogram::adaptive_storage<>;

using namespace boost::histogram;

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
  a.increase(0);
  BOOST_TEST(!(a == b));
  a = b;
  BOOST_TEST(a == b);
  a.increase(0);
  BOOST_TEST(!(a == b));
  a = b;
  a = prepare<T>(2);
  BOOST_TEST(!(a == b));
  a = b;
  BOOST_TEST(a == b);
}

#ifndef BOOST_HISTOGRAM_NO_SERIALIZATION
template <typename T>
void serialization_impl() {
  const auto a = prepare(1, T(1));
  std::ostringstream os;
  std::string buf;
  {
    std::ostringstream os;
    boost::archive::text_oarchive oa(os);
    oa << a;
    buf = os.str();
  }
  adaptive_storage_type b;
  BOOST_TEST(!(a == b));
  {
    std::istringstream is(buf);
    boost::archive::text_iarchive ia(is);
    ia >> b;
  }
  BOOST_TEST(a == b);
}

template <>
void serialization_impl<void>() {
  const auto a = prepare<void>(1);
  std::ostringstream os;
  std::string buf;
  {
    std::ostringstream os2;
    boost::archive::text_oarchive oa(os2);
    oa << a;
    buf = os2.str();
  }
  adaptive_storage_type b;
  BOOST_TEST(!(a == b));
  {
    std::istringstream is(buf);
    boost::archive::text_iarchive ia(is);
    ia >> b;
  }
  BOOST_TEST(a == b);
}
#endif

template <typename T>
void equal_impl() {
  auto a = prepare<void>(1);
  auto b = prepare(1, T(0));
  BOOST_TEST_EQ(a[0].value(), 0.0);
  BOOST_TEST_EQ(a[0].variance(), 0.0);
  BOOST_TEST(a == b);
  b.increase(0);
  BOOST_TEST(!(a == b));

  array_storage<unsigned> c;
  c.reset(1);
  auto d = prepare(1, T(0));
  BOOST_TEST(c == d);
  c.increase(0);
  BOOST_TEST(!(c == d));
}

template <>
void equal_impl<void>() {
  auto a = prepare<void>(1);
  auto b = prepare<uint8_t>(1, 0);
  auto c = prepare<uint8_t>(2, 0);
  auto d = prepare<unsigned>(1);
  BOOST_TEST_EQ(a[0].value(), 0.0);
  BOOST_TEST_EQ(a[0].variance(), 0.0);
  BOOST_TEST(a == b);
  BOOST_TEST(b == a);
  BOOST_TEST(a == d);
  BOOST_TEST(d == a);
  BOOST_TEST(!(a == c));
  BOOST_TEST(!(c == a));
  b.increase(0);
  BOOST_TEST(!(a == b));
  BOOST_TEST(!(b == a));
  d.increase(0);
  BOOST_TEST(!(a == d));
  BOOST_TEST(!(d == a));
}

template <typename T>
void increase_and_grow_impl() {
  auto tmax = std::numeric_limits<T>::max();
  auto s = prepare(2, tmax);
  auto n = s;
  auto n2 = s;

  n.increase(0);

  auto x = prepare<void>(2);
  x.increase(0);
  n2.add(0, x[0].value());

  double v = tmax;
  ++v;
  BOOST_TEST_EQ(n[0].value(), v);
  BOOST_TEST_EQ(n2[0].value(), v);
  BOOST_TEST_EQ(n[1].value(), 0.0);
  BOOST_TEST_EQ(n2[1].value(), 0.0);
}

template <>
void increase_and_grow_impl<void>() {
  auto s = prepare<void>(2);
  BOOST_TEST_EQ(s[0].value(), 0);
  BOOST_TEST_EQ(s[1].value(), 0);
  s.increase(0);
  BOOST_TEST_EQ(s[0].value(), 1);
  BOOST_TEST_EQ(s[1].value(), 0);
}

template <typename T>
void convert_array_storage_impl() {
  const auto aref = prepare(1, T(0));
  array_storage<uint8_t> s;
  s.reset(1);
  s.increase(0);

  auto a = aref;
  a = s;
  BOOST_TEST_EQ(a[0].value(), 1.0);
  BOOST_TEST(a == s);
  a.increase(0);
  BOOST_TEST(!(a == s));

  adaptive_storage_type b(s);
  BOOST_TEST_EQ(b[0].value(), 1.0);
  BOOST_TEST(b == s);
  b.increase(0);
  BOOST_TEST(!(b == s));

  auto c = aref;
  c.add(0, s[0]);
  BOOST_TEST_EQ(c[0].value(), 1.0);
  BOOST_TEST(c == s);
  BOOST_TEST(s == c);

  array_storage<float> t;
  t.reset(1);
  t.increase(0);
  while (t[0] < 1e20) t.add(0, t[0]);
  auto d = aref;
  d = t;
  BOOST_TEST(d == t);

  auto e = aref;
  e = s;
  BOOST_TEST_EQ(e[0].value(), 1.0);
  BOOST_TEST(e == s);
  e.increase(0);
  BOOST_TEST(!(e == s));

  adaptive_storage_type f(s);
  BOOST_TEST_EQ(f[0].value(), 1.0);
  BOOST_TEST(f == s);
  f.increase(0);
  BOOST_TEST(!(f == s));

  auto g = aref;
  g.add(0, s[0]);
  BOOST_TEST_EQ(g[0].value(), 1.0);
  BOOST_TEST(g == s);
  BOOST_TEST(s == g);

  array_storage<uint8_t> u;
  u.reset(2);
  u.increase(0);
  auto h = aref;
  BOOST_TEST(!(h == u));
  h = u;
  BOOST_TEST(h == u);
}

template <>
void convert_array_storage_impl<void>() {
  const auto aref = prepare<void>(1);
  BOOST_TEST_EQ(aref[0].value(), 0.0);
  array_storage<uint8_t> s;
  s.reset(1);
  s.increase(0);

  auto a = aref;
  a = s;
  BOOST_TEST_EQ(a[0].value(), 1.0);
  BOOST_TEST(a == s);
  a.increase(0);
  BOOST_TEST(!(a == s));

  auto c = aref;
  c.add(0, s[0]);
  BOOST_TEST_EQ(c[0].value(), 1.0);
  BOOST_TEST(c == s);
  BOOST_TEST(s == c);

  array_storage<uint8_t> t;
  t.reset(2);
  t.increase(0);
  auto d = aref;
  BOOST_TEST(!(d == t));
}

template <typename LHS, typename RHS>
void add_impl() {
  auto a = prepare<LHS>(2);
  auto b = prepare<RHS>(2);
  if (std::is_same<RHS, void>::value) {
    a += b;
    BOOST_TEST_EQ(a[0].value(), 0);
    BOOST_TEST_EQ(a[1].value(), 0);
  } else {
    b.increase(0);
    b.increase(0);
    a += b;
    BOOST_TEST_EQ(a[0].value(), 2);
    BOOST_TEST_EQ(a[0].variance(), 2);
    BOOST_TEST_EQ(a[1].value(), 0);
  }
}

template <typename LHS>
void add_impl_all_rhs() {
  add_impl<LHS, void>();
  add_impl<LHS, uint8_t>();
  add_impl<LHS, uint16_t>();
  add_impl<LHS, uint32_t>();
  add_impl<LHS, uint64_t>();
  add_impl<LHS, detail::mp_int>();
  add_impl<LHS, detail::wcount>();
}

int main() {
  // low-level tools
  {
    uint8_t c = 0;
    BOOST_TEST_EQ(detail::safe_increase(c), true);
    BOOST_TEST_EQ(c, 1);
    c = 255;
    BOOST_TEST_EQ(detail::safe_increase(c), false);
    BOOST_TEST_EQ(c, 255);
    BOOST_TEST_EQ(detail::safe_assign(c, 255), true);
    BOOST_TEST_EQ(detail::safe_assign(c, 256), false);
    BOOST_TEST_EQ(c, 255);
    c = 0;
    BOOST_TEST_EQ(detail::safe_radd(c, 255), true);
    BOOST_TEST_EQ(c, 255);
    c = 1;
    BOOST_TEST_EQ(detail::safe_radd(c, 255), false);
    BOOST_TEST_EQ(c, 1);
    c = 255;
    BOOST_TEST_EQ(detail::safe_radd(c, 1), false);
    BOOST_TEST_EQ(c, 255);
  }

  // empty state
  {
    adaptive_storage_type a;
    BOOST_TEST_EQ(a.size(), 0);
  }

  // copy
  {
    copy_impl<detail::wcount>();
    copy_impl<void>();
    copy_impl<uint8_t>();
    copy_impl<uint16_t>();
    copy_impl<uint32_t>();
    copy_impl<uint64_t>();
    copy_impl<detail::mp_int>();
  }

  // equal_operator
  {
    equal_impl<void>();
    equal_impl<uint8_t>();
    equal_impl<uint16_t>();
    equal_impl<uint32_t>();
    equal_impl<uint64_t>();
    equal_impl<detail::mp_int>();
    equal_impl<detail::wcount>();
  }

  // increase_and_grow
  {
    increase_and_grow_impl<void>();
    increase_and_grow_impl<uint8_t>();
    increase_and_grow_impl<uint16_t>();
    increase_and_grow_impl<uint32_t>();
    increase_and_grow_impl<uint64_t>();

    // only increase for mp_int
    auto a = prepare<detail::mp_int>(2, 1);
    BOOST_TEST_EQ(a[0].value(), 1);
    BOOST_TEST_EQ(a[1].value(), 0);
    a.increase(0);
    BOOST_TEST_EQ(a[0].value(), 2);
    BOOST_TEST_EQ(a[1].value(), 0);
  }

  // add
  {
    add_impl_all_rhs<void>();
    add_impl_all_rhs<uint8_t>();
    add_impl_all_rhs<uint16_t>();
    add_impl_all_rhs<uint32_t>();
    add_impl_all_rhs<uint64_t>();
    add_impl_all_rhs<detail::mp_int>();
    add_impl_all_rhs<detail::wcount>();
  }

  // add_and_grow
  {
    auto a = prepare<void>(1);
    a += a;
    BOOST_TEST_EQ(a[0].value(), 0);
    a.increase(0);
    double x = 1;
    auto b = prepare<void>(1);
    b.increase(0);
    BOOST_TEST_EQ(b[0].value(), x);
    for (unsigned i = 0; i < 80; ++i) {
      x += x;
      a.add(0, a[0].value());
      b += b;
      BOOST_TEST_EQ(a[0].value(), x);
      BOOST_TEST_EQ(a[0].variance(), x);
      BOOST_TEST_EQ(b[0].value(), x);
      BOOST_TEST_EQ(b[0].variance(), x);
      auto c = prepare<void>(1);
      c.add(0, a[0].value());
      BOOST_TEST_EQ(c[0].value(), x);
      BOOST_TEST_EQ(c[0].variance(), x);
      c.add(0, weight(0));
      BOOST_TEST_EQ(c[0].value(), x);
      BOOST_TEST_EQ(c[0].variance(), x);
      auto d = prepare<void>(1);
      d.add(0, weight(x));
      BOOST_TEST_EQ(d[0].value(), x);
      BOOST_TEST_EQ(d[0].variance(), x * x);
    }
  }

  // multiply
  {
    auto a = prepare<void>(2);
    a *= 2;
    BOOST_TEST_EQ(a[0].value(), 0);
    BOOST_TEST_EQ(a[1].value(), 0);
    a.increase(0);
    a *= 3;
    BOOST_TEST_EQ(a[0].value(), 3);
    BOOST_TEST_EQ(a[0].variance(), 9);
    BOOST_TEST_EQ(a[1].value(), 0);
    BOOST_TEST_EQ(a[1].variance(), 0);
    a.add(1, adaptive_storage_type::element_type(2, 5));
    BOOST_TEST_EQ(a[0].value(), 3);
    BOOST_TEST_EQ(a[0].variance(), 9);
    BOOST_TEST_EQ(a[1].value(), 2);
    BOOST_TEST_EQ(a[1].variance(), 5);
    a *= 3;
    BOOST_TEST_EQ(a[0].value(), 9);
    BOOST_TEST_EQ(a[0].variance(), 81);
    BOOST_TEST_EQ(a[1].value(), 6);
    BOOST_TEST_EQ(a[1].variance(), 45);
  }

  // convert_array_storage
  {
    convert_array_storage_impl<void>();
    convert_array_storage_impl<uint8_t>();
    convert_array_storage_impl<uint16_t>();
    convert_array_storage_impl<uint32_t>();
    convert_array_storage_impl<uint64_t>();
    convert_array_storage_impl<detail::mp_int>();
    convert_array_storage_impl<detail::wcount>();
  }

#ifndef BOOST_HISTOGRAM_NO_SERIALIZATION
  // serialization_test
  {
    serialization_impl<void>();
    serialization_impl<uint8_t>();
    serialization_impl<uint16_t>();
    serialization_impl<uint32_t>();
    serialization_impl<uint64_t>();
    serialization_impl<detail::mp_int>();
    serialization_impl<detail::wcount>();
  }
#endif

  return boost::report_errors();
}
