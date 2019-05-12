// Copyright (c) 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test is inspired by the corresponding boost/beast test of detail_variant.

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/detail/throw_exception.hpp>
#include <boost/histogram/detail/variant.hpp>
#include <boost/throw_exception.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace boost::histogram::detail;
using namespace std::literals;

template <int>
struct Q {
  Q() noexcept { ++Q::count; }
  Q(const Q& q) : data(q.data) {
    if (q.data == 0xBAD) // simulate failing copy ctor
      BOOST_THROW_EXCEPTION(std::bad_alloc{});
    ++Q::count;
  }
  Q(Q&& q) noexcept {
    data = q.data;
    moved = true;
    ++Q::count;
  }
  Q& operator=(const Q& q) {
    if (q.data == 0xBAD) // simulate failing copy ctor
      BOOST_THROW_EXCEPTION(std::bad_alloc{});
    data = q.data;
    return *this;
  }
  Q& operator=(Q&& q) noexcept {
    data = q.data;
    moved = true;
    return *this;
  }
  ~Q() { --Q::count; }

  Q(int x) : Q() { data = x; }

  operator int() const noexcept { return data; }

  int data;
  bool moved = false;
  static int count;
};

template <int N>
int Q<N>::count = 0;

int main() {
  // test Q
  BOOST_TEST_EQ(Q<1>::count, 0);
  {
    Q<1> q(5);
    BOOST_TEST_EQ(q, 5);
    BOOST_TEST_EQ(Q<1>::count, 1);
    Q<1> q2(q);
    BOOST_TEST_EQ(q2, 5);
    BOOST_TEST_NOT(q2.moved);
    Q<1> q3(std::move(q));
    BOOST_TEST_EQ(q3, 5);
    BOOST_TEST(q3.moved);
    Q<1> q4;
    q4 = Q<1>(3);
    BOOST_TEST_EQ(q4, 3);
    BOOST_TEST(q4.moved);
    Q<1> q5(0xBAD); // ok
    BOOST_TEST_THROWS((Q<1>(q5)), std::bad_alloc);
  }
  BOOST_TEST_EQ(Q<1>::count, 0);

  // default ctor and dtor
  {
    variant<Q<1>> v;
    BOOST_TEST_EQ(v.index(), 0);
    BOOST_TEST_EQ(Q<1>::count, 1);

    variant<> v2;
    (void)v2;
  }
  BOOST_TEST_EQ(Q<1>::count, 0);

  // copy ctor
  {
    using V = variant<Q<1>, Q<2>>;
    Q<1> q1{5};
    const V v1(q1);
    BOOST_TEST_EQ(Q<1>::count, 2);
    BOOST_TEST_EQ(Q<2>::count, 0);
    BOOST_TEST_EQ(v1.index(), 0);
    BOOST_TEST_EQ(v1, q1);
    BOOST_TEST_NOT(v1.get<Q<1>>().moved);
    const Q<2> q2{3};
    V v2(q2);
    BOOST_TEST_EQ(Q<1>::count, 2);
    BOOST_TEST_EQ(Q<2>::count, 2);
    BOOST_TEST_EQ(v2.index(), 1);
    BOOST_TEST_EQ(v2, q2);
    BOOST_TEST_NOT(v2.get<Q<2>>().moved);
    V v3(v1);
    BOOST_TEST_EQ(v3.index(), 0);
    BOOST_TEST_EQ(v3, q1);
    BOOST_TEST_EQ(Q<1>::count, 3);
    BOOST_TEST_EQ(Q<2>::count, 2);
    BOOST_TEST_NOT(v3.get<Q<1>>().moved);
    Q<1> q4(0xBAD);
    BOOST_TEST_THROWS((V(q4)), std::bad_alloc);
  }
  BOOST_TEST_EQ(Q<1>::count, 0);
  BOOST_TEST_EQ(Q<2>::count, 0);

  // move ctor
  {
    using V = variant<Q<1>, Q<2>>;
    V v1(Q<1>{5});
    BOOST_TEST_EQ(v1.index(), 0);
    BOOST_TEST(v1.get<Q<1>>().moved);
    BOOST_TEST_EQ(v1, Q<1>{5});
    V v2(Q<2>{3});
    BOOST_TEST_EQ(v2.index(), 1);
    BOOST_TEST(v2.get<Q<2>>().moved);
    BOOST_TEST_EQ(v2, Q<2>{3});
    Q<1> q{4};
    V v3(q);
    BOOST_TEST_NOT(v3.get<Q<1>>().moved);
    V v4(std::move(v3));
    BOOST_TEST(v4.get<Q<1>>().moved);
  }
  BOOST_TEST_EQ(Q<1>::count, 0);
  BOOST_TEST_EQ(Q<2>::count, 0);

  // move assign
  {
    using V = variant<Q<1>, Q<2>>;
    V v;
    BOOST_TEST_EQ(v.index(), 0);
    BOOST_TEST_NOT(v.get<Q<1>>().moved);
    v = Q<1>{5};
    BOOST_TEST_EQ(v.index(), 0);
    BOOST_TEST_EQ(v, Q<1>{5});
    BOOST_TEST(v.get<Q<1>>().moved);
    BOOST_TEST_EQ(Q<1>::count, 1);
    BOOST_TEST_EQ(Q<2>::count, 0);
    v = Q<2>{3};
    BOOST_TEST_EQ(v.index(), 1);
    BOOST_TEST_EQ(v, Q<2>{3});
    BOOST_TEST(v.get<Q<2>>().moved);
    BOOST_TEST_EQ(Q<1>::count, 0);
    BOOST_TEST_EQ(Q<2>::count, 1);
  }
  BOOST_TEST_EQ(Q<1>::count, 0);
  BOOST_TEST_EQ(Q<2>::count, 0);

  // copy assign
  {
    using V = variant<Q<1>, Q<2>, Q<3>>;
    V v;
    BOOST_TEST_EQ(v.index(), 0);
    Q<1> q{3};
    v = q;
    BOOST_TEST_EQ(v.index(), 0);
    BOOST_TEST_EQ(v, q);
    BOOST_TEST_NOT(v.get<Q<1>>().moved);
    BOOST_TEST_EQ(Q<1>::count, 2);
    BOOST_TEST_EQ(Q<2>::count, 0);
    Q<2> q2(5);
    v = q2;
    BOOST_TEST_EQ(v.index(), 1);
    BOOST_TEST_EQ(v, q2);
    BOOST_TEST_NOT(v.get<Q<2>>().moved);
    BOOST_TEST_EQ(Q<1>::count, 1);
    BOOST_TEST_EQ(Q<2>::count, 2);

    BOOST_TEST_EQ(v.index(), 1);
#ifndef BOOST_NO_EXCEPTIONS
    Q<3> q3(0xBAD);
    BOOST_TEST_THROWS(v = q3, std::bad_alloc);
    BOOST_TEST_EQ(v.index(), 0); // is now in default state
#endif
  }
  BOOST_TEST_EQ(Q<1>::count, 0);
  BOOST_TEST_EQ(Q<2>::count, 0);

  // get
  {
    variant<Q<1>, Q<2>> v;
    v = Q<1>(1);
    BOOST_TEST_EQ(v.get<Q<1>>(), 1);
    BOOST_TEST_THROWS(v.get<Q<2>>(), std::runtime_error);

    const auto& crv = v;
    BOOST_TEST_EQ(crv.get<Q<1>>(), 1);
    BOOST_TEST_THROWS(crv.get<Q<2>>(), std::runtime_error);

    auto p1 = v.get_if<Q<1>>();
    BOOST_TEST(p1 && *p1 == 1);
    p1->data = 3;

    auto p2 = crv.get_if<Q<1>>();
    BOOST_TEST(p2 && *p2 == 3);

    BOOST_TEST_NOT(v.get_if<Q<2>>());
    BOOST_TEST_NOT(v.get_if<int>());
    BOOST_TEST_NOT(crv.get_if<Q<2>>());
    BOOST_TEST_NOT(crv.get_if<int>());
  }

  // apply
  {
    variant<Q<1>, Q<2>> v;
    v = Q<1>(1);
    v.apply([](auto& x) {
      BOOST_TEST_EQ(x, 1);
      BOOST_TEST_TRAIT_SAME(decltype(x), Q<1>&);
    });
    v = Q<2>(2);
    const auto& crv = v;
    crv.apply([](const auto& x) {
      BOOST_TEST_EQ(x, 2);
      BOOST_TEST_TRAIT_SAME(decltype(x), const Q<2>&);
    });
  }

  // ostream
  {
    std::ostringstream os;
    variant<Q<1>, Q<2>> v(Q<1>{3});
    os << v;
    BOOST_TEST_EQ(os.str(), "3"s);
  }

  return boost::report_errors();
}
