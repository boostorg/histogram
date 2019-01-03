// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram.hpp>
#include <type_traits>
#include "utility_histogram.hpp"

using namespace boost::histogram;

template <typename Tag>
void run_tests() {
  auto a = make(Tag(), axis::integer<>(0, 2));
  auto b = a;
  a(0);
  b(1);
  auto c = a + b;
  BOOST_TEST_EQ(c.at(0), 1);
  BOOST_TEST_EQ(c.at(1), 1);
  c += b;
  BOOST_TEST_EQ(c.at(0), 1);
  BOOST_TEST_EQ(c.at(1), 2);
  auto d = a + b + c;
  BOOST_TEST_EQ(d.at(0), 2);
  BOOST_TEST_EQ(d.at(1), 3);
  BOOST_TEST_TRAIT_TRUE((std::is_same<decltype(d), decltype(a)>));
  auto e = 3 * a;
  auto f = b * 2;
  BOOST_TEST_TRAIT_FALSE((std::is_same<decltype(e), decltype(a)>));
  BOOST_TEST_TRAIT_FALSE((std::is_same<decltype(f), decltype(a)>));
  BOOST_TEST_EQ(e.at(0), 3);
  BOOST_TEST_EQ(e.at(1), 0);
  BOOST_TEST_EQ(f.at(0), 0);
  BOOST_TEST_EQ(f.at(1), 2);
  auto r = 1.0 * a;
  r += b;
  r += e;
  BOOST_TEST_EQ(r.at(0), 4);
  BOOST_TEST_EQ(r.at(1), 1);
  BOOST_TEST_EQ(r, a + b + 3 * a);
  auto s = r / 4;
  r /= 4;
  BOOST_TEST_EQ(r.at(0), 1);
  BOOST_TEST_EQ(r.at(1), 0.25);
  BOOST_TEST_EQ(r, s);
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
