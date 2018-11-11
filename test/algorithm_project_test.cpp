// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/algorithm/project.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/literals.hpp>
#include <vector>
#include "utility_histogram.hpp"

using namespace boost::histogram;
using namespace boost::histogram::literals; // to get _c suffix
using boost::histogram::algorithm::project;

template <typename Tag>
void run_tests() {
  auto h1 = make(Tag(), axis::integer<>(0, 2), axis::integer<>(0, 3));
  h1(0, 0);
  h1(0, 1);
  h1(1, 0);
  h1(1, 1);
  h1(1, 2);

  /*
    matrix layout:

    0 ->
  1 1 1 0 0
  | 1 1 0 0
  v 0 1 0 0
    0 0 0 0
    0 0 0 0
  */

  auto h1_0 = project(h1, 0_c);
  BOOST_TEST_EQ(h1_0.rank(), 1);
  BOOST_TEST_EQ(sum(h1_0), 5);
  BOOST_TEST_EQ(h1_0.at(0), 2);
  BOOST_TEST_EQ(h1_0.at(1), 3);
  BOOST_TEST(h1_0.axis() == h1.axis(0_c));

  auto h1_1 = project(h1, 1_c);
  BOOST_TEST_EQ(h1_1.rank(), 1);
  BOOST_TEST_EQ(sum(h1_1), 5);
  BOOST_TEST_EQ(h1_1.at(0), 2);
  BOOST_TEST_EQ(h1_1.at(1), 2);
  BOOST_TEST_EQ(h1_1.at(2), 1);
  BOOST_TEST(h1_1.axis() == h1.axis(1_c));

  auto h2 =
      make(Tag(), axis::integer<>(0, 2), axis::integer<>(0, 3), axis::integer<>(0, 4));
  h2(0, 0, 0);
  h2(0, 1, 0);
  h2(0, 1, 1);
  h2(0, 0, 2);
  h2(1, 0, 2);

  auto h2_0 = project(h2, 0_c);
  BOOST_TEST_EQ(h2_0.rank(), 1);
  BOOST_TEST_EQ(sum(h2_0), 5);
  BOOST_TEST_EQ(h2_0.at(0), 4);
  BOOST_TEST_EQ(h2_0.at(1), 1);
  BOOST_TEST(h2_0.axis() == axis::integer<>(0, 2));

  auto h2_1 = project(h2, 1_c);
  BOOST_TEST_EQ(h2_1.rank(), 1);
  BOOST_TEST_EQ(sum(h2_1), 5);
  BOOST_TEST_EQ(h2_1.at(0), 3);
  BOOST_TEST_EQ(h2_1.at(1), 2);
  BOOST_TEST(h2_1.axis() == axis::integer<>(0, 3));

  auto h2_2 = project(h2, 2_c);
  BOOST_TEST_EQ(h2_2.rank(), 1);
  BOOST_TEST_EQ(sum(h2_2), 5);
  BOOST_TEST_EQ(h2_2.at(0), 2);
  BOOST_TEST_EQ(h2_2.at(1), 1);
  BOOST_TEST_EQ(h2_2.at(2), 2);
  BOOST_TEST(h2_2.axis() == axis::integer<>(0, 4));

  auto h2_01 = project(h2, 0_c, 1_c);
  BOOST_TEST_EQ(h2_01.rank(), 2);
  BOOST_TEST_EQ(sum(h2_01), 5);
  BOOST_TEST_EQ(h2_01.at(0, 0), 2);
  BOOST_TEST_EQ(h2_01.at(0, 1), 2);
  BOOST_TEST_EQ(h2_01.at(1, 0), 1);
  BOOST_TEST(h2_01.axis(0_c) == axis::integer<>(0, 2));
  BOOST_TEST(h2_01.axis(1_c) == axis::integer<>(0, 3));

  auto h2_02 = project(h2, 0_c, 2_c);
  BOOST_TEST_EQ(h2_02.rank(), 2);
  BOOST_TEST_EQ(sum(h2_02), 5);
  BOOST_TEST_EQ(h2_02.at(0, 0), 2);
  BOOST_TEST_EQ(h2_02.at(0, 1), 1);
  BOOST_TEST_EQ(h2_02.at(0, 2), 1);
  BOOST_TEST_EQ(h2_02.at(1, 2), 1);
  BOOST_TEST(h2_02.axis(0_c) == axis::integer<>(0, 2));
  BOOST_TEST(h2_02.axis(1_c) == axis::integer<>(0, 4));

  auto h2_12 = project(h2, 1_c, 2_c);
  BOOST_TEST_EQ(h2_12.rank(), 2);
  BOOST_TEST_EQ(sum(h2_12), 5);
  BOOST_TEST_EQ(h2_12.at(0, 0), 1);
  BOOST_TEST_EQ(h2_12.at(1, 0), 1);
  BOOST_TEST_EQ(h2_12.at(1, 1), 1);
  BOOST_TEST_EQ(h2_12.at(0, 2), 2);
  BOOST_TEST(h2_12.axis(0_c) == axis::integer<>(0, 3));
  BOOST_TEST(h2_12.axis(1_c) == axis::integer<>(0, 4));
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  {
    auto h1 = make(dynamic_tag(), axis::integer<>(0, 2), axis::integer<>(0, 3));
    h1(0, 0);
    h1(0, 1);
    h1(1, 0);
    h1(1, 1);
    h1(1, 2);

    std::vector<int> x;

    x = {0};
    auto h1_0 = project(h1, x.begin(), x.end());
    BOOST_TEST_EQ(h1_0.rank(), 1);
    BOOST_TEST_EQ(sum(h1_0), 5);
    BOOST_TEST_EQ(h1_0.at(0), 2);
    BOOST_TEST_EQ(h1_0.at(1), 3);
    BOOST_TEST(h1_0.axis() == h1.axis(0_c));

    x = {1};
    auto h1_1 = project(h1, x.begin(), x.end());
    BOOST_TEST_EQ(h1_1.rank(), 1);
    BOOST_TEST_EQ(sum(h1_1), 5);
    BOOST_TEST_EQ(h1_1.at(0), 2);
    BOOST_TEST_EQ(h1_1.at(1), 2);
    BOOST_TEST_EQ(h1_1.at(2), 1);
    BOOST_TEST(h1_1.axis() == h1.axis(1_c));
  }

  return boost::report_errors();
}
