// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/weighted_mean.hpp>
#include <boost/histogram/algorithm/empty.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <unordered_map>
#include <vector>
#include "throw_exception.hpp"
#include "utility_histogram.hpp"

using namespace boost::histogram;
using boost::histogram::algorithm::empty;

template <typename Tag>
void run_tests() {
  auto ax = axis::integer<>(0, 10);

  auto h1 = make(Tag(), ax);
  BOOST_TEST(empty(h1, coverage::all));
  BOOST_TEST(empty(h1, coverage::inner));
  for (int i = -1; i < 11; ++i) {
    h1.reset();
    h1(i);
    BOOST_TEST(!empty(h1, coverage::all));
    if (i == -1 || i == 10) {
      BOOST_TEST(empty(h1, coverage::inner));
    } else {
      BOOST_TEST(!empty(h1, coverage::inner));
    }
  }

  auto h2 = make_s(Tag(), std::vector<double>(), ax, ax);
  BOOST_TEST(empty(h2, coverage::all));
  BOOST_TEST(empty(h2, coverage::inner));
  for (int i = -1; i < 11; ++i) {
    for (int j = -1; j < 11; ++j) {
      h2.reset();
      h2(i, j);
      BOOST_TEST(!empty(h2, coverage::all));

      if ((i == -1 || i == 10) || (j == -1 || j == 10)) {
        BOOST_TEST(empty(h2, coverage::inner));
      } else {
        BOOST_TEST(!empty(h2, coverage::inner));
      }
    }
  }

  /* BROKEN, SEE https://github.com/boostorg/histogram/issues/244
  auto h3 = make_s(Tag(), std::array<int, 12>(), ax);
  BOOST_TEST(empty(h3, coverage::all));
  BOOST_TEST(empty(h3, coverage::inner));
  h3(-2);
  BOOST_TEST(!empty(h3, coverage::all));
  BOOST_TEST(empty(h3, coverage::inner));
  h3(2);
  BOOST_TEST(!empty(h3, coverage::all));
  BOOST_TEST(!empty(h3, coverage::inner));
  */

  auto h4 = make_s(Tag(), std::unordered_map<std::size_t, int>(), ax);
  BOOST_TEST(empty(h4, coverage::all));
  BOOST_TEST(empty(h4, coverage::inner));
  h4(-2);
  BOOST_TEST(!empty(h4, coverage::all));
  BOOST_TEST(empty(h4, coverage::inner));
  h4(2);
  BOOST_TEST(!empty(h4, coverage::all));
  BOOST_TEST(!empty(h4, coverage::inner));

  auto h5 = make_s(Tag(), std::vector<accumulators::weighted_sum<>>(),
                   axis::integer<>(0, 10), axis::integer<>(0, 10));
  BOOST_TEST(empty(h5, coverage::all));
  BOOST_TEST(empty(h5, coverage::inner));
  h5.reset();
  h5(weight(2), -2, -4);
  BOOST_TEST(!empty(h5, coverage::all));
  BOOST_TEST(empty(h5, coverage::inner));
  h5.reset();
  h5(weight(1), -4, 2);
  BOOST_TEST(!empty(h5, coverage::all));
  BOOST_TEST(empty(h5, coverage::inner));
  h5.reset();
  h5(weight(3), 3, 5);
  BOOST_TEST(!empty(h5, coverage::all));
  BOOST_TEST(!empty(h5, coverage::inner));

  auto h6 = make_s(Tag(), std::vector<accumulators::weighted_mean<>>(),
                   axis::integer<>(0, 10), axis::integer<>(0, 10));
  BOOST_TEST(empty(h6, coverage::all));
  BOOST_TEST(empty(h6, coverage::inner));
  h6.reset();
  h6(weight(2), -2, -4);
  BOOST_TEST(!empty(h6, coverage::all));
  BOOST_TEST(empty(h6, coverage::inner));
  h6.reset();
  h6(weight(1), -4, 2);
  BOOST_TEST(!empty(h6, coverage::all));
  BOOST_TEST(empty(h6, coverage::inner));
  h6.reset();
  h6(weight(3), 3, 5);
  BOOST_TEST(!empty(h6, coverage::all));
  BOOST_TEST(!empty(h6, coverage::inner));
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
