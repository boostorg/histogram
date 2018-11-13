// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include "utility_histogram.hpp"

using namespace boost::histogram;
using namespace boost::histogram::literals; // to get _c suffix
using boost::histogram::algorithm::sum;

template <typename Tag>
void run_tests() {
  auto ax = axis::integer<>(0, 100);

  auto h1 = make(Tag(), ax);
  for (unsigned i = 0; i < 100; ++i) h1(i);
  BOOST_TEST_EQ(sum(h1), 100);

  auto h2 = make_s(Tag(), std::vector<double>(), ax, ax);
  for (unsigned i = 0; i < 100; ++i)
    for (unsigned j = 0; j < 100; ++j) h2(i, j);

  BOOST_TEST_EQ(sum(h2), 100000);

  auto h3 = make_s(Tag(), std::array<int, 102>(), ax);
  for (unsigned i = 0; i < 100; ++i) h3(i);

  BOOST_TEST_EQ(sum(h3), 100);

  auto h4 = make_s(Tag(), std::unordered_map<std::size_t, int>(), ax);
  for (unsigned i = 0; i < 100; ++i) h3(i);

  BOOST_TEST_EQ(sum(h3), 100);
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
