// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <array>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/adaptive_storage.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/weight_counter.hpp>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>
#include "utility_histogram.hpp"

using namespace boost::histogram;
using namespace boost::histogram::literals; // to get _c suffix

int main() {
  // special stuff that only works with dynamic_tag

  // init
  {
    auto v = std::vector<axis::variant<axis::regular<>, axis::integer<>>>();
    v.emplace_back(axis::regular<>(4, -1, 1));
    v.emplace_back(axis::integer<>(1, 7));
    auto h = make_histogram(v.begin(), v.end());
    BOOST_TEST_EQ(h.rank(), 2);
    BOOST_TEST_EQ(h.axis(0), v[0]);
    BOOST_TEST_EQ(h.axis(1), v[1]);

    auto h2 = make_histogram_with(std::vector<int>(), v);
    BOOST_TEST_EQ(h2.rank(), 2);
    BOOST_TEST_EQ(h2.axis(0), v[0]);
    BOOST_TEST_EQ(h2.axis(1), v[1]);

    auto v2 = std::vector<axis::regular<>>();
    v2.emplace_back(10, 0, 1);
    v2.emplace_back(20, 0, 2);
    auto h3 = make_histogram(v2);
    BOOST_TEST_EQ(h3.axis(0), v2[0]);
    BOOST_TEST_EQ(h3.axis(1), v2[1]);
  }

  // bad fill argument
  {
    auto h = make(dynamic_tag(), axis::integer<>(0, 3));
    BOOST_TEST_THROWS(h(std::string()), std::invalid_argument);
  }

  // axis methods
  {
    auto c = make(dynamic_tag(), axis::category<std::string>({"A", "B"}));
    BOOST_TEST_THROWS(c.axis().value(0), std::runtime_error);
  }

  // reduce
  {
    auto h1 = make(dynamic_tag(), axis::integer<>(0, 2), axis::integer<>(0, 3));
    h1(0, 0);
    h1(0, 1);
    h1(1, 0);
    h1(1, 1);
    h1(1, 2);

    std::vector<int> x;

    x = {0};
    auto h1_0 = h1.reduce_to(x.begin(), x.end());
    BOOST_TEST_EQ(h1_0.rank(), 1);
    BOOST_TEST_EQ(sum(h1_0), 5);
    BOOST_TEST_EQ(h1_0.at(0), 2);
    BOOST_TEST_EQ(h1_0.at(1), 3);
    BOOST_TEST(h1_0.axis() == h1.axis(0_c));

    x = {1};
    auto h1_1 = h1.reduce_to(x.begin(), x.end());
    BOOST_TEST_EQ(h1_1.rank(), 1);
    BOOST_TEST_EQ(sum(h1_1), 5);
    BOOST_TEST_EQ(h1_1.at(0), 2);
    BOOST_TEST_EQ(h1_1.at(1), 2);
    BOOST_TEST_EQ(h1_1.at(2), 1);
    BOOST_TEST(h1_1.axis() == h1.axis(1_c));
  }

  // wrong dimension
  {
    auto h1 = make(dynamic_tag(), axis::integer<>(0, 2));
    h1(1);
    BOOST_TEST_THROWS(h1.at(0, 0), std::invalid_argument);
    BOOST_TEST_THROWS(h1.at(std::make_tuple(0, 0)), std::invalid_argument);
  }

  {
    auto h = make_histogram(std::vector<axis::integer<>>(1, axis::integer<>(0, 3)));
    h(0);
    h(1);
    h(2);
    BOOST_TEST_EQ(h.at(0), 1);
    BOOST_TEST_EQ(h.at(1), 1);
    BOOST_TEST_EQ(h.at(2), 1);
  }

  return boost::report_errors();
}
