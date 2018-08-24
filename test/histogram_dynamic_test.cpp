// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <array>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <boost/histogram/storage/adaptive_storage.hpp>
#include <boost/histogram/storage/array_storage.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>
#include "utility.hpp"

using namespace boost::histogram;
using namespace boost::histogram::literals; // to get _c suffix

int main() {
  // special stuff that only works with dynamic_tag

  // init
  {
    auto v = std::vector<axis::any<axis::regular<>, axis::integer<>>>();
    v.push_back(axis::regular<>(4, -1, 1));
    v.push_back(axis::integer<>(1, 7));
    auto h = make_dynamic_histogram(v.begin(), v.end());
    BOOST_TEST_EQ(h.dim(), 2);
    BOOST_TEST_EQ(h.axis(0), v[0]);
    BOOST_TEST_EQ(h.axis(1), v[1]);

    auto h2 = make_dynamic_histogram_with(array_storage<int>(), v.begin(), v.end());
    BOOST_TEST_EQ(h2.dim(), 2);
    BOOST_TEST_EQ(h2.axis(0), v[0]);
    BOOST_TEST_EQ(h2.axis(1), v[1]);
  }

  // bad fill argument
  {
    auto h = make_dynamic_histogram(axis::integer<>(0, 3));
    BOOST_TEST_THROWS(h(std::string()), std::invalid_argument);
  }

  // axis methods
  {
    enum { A, B };
    auto c = make_dynamic_histogram(axis::category<>({A, B}));
    BOOST_TEST_THROWS(c.axis().lower(0), std::runtime_error);
  }

  // reduce
  {
    auto h1 = make_dynamic_histogram(axis::integer<>(0, 2), axis::integer<>(0, 3));
    h1(0, 0);
    h1(0, 1);
    h1(1, 0);
    h1(1, 1);
    h1(1, 2);

    std::vector<int> x;

    x = {0};
    auto h1_0 = h1.reduce_to(x.begin(), x.end());
    BOOST_TEST_EQ(h1_0.dim(), 1);
    BOOST_TEST_EQ(sum(h1_0), 5);
    BOOST_TEST_EQ(h1_0.at(0), 2);
    BOOST_TEST_EQ(h1_0.at(1), 3);
    BOOST_TEST(h1_0.axis() == h1.axis(0_c));

    x = {1};
    auto h1_1 = h1.reduce_to(x.begin(), x.end());
    BOOST_TEST_EQ(h1_1.dim(), 1);
    BOOST_TEST_EQ(sum(h1_1), 5);
    BOOST_TEST_EQ(h1_1.at(0), 2);
    BOOST_TEST_EQ(h1_1.at(1), 2);
    BOOST_TEST_EQ(h1_1.at(2), 1);
    BOOST_TEST(h1_1.axis() == h1.axis(1_c));
  }

  // histogram iterator
  {
    auto h = make_dynamic_histogram(axis::integer<>(0, 3));
    const auto& a = h.axis();
    h(weight(2), 0);
    h(1);
    h(1);
    auto it = h.begin();
    BOOST_TEST_EQ(it.dim(), 1);

    BOOST_TEST_EQ(it.idx(0), 0);
    BOOST_TEST_EQ(it.bin(0), a[0]);
    BOOST_TEST_EQ(it->value(), 2);
    BOOST_TEST_EQ(it->variance(), 4);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 1);
    BOOST_TEST_EQ(it.bin(0), a[1]);
    BOOST_TEST_EQ(*it, 2);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 2);
    BOOST_TEST_EQ(it.bin(0), a[2]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST_EQ(it.idx(0), 3);
    BOOST_TEST_EQ(it.bin(0), a[3]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST_EQ(it.idx(0), -1);
    BOOST_TEST_EQ(it.bin(0), a[-1]);
    BOOST_TEST_EQ(*it, 0);
    ++it;
    BOOST_TEST(it == h.end());
  }

  return boost::report_errors();
}
