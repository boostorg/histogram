// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <string>
#include <utility>
#include "utility_histogram.hpp"
#include "utility_meta.hpp"

using namespace boost::histogram;

using integer = axis::integer<int, axis::null_type, axis::option::growth>;
using category = axis::category<std::string, axis::null_type, axis::option::growth>;

template <typename Tag>
void run_tests() {
  {
    auto h = make_s(Tag(), std::vector<int>(), integer());
    const auto& a = h.axis();
    BOOST_TEST_EQ(h.size(), 0);
    BOOST_TEST_EQ(a.size(), 0);
    h(0);
    BOOST_TEST_EQ(h.size(), 1);
    BOOST_TEST_EQ(h[0], 1);
    h(2);
    BOOST_TEST_EQ(h.size(), 3);
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 0);
    BOOST_TEST_EQ(h[2], 1);
    h(-2);
    BOOST_TEST_EQ(h.size(), 5);
    BOOST_TEST_EQ(a.size(), 5);
    BOOST_TEST_EQ(h[0], 1);
    BOOST_TEST_EQ(h[1], 0);
    BOOST_TEST_EQ(h[2], 1);
    BOOST_TEST_EQ(h[3], 0);
    BOOST_TEST_EQ(h[4], 1);
  }

  {
    auto h = make_s(Tag(), std::vector<int>(), integer(), category());
    const auto& a = h.axis(0);
    const auto& b = h.axis(1);
    BOOST_TEST_EQ(a.size(), 0);
    BOOST_TEST_EQ(b.size(), 0);
    BOOST_TEST_EQ(h.size(), 0);
    h(0, "x");
    BOOST_TEST_EQ(a.size(), 1);
    BOOST_TEST_EQ(b.size(), 1);
    BOOST_TEST_EQ(h.size(), 1);
    h(2, "x");
    BOOST_TEST_EQ(h.size(), 3);
    h(1, "y");
    BOOST_TEST_EQ(h.size(), 6);
    BOOST_TEST_EQ(a.size(), 3);
    BOOST_TEST_EQ(b.size(), 2);
    BOOST_TEST_EQ(h.at(a(0), b("x")), 1);
    BOOST_TEST_EQ(h.at(a(1), b("x")), 0);
    BOOST_TEST_EQ(h.at(a(2), b("x")), 1);
    BOOST_TEST_EQ(h.at(a(0), b("y")), 0);
    BOOST_TEST_EQ(h.at(a(1), b("y")), 1);
    BOOST_TEST_EQ(h.at(a(2), b("y")), 0);
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
