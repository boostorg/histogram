// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/indexed.hpp>
#include <vector>
#include "utility_histogram.hpp"

using namespace boost::histogram;

template <typename Tag>
void run_tests() {
  // histogram iterator 1D
  {
    auto h = make(Tag(), axis::integer<>(0, 3));
    h(weight(2), 0);
    h(1);
    h(1);

    auto ind = indexed(h);
    auto it = ind.begin();
    BOOST_TEST_EQ(it->first.size(), 1);

    BOOST_TEST_EQ(it->first[0], 0);
    BOOST_TEST_EQ(it->second, 2);
    ++it;
    BOOST_TEST_EQ(it->first[0], 1);
    BOOST_TEST_EQ(it->second, 2);
    ++it;
    BOOST_TEST_EQ(it->first[0], 2);
    BOOST_TEST_EQ(it->second, 0);
    ++it;
    BOOST_TEST_EQ(it->first[0], 3);
    BOOST_TEST_EQ(it->second, 0);
    ++it;
    BOOST_TEST_EQ(it->first[0], -1);
    BOOST_TEST_EQ(it->second, 0);
    ++it;
    BOOST_TEST(it == ind.end());
  }

  // histogram iterator 2D
  {
    auto h = make_s(Tag(), std::vector<int>(), axis::integer<>(0, 1),
                    axis::integer<int, axis::null_type, axis::option_type::none>(2, 4));
    h(weight(2), 0, 2);
    h(-1, 2);
    h(1, 3);

    BOOST_TEST_EQ(axis::traits::extend(h.axis(0)), 3);
    BOOST_TEST_EQ(axis::traits::extend(h.axis(1)), 2);

    auto ind = indexed(h);
    auto it = ind.begin();
    BOOST_TEST_EQ(it->first.size(), 2);

    BOOST_TEST_EQ(it->first[0], 0);
    BOOST_TEST_EQ(it->first[1], 0);
    BOOST_TEST_EQ(it->second, 2);
    ++it;
    BOOST_TEST_EQ(it->first[0], 1);
    BOOST_TEST_EQ(it->first[1], 0);
    BOOST_TEST_EQ(it->second, 0);
    ++it;
    BOOST_TEST_EQ(it->first[0], -1);
    BOOST_TEST_EQ(it->first[1], 0);
    BOOST_TEST_EQ(it->second, 1);
    ++it;
    BOOST_TEST_EQ(it->first[0], 0);
    BOOST_TEST_EQ(it->first[1], 1);
    BOOST_TEST_EQ(it->second, 0);
    ++it;
    BOOST_TEST_EQ(it->first[0], 1);
    BOOST_TEST_EQ(it->first[1], 1);
    BOOST_TEST_EQ(it->second, 1);
    ++it;
    BOOST_TEST_EQ(it->first[0], -1);
    BOOST_TEST_EQ(it->first[1], 1);
    BOOST_TEST_EQ(it->second, 0);
    ++it;
    BOOST_TEST(it == ind.end());
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
