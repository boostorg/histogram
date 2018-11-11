// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram/detail/index_mapper.hpp>
#include <vector>

using namespace boost::histogram::detail;

int main() {
  // index_mapper 1
  {
    // shape: 2, 3; 2, 0
    index_mapper m(2);
    m[0] = std::make_pair(1, 1);
    m[1] = std::make_pair(2, 0);
    m.ntotal = 6;
    BOOST_TEST_EQ(m.first, 0);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 1);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 2);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 3);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 4);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 5);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), false);
  }

  // index_mapper 2
  {
    // shape: 2, 3; 0, 3
    index_mapper m(2);
    m[0] = std::make_pair(1, 0);
    m[1] = std::make_pair(2, 1);
    m.ntotal = 6;
    BOOST_TEST_EQ(m.first, 0);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 1);
    BOOST_TEST_EQ(m.second, 0);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 2);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 3);
    BOOST_TEST_EQ(m.second, 1);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 4);
    BOOST_TEST_EQ(m.second, 2);
    BOOST_TEST_EQ(m.next(), true);
    BOOST_TEST_EQ(m.first, 5);
    BOOST_TEST_EQ(m.second, 2);
    BOOST_TEST_EQ(m.next(), false);
  }

  return boost::report_errors();
}
