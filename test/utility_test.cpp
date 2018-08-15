// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utility.hpp"
#include <boost/core/lightweight_test.hpp>
#include <sstream>
#include <tuple>
#include <vector>

using namespace boost::histogram;

int main() {
  // vector streaming
  {
    std::ostringstream os;
    std::vector<int> v = {1, 3, 2};
    os << v;
    BOOST_TEST_EQ(os.str(), std::string("[ 1 3 2 ]"));
  }

  // tuple streaming
  {
    std::ostringstream os;
    auto v = std::make_tuple(1, 2.5, "hi");
    os << v;
    BOOST_TEST_EQ(os.str(), std::string("[ 1 2.5 hi ]"));
  }
  return boost::report_errors();

  // tracing_allocator
  {
    std::set<std::string> types;
    std::size_t allocated_bytes = 0;
    std::size_t deallocated_bytes = 0;
    tracing_allocator<char> a(types, allocated_bytes, deallocated_bytes);
    auto p1 = a.allocate(2);
    a.deallocate(p1, 2);
    tracing_allocator<int> b(a);
    auto p2 = b.allocate(3);
    b.deallocate(p2, 3);
    auto expected = {"char", "int"};
    BOOST_TEST_ALL_EQ(types.begin(), types.end(), expected.begin(), expected.end());
    BOOST_TEST_EQ(allocated_bytes, 2 + 3 * sizeof(int));
    BOOST_TEST_EQ(allocated_bytes, deallocated_bytes);
  }
}
