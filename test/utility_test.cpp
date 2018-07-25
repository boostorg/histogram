// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utility.hpp"
#include <boost/core/lightweight_test.hpp>
#include <sstream>
#include <tuple>
#include <vector>

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
}
