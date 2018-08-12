// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <sstream>
#include <string>
#include "utility.hpp"

namespace bhd = boost::histogram::detail;
namespace bhad = boost::histogram::axis::detail;

int main() {
  // escape0
  {
    std::ostringstream os;
    bhad::escape_string(os, std::string("abc"));
    BOOST_TEST_EQ(os.str(), std::string("'abc'"));
  }

  // escape1
  {
    std::ostringstream os;
    bhad::escape_string(os, std::string("abc\n"));
    BOOST_TEST_EQ(os.str(), std::string("'abc\n'"));
  }

  // escape2
  {
    std::ostringstream os;
    bhad::escape_string(os, std::string("'abc'"));
    BOOST_TEST_EQ(os.str(), std::string("'\\\'abc\\\''"));
  }

  // cat
  { BOOST_TEST_EQ(bhd::cat("foo", 1, "bar"), std::string("foo1bar")); }

  return boost::report_errors();
}
