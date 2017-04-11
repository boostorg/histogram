// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/histogram/detail/weight.hpp>
#include <boost/variant.hpp>
#include <cstring>
#include <sstream>
using namespace boost::histogram::detail;

int main() {
  // weight
  {
    BOOST_TEST(weight(0) == weight());
    weight w(1);
    BOOST_TEST(w == weight(1));
    BOOST_TEST(w != weight());
    BOOST_TEST(1 == w);
    BOOST_TEST(w == 1);
    BOOST_TEST(2 != w);
    BOOST_TEST(w != 2);
  }

  // escape0
  {
    std::ostringstream os;
    escape(os, std::string("abc"));
    BOOST_TEST_EQ(os.str(), std::string("'abc'"));
  }

  // escape1
  {
    std::ostringstream os;
    escape(os, std::string("abc\n"));
    BOOST_TEST_EQ(os.str(), std::string("'abc\n'"));
  }

  // escape2
  {
    std::ostringstream os;
    escape(os, std::string("'abc'"));
    BOOST_TEST_EQ(os.str(), std::string("'\\\'abc\\\''"));
  }

  // assign_axis unreachable branch
  {
    using V1 = boost::variant<float>;
    using V2 = boost::variant<int>;
    V1 v1(1.0);
    V2 v2(2);
    boost::apply_visitor(assign_axis<V1>(v1), v2);
    BOOST_TEST_EQ(v1, V1(1.0));
    BOOST_TEST_EQ(v2, V2(2));
  }

  return boost::report_errors();
}
