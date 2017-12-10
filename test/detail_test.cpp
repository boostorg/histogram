// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/axis_visitor.hpp>
#include <boost/histogram/detail/utility.hpp>
#include <boost/variant.hpp>
#include <sstream>
#include <string>
#include <vector>
using namespace boost::histogram::detail;

int main() {
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

  // index_mapper 1
  {
    std::vector<unsigned> n{{2, 2}};
    std::vector<bool> b{{true, false}};
    index_mapper m(std::move(n), std::move(b));
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
    BOOST_TEST_EQ(m.next(), false);
  }

  // index_mapper 2
  {
    std::vector<unsigned> n{{2, 2}};
    std::vector<bool> b{{false, true}};
    index_mapper m(std::move(n), std::move(b));
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
    BOOST_TEST_EQ(m.next(), false);
  }

  return boost::report_errors();
}
