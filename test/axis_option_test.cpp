// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/fwd.hpp>

using namespace boost::histogram::axis;

int main() {
  BOOST_TEST_EQ(option::use_default, option::underflow | option::overflow);
  BOOST_TEST(test(option::use_default, option::underflow));
  BOOST_TEST(test(option::use_default, option::overflow));
  BOOST_TEST_NOT(test(option::use_default, option::circular));
  BOOST_TEST_NOT(test(option::use_default, option::growth));
  BOOST_TEST_EQ(join(option::use_default, option::underflow), option::use_default);
  BOOST_TEST_EQ(join(option::use_default, option::overflow), option::use_default);
  BOOST_TEST_EQ(join(option::use_default, option::circular),
                option::overflow | option::circular);
  BOOST_TEST_EQ(join(option::use_default, option::growth), option::growth);
  BOOST_TEST_EQ(join(option::growth, option::underflow), option::underflow);
  BOOST_TEST_EQ(join(option::growth, option::overflow), option::overflow);
  BOOST_TEST_EQ(join(option::growth, option::use_default), option::use_default);
  return boost::report_errors();
}
