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
  BOOST_TEST_EQ(option::defaults, option::underflow | option::overflow);
  BOOST_TEST(test(option::defaults, option::underflow));
  BOOST_TEST(test(option::defaults, option::overflow));
  BOOST_TEST_NOT(test(option::defaults, option::circular));
  BOOST_TEST_NOT(test(option::defaults, option::growth));
  BOOST_TEST_EQ(join(option::defaults, option::underflow), option::defaults);
  BOOST_TEST_EQ(join(option::defaults, option::overflow), option::defaults);
  BOOST_TEST_EQ(join(option::defaults, option::circular),
                option::overflow | option::circular);
  BOOST_TEST_EQ(join(option::defaults, option::growth), option::growth);
  BOOST_TEST_EQ(join(option::growth, option::underflow), option::underflow);
  BOOST_TEST_EQ(join(option::growth, option::overflow), option::overflow);
  BOOST_TEST_EQ(join(option::growth, option::defaults), option::defaults);
  return boost::report_errors();
}
