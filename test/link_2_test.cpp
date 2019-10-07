// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/ostream.hpp>

// All files should be included above

int do_nothing();

// Verifies there are no functions with missing inline
int main() {

  int a = do_nothing();
  BOOST_TEST_EQ(a, 7);

  return boost::report_errors();
}
