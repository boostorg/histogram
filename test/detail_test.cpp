// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/cat.hpp>

using namespace boost::histogram::detail;

int main() {
  BOOST_TEST_EQ(cat("foo", 1, "bar"), "foo1bar");

  return boost::report_errors();
}
