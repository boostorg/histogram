// Copyright (c) 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test is inspired by the corresponding boost/beast test of detail_variant.

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/variant.hpp>
#include <boost/histogram/detail/variant_serialization.hpp>
#include "utility_serialization.hpp"

using namespace boost::histogram::detail;

int main() {
  const char* filename = XML_PATH "detail_variant_serialization_test.xml";

  {
    variant<int, double> a(1.0);
    print_xml(filename, a);

    variant<int, double> b(42);
    BOOST_TEST_NE(a, b);
    load_xml(filename, b);
    BOOST_TEST_EQ(a, b);

    variant<int> c; // load incompatible version
    BOOST_TEST_THROWS(load_xml(filename, c), boost::archive::archive_exception);
  }

  return boost::report_errors();
}
