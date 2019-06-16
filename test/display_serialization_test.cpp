// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include "throw_exception.hpp"
#include <sstream>
#include <vector>
#include "utility_allocator.hpp"
#include "std_ostream.hpp"
#include <boost/assert.hpp>
#include <boost/histogram/serialization.hpp>
#include "utility_serialization.hpp"
#include "../include/boost/histogram/display.hpp"


#include <boost/histogram/ostream.hpp>
#include <cmath>
#include <string>
#include "utility_histogram.hpp"

using namespace boost::histogram;

  namespace {
  const std::string expected_1 = "\n"
  "                  +------------------------------------------------------------+\n"
  "  [-inf, -0.5)  0 |                                                            |\n"
  "  [-0.5,  0.0)  1 |***********                                                 |\n"
  "  [ 0.0,  0.5)  5 |*********************************************************   |\n"
  "  [ 0.5,  1.0)  3 |**********************************                          |\n"
  "  [ 1.0,  1.5)  0 |                                                            |\n"
  "  [ 1.5,  2.0)  1 |***********                                                 |\n"
  "  [ 2.0,  inf]  0 |                                                            |\n"
  "                  +------------------------------------------------------------+\n\n";
  }
  

void run_tests(const std::string& filename, const std::string& expected) {
  auto h1 = make_histogram(axis::regular<>(1, -0.5, 2.0));
  h1(0.5);

  auto h2 = decltype(h1)();
  BOOST_TEST_NE(h1, h2);
  load_xml(filename, h2);

  std::ostringstream os;  
  display::display(h2, os);

  BOOST_TEST_EQ(os.str(), expected);
}

int main(int argc, char** argv) {
  BOOST_ASSERT(argc == 2);

  run_tests(join(argv[1], "display_serialization_test_1.xml"), expected_1);
  
  
  
  return boost::report_errors();
}


