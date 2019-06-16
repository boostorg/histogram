// Copyright (c) 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/assert.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/display.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/serialization.hpp>
#include <sstream>
#include <string>
#include "throw_exception.hpp"
#include "utility_allocator.hpp"
#include "utility_serialization.hpp"

using namespace boost::histogram;

namespace {
const std::string expected_1 =
    "\n"
    "                  +------------------------------------------------------------+\n"
    "  [-inf, -0.5)  0 |                                                            |\n"
    "  [-0.5,  0.0)  1 |***********                                                 |\n"
    "  [ 0.0,  0.5)  5 |*********************************************************   |\n"
    "  [ 0.5,  1.0)  3 |**********************************                          |\n"
    "  [ 1.0,  1.5)  0 |                                                            |\n"
    "  [ 1.5,  2.0)  1 |***********                                                 |\n"
    "  [ 2.0,  inf]  0 |                                                            |\n"
    "                  +------------------------------------------------------------+\n"
    "\n";

const std::string expected_2 =
    "\n"
    "                   +------------------------------------------------------------+\n"
    "  [-inf, -5.0)  0  |                                                            |\n"
    "  [-5.0, -4.1)  19 |*********************************************************   |\n"
    "  [-4.1, -3.3)  16 |************************************************            |\n"
    "  [-3.3, -2.4)  12 |************************************                        |\n"
    "  [-2.4, -1.6)  14 |******************************************                  |\n"
    "  [-1.6, -0.7)  14 |******************************************                  |\n"
    "  [-0.7,  0.1)  10 |******************************                              |\n"
    "  [ 0.1,  1.0)  15 |*********************************************               |\n"
    "  [ 1.0,  inf]  0  |                                                            |\n"
    "                   +------------------------------------------------------------+\n"
    "\n";

const std::string expected_3 =
  "\n"
  "                     +------------------------------------------------------------+\n"
  "  [-inf, -3.0)  0    |                                                            |\n"
  "  [-3.0, -2.4)  819  |**************************************                      |\n"
  "  [-2.4, -1.8)  1181 |*******************************************************     |\n"
  "  [-1.8, -1.2)  854  |****************************************                    |\n"
  "  [-1.2, -0.6)  1162 |*******************************************************     |\n"
  "  [-0.6,  0.0)  884  |*****************************************                   |\n"
  "  [ 0.0,  0.6)  1037 |*************************************************           |\n"
  "  [ 0.6,  1.2)  1020 |************************************************            |\n"
  "  [ 1.2,  1.8)  1203 |*********************************************************   |\n"
  "  [ 1.8,  2.4)  844  |***************************************                     |\n"
  "  [ 2.4,  3.0)  996  |***********************************************             |\n"
  "  [ 3.0,  inf]  0    |                                                            |\n"
  "                     +------------------------------------------------------------+\n"
  "\n";
} // namespace

void run_tests(const std::string& filename, const std::string& expected) {
  auto h1 = make_histogram(axis::regular<>(1, -0.5, 2.0));
  h1(0.5);

  auto h2 = decltype(h1)();
  BOOST_TEST_NE(h1, h2);
  load_xml(filename, h2);

  std::ostringstream os;
  display::display(h2, os);
  std::cout << os.str();
  BOOST_TEST_EQ(os.str(), expected);
}

int main(int argc, char** argv) {
  BOOST_ASSERT(argc == 2);

  run_tests(join(argv[1], "display_serialization_test_1.xml"), expected_1);
  run_tests(join(argv[1], "display_serialization_test_2.xml"), expected_2);
  run_tests(join(argv[1], "display_serialization_test_3.xml"), expected_3);
  return boost::report_errors();
}
