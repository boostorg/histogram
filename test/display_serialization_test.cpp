// Copyright (c) 2019  pb
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/assert.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/display.hpp>
#include <boost/histogram/serialization.hpp>
#include <sstream>
#include <string>
#include "throw_exception.hpp"
#include "utility_allocator.hpp"
#include "utility_serialization.hpp"

using namespace boost::histogram;

namespace {
const std::string h1_expected_r = //regular
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
    
const std::string h2_expected_r = //regular
    "\n"
    "                   +-----------------------------------------------------------+\n"
    "  [-inf, -5.0)  0  |                                                           |\n"
    "  [-5.0, -4.1)  19 |********************************************************   |\n"
    "  [-4.1, -3.3)  16 |***********************************************            |\n"
    "  [-3.3, -2.4)  12 |***********************************                        |\n"
    "  [-2.4, -1.6)  14 |*****************************************                  |\n"
    "  [-1.6, -0.7)  14 |*****************************************                  |\n"
    "  [-0.7,  0.1)  10 |******************************                             |\n"
    "  [ 0.1,  1.0)  15 |********************************************               |\n"
    "  [ 1.0,  inf]  0  |                                                           |\n"
    "                   +-----------------------------------------------------------+\n"
    "\n";

const std::string h3_expected_r = //regular
    "\n"
    "                     +---------------------------------------------------------+\n"
    "  [-inf, -3.0)  0    |                                                         |\n"
    "  [-3.0, -2.4)  819  |*************************************                    |\n"
    "  [-2.4, -1.8)  1181 |*****************************************************    |\n"
    "  [-1.8, -1.2)  854  |**************************************                   |\n"
    "  [-1.2, -0.6)  1162 |****************************************************     |\n"
    "  [-0.6,  0.0)  884  |****************************************                 |\n"
    "  [ 0.0,  0.6)  1037 |***********************************************          |\n"
    "  [ 0.6,  1.2)  1020 |**********************************************           |\n"
    "  [ 1.2,  1.8)  1203 |******************************************************   |\n"
    "  [ 1.8,  2.4)  844  |**************************************                   |\n"
    "  [ 2.4,  3.0)  996  |*********************************************            |\n"
    "  [ 3.0,  inf]  0    |                                                         |\n"
    "                     +---------------------------------------------------------+\n"
    "\n";

const std::string h4_expected_r = 
    "\n"
    "                 +-------------------------------------------------------------+\n"
    "  [-inf, 0.0)  0 |                                                             |\n"
    "  [ 0.0, 0.5)  2 |**********************************************************   |\n"
    "  [ 0.5, 1.0)  0 |                                                             |\n"
    "  [ 1.0, 1.5)  1 |*****************************                                |\n"
    "  [ 1.5, 2.0)  1 |*****************************                                |\n"
    "  [ 2.0, inf]  1 |*****************************                                |\n"
    "                 +-------------------------------------------------------------+\n"
    "\n";

const std::string h4_expected_n = //narow
    "\n"
    "                 +-------------------------------+\n"
    "  [-inf, 0.0)  0 |                               |\n"
    "  [ 0.0, 0.5)  2 |*****************************  |\n"
    "  [ 0.5, 1.0)  0 |                               |\n"
    "  [ 1.0, 1.5)  1 |***************                |\n"
    "  [ 1.5, 2.0)  1 |***************                |\n"
    "  [ 2.0, inf]  1 |***************                |\n"
    "                 +-------------------------------+\n"
    "\n";

const std::string h4_expected_w = //wide
    "\n"
    "                 +-----------------------------------------------------------------------+\n"
    "  [-inf, 0.0)  0 |                                                                       |\n"
    "  [ 0.0, 0.5)  2 |*******************************************************************    |\n"
    "  [ 0.5, 1.0)  0 |                                                                       |\n"
    "  [ 1.0, 1.5)  1 |**********************************                                     |\n"
    "  [ 1.5, 2.0)  1 |**********************************                                     |\n"
    "  [ 2.0, inf]  1 |**********************************                                     |\n"
    "                 +-----------------------------------------------------------------------+\n"
    "\n";

} // namespace

template <class Histogram>
void run_simple_test(const Histogram& h, const std::string& expected, const unsigned int width = 0)
{
  std::ostringstream os;
  if(width != 0)
    os << std::setw(width) << h;
  else
    os << h;
  BOOST_TEST_EQ(os.str(), expected);
  std::cout << os.str();
}

void run_tests(const std::string& filename, const std::string& expected) {

  auto h1 = make_histogram(axis::regular<>()); // create an empty histogram
  auto h2 = decltype(h1)(); // create a default-constructed second histogram

  BOOST_TEST_NE(h1, h2);
  load_xml(filename, h2);

  run_simple_test(h2, expected);
}

int main(int argc, char** argv) {
  BOOST_ASSERT(argc == 2);
  
  static auto h1 = make_histogram( axis::regular<>(5, -0.5, 2.0) );
  for (auto& value : { 0.5, 0.5, 0.3, -0.2, 1.6, 0.0, 0.1, 0.1, 0.6, 0.4 })
    h1(value);
  run_simple_test(h1, h1_expected_r);


  const std::vector<double> vec = {-4.9, -4.7, -4.7, -4.6, -4.6, -4.6, -4.6, -4.6
                                  , -4.6, -4.4, -4.4, -4.4, -4.4, -4.4, -4.4, -4.3
                                  , -4.2, -4.2, -4.2, -4, -4, -4, -4, -3.9, -3.8
                                  , -3.8, -3.8, -3.8, -3.7, -3.7, -3.6, -3.6, -3.6
                                  , -3.6, -3.3, -3.2, -3.2, -3, -3, -3, -2.9, -2.8
                                  , -2.8, -2.8, -2.8, -2.6, -2.6, -2.4, -2.2, -2, -1.9
                                  , -1.9, -1.9, -1.9, -1.8, -1.8, -1.7, -1.7, -1.7
                                  , -1.7, -1.6, -1.5, -1.4, -1.3, -1.3, -1.2, -1, -1
                                  , -1, -0.9, -0.9, -0.9, -0.8, -0.8, -0.8, -0.7, -0.6
                                  , -0.6, -0.6, -0.5, -0.3, -0.2, 0, 0, 0.1, 0.2, 0.3
                                  , 0.3, 0.3, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.7, 0.8
                                  , 0.8, 0.9, 0.9};

  static auto h2 = make_histogram( axis::regular<>(7, -5.0, 1.0) );
  for (const double& value : vec)
    h2(value);
  run_simple_test(h2, h2_expected_r);


  run_tests(join(argv[1], "display_serialization_test.xml"), h3_expected_r);


  static auto h4 = make_histogram( axis::regular<>(4, 0.0, 2.0) );
  for (auto& value : { 0.4, 1.1, 0.3, 1.7, 10. })
    h4(value);

  run_simple_test(h4, h4_expected_r);
  run_simple_test(h4, h4_expected_n, 50);
  run_simple_test(h4, h4_expected_w, 90);

  return boost::report_errors();
}
