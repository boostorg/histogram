// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <ostream>
#include <sstream>

using namespace boost::histogram;
using wcount = weight_counter<double>;

int main() {
  {
    wcount w(1);
    std::ostringstream os;
    os << w;
    BOOST_TEST_EQ(os.str(), std::string("weight_counter(1, 1)"));

    BOOST_TEST_EQ(w, wcount(1));
    BOOST_TEST_NE(w, wcount(0));
    BOOST_TEST_EQ(1, w);
    BOOST_TEST_EQ(w, 1);
    BOOST_TEST_NE(2, w);
    BOOST_TEST_NE(w, 2);

    w += weight(2);
    BOOST_TEST_EQ(w.value(), 3);
    BOOST_TEST_EQ(w.variance(), 5);
    BOOST_TEST_EQ(w, wcount(3, 5));
    BOOST_TEST_NE(w, wcount(3));

    w += wcount(1, 2);
    BOOST_TEST_EQ(w.value(), 4);
    BOOST_TEST_EQ(w.variance(), 7);
  }

  {
    // consistency: a weighted counter increased by weight 1 multiplied
    // by 2 must be the same as a weighted counter increased by weight 2
    wcount u(0);
    u += weight(1);
    u *= 2;
    BOOST_TEST_EQ(u, wcount(2, 4));

    wcount v(0);
    v += weight(2);
    BOOST_TEST_EQ(u, v);
  }

  {
    // consistency : a weight counter increased by a real number x
    // is the same was adding x times weight(1)
    wcount x(0);
    x += 2;
    BOOST_TEST_EQ(x, wcount(2, 2));
  }

  {
    // automatic conversion to RealType
    wcount x(1, 2);
    BOOST_TEST_NE(x, 1);
    BOOST_TEST_EQ(static_cast<double>(x), 1);
  }

  return boost::report_errors();
}
