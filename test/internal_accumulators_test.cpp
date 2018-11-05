// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/mean.hpp>
#include <boost/histogram/accumulators/ostream_operators.hpp>
#include <boost/histogram/accumulators/weight.hpp>
#include <boost/histogram/weight.hpp>
#include <ostream>
#include <sstream>

using namespace boost::histogram;

int main() {
  {
    using w_t = accumulators::weight<double>;
    w_t w;
    std::ostringstream os;
    os << w;
    BOOST_TEST_EQ(os.str(), std::string("weight(0, 0)"));

    BOOST_TEST_EQ(w, w_t(0));
    BOOST_TEST_NE(w, w_t(1));
    w = 1;
    BOOST_TEST_EQ(1, w);
    BOOST_TEST_EQ(w, 1);
    BOOST_TEST_NE(2, w);
    BOOST_TEST_NE(w, 2);

    w(weight(2));
    BOOST_TEST_EQ(w.value(), 3);
    BOOST_TEST_EQ(w.variance(), 5);
    BOOST_TEST_EQ(w, w_t(3, 5));
    BOOST_TEST_NE(w, w_t(3));

    w += w_t(1, 2);
    BOOST_TEST_EQ(w.value(), 4);
    BOOST_TEST_EQ(w.variance(), 7);

    // consistency: a weighted counter increased by weight 1 multiplied
    // by 2 must be the same as a weighted counter increased by weight 2
    w_t u(0);
    u(weight(1));
    u *= 2;
    BOOST_TEST_EQ(u, w_t(2, 4));

    w_t v(0);
    v(weight(2));
    BOOST_TEST_EQ(u, v);

    // consistency : a weight counter increased by a real number x
    // is the same was adding x times weight(1)
    w_t x(0);
    x += 2;
    BOOST_TEST_EQ(x, w_t(2, 2));

    // conversion to RealType
    w_t y(1, 2);
    BOOST_TEST_NE(y, 1);
    BOOST_TEST_EQ(static_cast<double>(y), 1);
  }

  {
    using m_t = accumulators::mean<double>;
    m_t a;
    BOOST_TEST_EQ(a.sum(), 0);

    a(1);
    a(2);
    a(3);
    std::ostringstream os;
    os << a;
    BOOST_TEST_EQ(os.str(), std::string("mean(3, 2, 1)"));

    // TODO test mean
  }

  return boost::report_errors();
}
