// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/mean.hpp>
#include <boost/histogram/accumulators/neumaier.hpp>
#include <boost/histogram/accumulators/ostream_operators.hpp>
#include <boost/histogram/accumulators/weight.hpp>
#include <boost/histogram/accumulators/weighted_mean.hpp>
#include <sstream>
#include "is_close.hpp"

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
    BOOST_TEST_EQ(w.value(), 1);
    BOOST_TEST_EQ(w.variance(), 1);
    BOOST_TEST_EQ(w, 1);
    BOOST_TEST_NE(w, 2);

    w(2);
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
    u(1);
    u *= 2;
    BOOST_TEST_EQ(u, w_t(2, 4));

    w_t v(0);
    v(2);
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

    a(4);
    a(7);
    a(13);
    a(16);

    BOOST_TEST_EQ(a.sum(), 4);
    BOOST_TEST_EQ(a.value(), 10);
    BOOST_TEST_EQ(a.variance(), 30);

    std::ostringstream os;
    os << a;
    BOOST_TEST_EQ(os.str(), std::string("mean(4, 10, 30)"));

    m_t b;
    b(1e8 + 4);
    b(1e8 + 7);
    b(1e8 + 13);
    b(1e8 + 16);

    BOOST_TEST_EQ(b.sum(), 4);
    BOOST_TEST_EQ(b.value(), 1e8 + 10);
    BOOST_TEST_EQ(b.variance(), 30);

    auto c = a;
    c += a; // same as feeding all samples twice

    BOOST_TEST_EQ(c.sum(), 8);
    BOOST_TEST_EQ(c.value(), 10);
    BOOST_TEST_IS_CLOSE(c.variance(), 25.714, 1e-3);
  }

  {
    using m_t = accumulators::weighted_mean<double>;
    m_t a;
    BOOST_TEST_EQ(a.sum(), 0);

    a(0.5, 1);
    a(1.0, 2);
    a(0.5, 3);

    BOOST_TEST_EQ(a.sum(), 2);
    BOOST_TEST_EQ(a.value(), 2);
    BOOST_TEST_IS_CLOSE(a.variance(), 0.8, 1e-3);

    std::ostringstream os;
    os << a;
    BOOST_TEST_EQ(os.str(), std::string("weighted_mean(2, 1.5, 2, 0.8)"));

    auto b = a;
    b += a; // same as feeding all samples twice

    BOOST_TEST_EQ(b.sum(), 4);
    BOOST_TEST_EQ(b.value(), 2);
    BOOST_TEST_IS_CLOSE(b.variance(), 0.615, 1e-3);
  }

  {
    double bad_sum = 0;
    bad_sum += 1;
    bad_sum += 1e100;
    bad_sum += 1;
    bad_sum += -1e100;
    BOOST_TEST_EQ(bad_sum, 0); // instead of 2

    accumulators::neumaier<double> sum;
    sum();      // equivalent to sum += 1
    sum(1e100); // equivalent to sum += 1e100
    sum += 1;
    sum += -1e100;
    BOOST_TEST_EQ(sum.value(), 2);
  }

  {
    accumulators::weight<accumulators::neumaier<>> w;

    w();
    w(1e100);
    w();
    w(-1e100);

    BOOST_TEST_EQ(w.value(), 2);
    BOOST_TEST_EQ(w.variance(), 2e200);
  }

  return boost::report_errors();
}
