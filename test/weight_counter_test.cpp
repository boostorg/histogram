// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/storage/weight_counter.hpp>
#include <ostream>

namespace boost {
namespace histogram {
template <typename T>
std::ostream &operator<<(std::ostream &os, const weight_counter<T> &w) {
  os << "[ " << w.value() << ", " << w.variance() << "]";
  return os;
}
} // namespace histogram
} // namespace boost

int main() {
  using weight_counter = boost::histogram::weight_counter<double>;
  using boost::histogram::weight;

  weight_counter w(1);
  BOOST_TEST_EQ(w, weight_counter(1));
  BOOST_TEST_NE(w, weight_counter(0));
  BOOST_TEST_EQ(1, w);
  BOOST_TEST_EQ(w, 1);
  BOOST_TEST_NE(2, w);
  BOOST_TEST_NE(w, 2);

  w += weight(2);
  BOOST_TEST_EQ(w.value(), 3);
  BOOST_TEST_EQ(w.variance(), 5);

  // consistency: a weighted counter increased by weight 1 multiplied
  // by 2 must be the same as a weighted counter increased by weight 2
  weight_counter u(0);
  u += weight(1);
  u *= 2;
  BOOST_TEST_EQ(u, weight_counter(2, 4));

  weight_counter v(0);
  v += weight(2);
  BOOST_TEST_EQ(u, v);

  weight_counter x(0);
  x += 2;
  BOOST_TEST_EQ(x, weight_counter(2, 2));

  return boost::report_errors();
}
