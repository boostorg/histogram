// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/sum.hpp>
#include <boost/histogram/multi_storage.hpp>
#include "throw_exception.hpp"

using namespace boost::histogram;

template <class T>
void test() {
  using M = multi_storage<T>;

  M m(2);

  BOOST_TEST_EQ(m.width(), 2);
  BOOST_TEST_EQ(m.size(), 0);

  m.reset(3);

  BOOST_TEST_EQ(m.width(), 2);
  BOOST_TEST_EQ(m.size(), 3);
}

int main() {

  test<double>();
  test<accumulators::sum<double>>();

  return boost::report_errors();
}