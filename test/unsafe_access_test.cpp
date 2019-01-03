// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/indexed.hpp>
#include <boost/histogram/make_histogram.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include "utility_histogram.hpp"

using namespace boost::histogram;

template <typename Tag>
void run_tests() {
  using reg = axis::regular<>;
  auto h = make(Tag(), reg(4, 1, 5), reg(3, -1, 2));
  for (std::size_t i = 0; i < h.size(); ++i) unsafe_access::storage(h).set(i, 1);

  unsafe_access::set_value(h, {0, 0}, 5);
  BOOST_TEST_EQ(h.at(0, 0), 5);
  unsafe_access::set_value(h, {0, 0}, 1);

  for (auto x : indexed(h, true)) {
    BOOST_TEST_EQ(*x, 1);
    unsafe_access::add_value(h, x, 1);
    BOOST_TEST_EQ(*x, 2);
    unsafe_access::set_value(h, x, 3);
    BOOST_TEST_EQ(*x, 3);
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();
  return boost::report_errors();
}
