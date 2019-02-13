// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/serialization.hpp>
#include <cmath>
#include "utility_histogram.hpp"
#include "utility_serialization.hpp"

using namespace boost::histogram;

template <typename Tag>
void run_tests(const char* filename) {
  // histogram_serialization
  namespace tr = axis::transform;
  auto a =
      make_s(Tag(), std::map<std::size_t, int>(), axis::regular<>(3, -1, 1, "reg"),
             axis::circular<>(2, 0.0, 1.0, "cir"),
             axis::regular<double, tr::log>(3, 1, std::exp(2), "reg-log"),
             axis::regular<double, tr::pow, std::vector<int>, axis::option::overflow>(
                 tr::pow(0.5), 3, 1, 100, {1, 2, 3}),
             axis::variable<>({1.0, 2.0, 3.0}, "var"), axis::category<>{3, 1, 2},
             axis::integer<int, axis::null_type>(0, 2));
  a(0.5, 0.2, 20, 20, 2.5, 1, 1);
  save_xml(filename, a);

  auto b = decltype(a)();
  BOOST_TEST_NE(a, b);
  load_xml(filename, b);
  BOOST_TEST_EQ(a, b);
}

int main() {
  run_tests<static_tag>(XML_PATH "histogram_serialization_test_static.xml");
  run_tests<dynamic_tag>(XML_PATH "histogram_serialization_test_dynamic.xml");
  return boost::report_errors();
}
