// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <map>
#include <vector>
#include "utility_serialization.hpp"

using namespace boost::histogram;

template <typename T>
void test_serialization(const char* filename) {
  auto a = storage_adaptor<T>();
  a.reset(3);
  a[1] += 1;
  a[2] += 2;
  save_xml(filename, a);

  auto b = storage_adaptor<T>();
  BOOST_TEST_NOT(a == b);
  load_xml(filename, b);
  BOOST_TEST(a == b);
}

int main() {
  test_serialization<std::vector<int>>(VECTOR_INT_XML);
  test_serialization<std::array<unsigned, 10>>(ARRAY_UNSIGNED_XML);
  test_serialization<std::map<std::size_t, double>>(MAP_DOUBLE_XML);

  return boost::report_errors();
}
