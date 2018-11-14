// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace boost::histogram;

template <typename T>
void test_serialization() {
  auto a = storage_adaptor<T>();
  a.reset(3);
  a(0);
  a(2);
  std::ostringstream os;
  std::string buf;
  {
    std::ostringstream os;
    boost::archive::text_oarchive oa(os);
    oa << a;
    buf = os.str();
  }
  auto b = storage_adaptor<T>();
  BOOST_TEST(!(a == b));
  {
    std::istringstream is(buf);
    boost::archive::text_iarchive ia(is);
    ia >> b;
  }
  BOOST_TEST(a == b);
}

int main() {
  test_serialization<std::vector<int>>();
  // test_serialization<std::map<std::size_t, double>>();
  // test_serialization<std::array<unsigned, 10>>();

  return boost::report_errors();
}
