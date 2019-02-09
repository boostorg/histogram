// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/serialization.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include "utility_histogram.hpp"

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

  std::string ref;
  {
    std::ifstream file;
    file.open(filename);
    assert(file.is_open());
    while (file.good()) {
      char buf[1024];
      file.read(buf, 1024);
      ref.append(buf, file.gcount());
    }
  }

  {
    std::string ofn(filename);
    ofn.erase(0, ofn.rfind("/") + 1);
    ofn.append(".new");
    std::ofstream of(ofn);
    boost::archive::xml_oarchive oa(of);
    oa << boost::serialization::make_nvp("hist", a);
  }

  auto b = decltype(a)();
  BOOST_TEST_NE(a, b);
  {
    std::istringstream is(ref);
    boost::archive::xml_iarchive ia(is);
    ia >> boost::serialization::make_nvp("hist", b);
  }
  BOOST_TEST_EQ(a, b);
}

int main() {
  run_tests<static_tag>(STATIC_XML);
  run_tests<dynamic_tag>(DYNAMIC_XML);
  return boost::report_errors();
}
