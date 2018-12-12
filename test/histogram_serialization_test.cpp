// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/core/lightweight_test.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <boost/histogram/serialization.hpp>
#include <sstream>
#include <string>
#include "utility_histogram.hpp"

using namespace boost::histogram;

template <typename Tag>
void run_tests() {
  // histogram_serialization
  {
    namespace tr = axis::transform;
    auto a = make(
        Tag(), axis::regular<>(3, -1, 1, "axis 0"),
        axis::circular<>(4, 0.0, 1.0, "axis 1"),
        axis::regular<tr::log<>>(3, 1, 100, "axis 2"),
        axis::regular<tr::pow<>, boost::container::string, axis::option_type::overflow>(
            tr::pow<>(0.5), 3, 1, 100, "axis 3"),
        axis::variable<>({0.1, 0.2, 0.3, 0.4, 0.5}, "axis 4"), axis::category<>{3, 1, 2},
        axis::integer<int, axis::null_type>(0, 2));
    a(0.5, 0.2, 20, 20, 0.25, 1, 1);
    std::string buf;
    {
      std::ostringstream os;
      boost::archive::text_oarchive oa(os);
      oa << a;
      buf = os.str();
    }
    auto b = decltype(a)();
    BOOST_TEST_NE(a, b);
    {
      std::istringstream is(buf);
      boost::archive::text_iarchive ia(is);
      ia >> b;
    }
    BOOST_TEST_EQ(a, b);
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  return boost::report_errors();
}
