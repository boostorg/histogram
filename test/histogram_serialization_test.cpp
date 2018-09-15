// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/literals.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/histogram/ostream_operators.hpp>
#include <string>
#include <sstream>
#include "utility.hpp"

using namespace boost::histogram;

template <typename Tag>
void run_tests() {
  // histogram_serialization
  {
    enum { A, B, C };
    auto a = make(
        Tag(), axis::regular<>(3, -1, 1, "r"), axis::circular<>(4, 0.0, 1.0, "p"),
        axis::regular<axis::transform::log>(3, 1, 100, "lr"),
        axis::regular<axis::transform::pow>(3, 1, 100, "pr", axis::uoflow_type::on, 0.5),
        axis::variable<>({0.1, 0.2, 0.3, 0.4, 0.5}, "v"), axis::category<>{A, B, C},
        axis::integer<>(0, 2, "i"));
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
