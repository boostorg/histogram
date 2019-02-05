// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/core/lightweight_test_trait.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <type_traits>
#include "utility_histogram.hpp"

using namespace boost::histogram;

template <typename Tag>
void run_tests() {
  // arithmetic operators
  {
    auto a = make(Tag(), axis::integer<>(0, 2));
    auto b = a;
    a(0);
    b(1);
    auto c = a + b;
    BOOST_TEST_EQ(c.at(0), 1);
    BOOST_TEST_EQ(c.at(1), 1);
    c += b;
    BOOST_TEST_EQ(c.at(0), 1);
    BOOST_TEST_EQ(c.at(1), 2);
    auto d = a + b + c;
    BOOST_TEST_TRAIT_TRUE((std::is_same<decltype(d), decltype(a)>));
    BOOST_TEST_EQ(d.at(0), 2);
    BOOST_TEST_EQ(d.at(1), 3);

    auto d2 = d - a - b - c;
    BOOST_TEST_TRAIT_TRUE((std::is_same<decltype(d2), decltype(a)>));
    BOOST_TEST_EQ(d2.at(0), 0);
    BOOST_TEST_EQ(d2.at(1), 0);
    d2 -= a;
    BOOST_TEST_EQ(d2.at(0), -1);
    BOOST_TEST_EQ(d2.at(1), 0);

    auto d3 = d;
    d3 *= d;
    BOOST_TEST_EQ(d3.at(0), 4);
    BOOST_TEST_EQ(d3.at(1), 9);
    auto d4 = d3 * (1 * d); // converted return type
    BOOST_TEST_TRAIT_FALSE((std::is_same<decltype(d4), decltype(d3)>));
    BOOST_TEST_EQ(d4.at(0), 8);
    BOOST_TEST_EQ(d4.at(1), 27);
    d4 /= d;
    BOOST_TEST_EQ(d4.at(0), 4);
    BOOST_TEST_EQ(d4.at(1), 9);
    auto d5 = d4 / d;
    BOOST_TEST_EQ(d5.at(0), 2);
    BOOST_TEST_EQ(d5.at(1), 3);

    auto e = 3 * a; // converted return type
    auto f = b * 2; // converted return type
    BOOST_TEST_TRAIT_FALSE((std::is_same<decltype(e), decltype(a)>));
    BOOST_TEST_TRAIT_FALSE((std::is_same<decltype(f), decltype(a)>));
    BOOST_TEST_EQ(e.at(0), 3);
    BOOST_TEST_EQ(e.at(1), 0);
    BOOST_TEST_EQ(f.at(0), 0);
    BOOST_TEST_EQ(f.at(1), 2);
    auto r = 1.0 * a;
    r += b;
    r += e;
    BOOST_TEST_EQ(r.at(0), 4);
    BOOST_TEST_EQ(r.at(1), 1);
    BOOST_TEST_EQ(r, a + b + 3 * a);
    auto s = r / 4;
    r /= 4;
    BOOST_TEST_EQ(r.at(0), 1);
    BOOST_TEST_EQ(r.at(1), 0.25);
    BOOST_TEST_EQ(r, s);
  }

  // histogram_ostream
  {
    auto a = make(Tag(), axis::regular<>(3, -1, 1, "r"));
    std::ostringstream os1;
    os1 << a;
    BOOST_TEST_EQ(os1.str(), std::string("histogram(regular(3, -1, 1, metadata=\"r\", "
                                         "options=underflow | overflow))"));

    auto b = make(Tag(), axis::regular<>(3, -1, 1, "r"), axis::integer<>(0, 2, "i"));
    std::ostringstream os2;
    os2 << b;
    BOOST_TEST_EQ(
        os2.str(),
        std::string("histogram(\n"
                    "  regular(3, -1, 1, metadata=\"r\", options=underflow | overflow),\n"
                    "  integer(0, 2, metadata=\"i\", options=underflow | overflow)\n"
                    ")"));
  }
}

int main() {
  run_tests<static_tag>();
  run_tests<dynamic_tag>();

  {
    auto h = histogram<std::vector<axis::regular<>>>();
    std::ostringstream os;
    os << h;
    BOOST_TEST_EQ(os.str(), std::string("histogram()"));
  }

  return boost::report_errors();
}
