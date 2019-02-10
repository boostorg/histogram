// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/mp11.hpp>
#include "utility_axis.hpp"

using namespace boost::histogram;

int main() {
  {
    auto a = axis::integer<>(1, 3);
    BOOST_TEST_EQ(axis::traits::index(a, 1), 0);
    BOOST_TEST_EQ(axis::traits::value(a, 0), 1);
    BOOST_TEST_EQ(axis::traits::width(a, 0), 0);

    auto b = axis::integer<double>(1, 3);
    BOOST_TEST_EQ(axis::traits::index(b, 1), 0);
    BOOST_TEST_EQ(axis::traits::value(b, 0), 1);
    BOOST_TEST_EQ(axis::traits::width(b, 0), 1);

    auto c = axis::category<std::string>{"red", "blue"};
    BOOST_TEST_EQ(axis::traits::index(c, "blue"), 1);
    BOOST_TEST_EQ(axis::traits::value(c, 0), std::string("red"));
    BOOST_TEST_EQ(axis::traits::width(c, 0), 0);
  }

  {
    auto a = axis::integer<>();
    using A =
        boost::mp11::mp_bool<axis::test(axis::traits::options(a), axis::option::growth)>;
    BOOST_TEST_EQ(A::value, false);

    auto b = axis::integer<int, axis::null_type, axis::option::growth>();
    using B =
        boost::mp11::mp_bool<axis::test(axis::traits::options(b), axis::option::growth)>;
    BOOST_TEST_EQ(B::value, true);
  }

  {
    auto a = axis::integer<int, axis::null_type, axis::option::growth>();
    BOOST_TEST_EQ(axis::traits::update(a, 0),
                  (std::pair<axis::index_type, axis::index_type>(0, -1)));
    BOOST_TEST_THROWS(axis::traits::update(a, "foo"), std::invalid_argument);
  }

  return boost::report_errors();
}
