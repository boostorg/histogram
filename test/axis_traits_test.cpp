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
    auto a = axis::integer<>();
    using B =
        boost::mp11::mp_bool<axis::test(axis::traits::options(a), axis::option::growth)>;
    BOOST_TEST_NOT(B::value);
  }

  {
    auto a = axis::integer<int, axis::null_type, axis::option::growth>();
    using B =
        boost::mp11::mp_bool<axis::test(axis::traits::options(a), axis::option::growth)>;
    BOOST_TEST_EQ(B::value, true);
  }
  return boost::report_errors();
}
