// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/axis/axis.hpp>
#include <boost/histogram/axis/any.hpp>
#include <boost/variant.hpp>
#include <iostream>

namespace boost { namespace histogram {
    using axis_variant =
        typename make_variant_over<axis::builtins>::type;
}}

int main() {

#define SIZEOF(axis)                                                        \
  std::cout << #axis << " " << sizeof(boost::histogram::axis) << std::endl
SIZEOF(axis::regular<>);
SIZEOF(axis::circular<>);
SIZEOF(axis::variable<>);
SIZEOF(axis::integer<>);
SIZEOF(axis::category<>);
SIZEOF(axis::any<>);

}
