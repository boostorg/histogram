// Copyright 2022 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/erf_inv.hpp>
#include <cmath>
#include <limits>
#include "is_close.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram::detail;

int main() {
  auto x = {-0.9, -0.75, -0.5,   0.0,     0.1,      0.6,
            0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999};

  for (auto&& xi : x)
    BOOST_TEST_IS_CLOSE(std::erf(erf_inv(xi)), xi,
                        std::numeric_limits<double>::epsilon());

  return boost::report_errors();
}
