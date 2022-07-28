// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/detail/probit.hpp>
#include <boost/histogram/detail/debug.hpp>
#include "throw_exception.hpp"
#include "is_close.hpp"
#include <limits>
#include <array>

using namespace boost::histogram::detail;

template <class T>
void run_tests() {
  const T inf = std::numeric_limits<T>::infinity();
  const std::array<T, 9> x = {-0.9, -0.7, -0.3, .0, 0.4, 0.5, 0.9, 0.99, 0.999}; // causes warning C4305: 'initializing': truncation from 'double' to '_Ty' warning
  const std::array<T, 9> y = {-1.16308715, -0.73286908, -0.27246271, 0.0, 0.37080716,
    0.47693628,  1.16308715,  1.82138637,  2.32675377};
  for (int i = 0; i<x.size(); ++i){
    DEBUG(erf_inv_approx(x[i]));
    DEBUG(y[i]);
    BOOST_TEST_IS_CLOSE(erf_inv_approx(x[i]), y[i], 1e-3);
  }
  BOOST_TEST_EQ(erf_inv_approx(T(-1.0)), -inf);
  BOOST_TEST_EQ(erf_inv_approx(T(1.0)), inf);
}

int main() {
  run_tests<double>();
  run_tests<float>();
  return boost::report_errors();
}
