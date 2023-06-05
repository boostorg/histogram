// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/core/lightweight_test.hpp>
#include <boost/histogram/accumulators/weighted_fraction.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/utility/wilson_interval.hpp>
#include <cmath>
#include <limits>
#include "is_close.hpp"
#include "str.hpp"
#include "throw_exception.hpp"

using namespace boost::histogram;
using namespace std::literals;

template <class T>
void run_tests() {
  using type_fw_t = accumulators::weighted_fraction<T>;
  using type_f_t = accumulators::fraction<T>;

  const double eps = std::numeric_limits<typename type_fw_t::real_type>::epsilon();

  {
    type_fw_t f;
    BOOST_TEST_EQ(f.successes(), 0);
    BOOST_TEST_EQ(f.failures(), 0);
    BOOST_TEST_EQ(f.sum_w2(), 0);
    BOOST_TEST(std::isnan(f.value()));
    BOOST_TEST(std::isnan(f.variance()));

    const auto ci = f.confidence_interval();
    BOOST_TEST(std::isnan(ci.first));
    BOOST_TEST(std::isnan(ci.second));
  }

  {
    type_fw_t a(type_f_t(1, 0), 1);
    type_fw_t b(type_f_t(0, 1), 1);
    a += b;
    BOOST_TEST_EQ(a, type_fw_t(type_f_t(1, 1), 2));

    a(weight(2), true);  // adds 2 trues and 2^2 to sum_of_weights_squared
    a(weight(3), false);  // adds 3 falses and 3^2 to sum_of_weights_squared
    BOOST_TEST_EQ(a, type_fw_t(type_f_t(3, 4), 15));
  }

  {
    type_fw_t f;
    BOOST_TEST_EQ(f.successes(), 0);
    BOOST_TEST_EQ(f.failures(), 0);
    BOOST_TEST_EQ(f.sum_w2(), 0);

    f(true);
    BOOST_TEST_EQ(f.successes(), 1);
    BOOST_TEST_EQ(f.failures(), 0);
    BOOST_TEST_EQ(f.sum_w2(), 1);
    BOOST_TEST_EQ(str(f), "weighted_fraction(fraction(1, 0), 1)"s);
    f(false);
    BOOST_TEST_EQ(f.successes(), 1);
    BOOST_TEST_EQ(f.failures(), 1);
    BOOST_TEST_EQ(f.sum_w2(), 2);
    BOOST_TEST_EQ(str(f), "weighted_fraction(fraction(1, 1), 2)"s);
    BOOST_TEST_EQ(str(f, 41, false), "     weighted_fraction(fraction(1, 1), 2)"s);
    BOOST_TEST_EQ(str(f, 41, true), "weighted_fraction(fraction(1, 1), 2)     "s);
  }

  {
    type_fw_t f(type_f_t(3, 1), 4);
    BOOST_TEST_EQ(f.successes(), 3);
    BOOST_TEST_EQ(f.failures(), 1);
    BOOST_TEST_EQ(f.value(), 0.75);
    BOOST_TEST_IS_CLOSE(f.variance(), 0.75 * (1 - 0.75) / 4, eps);
    const auto ci = f.confidence_interval();

    BOOST_TEST_EQ(f.count(), 4);

    // const auto expected = utility::wilson_interval<double>()(3, 1);
    const auto wilson = utility::wilson_interval<double>();
    const auto expected = wilson.wilson_solve({
      .n_eff = 4,
      .p_hat = 0.75,
      .correction = 1.46875  // f(n) = (n³ + n² + 2n + 6) / n³ evaluated at n=4
    });

    BOOST_TEST_IS_CLOSE(ci.first, expected.first, eps);
    BOOST_TEST_IS_CLOSE(ci.second, expected.second, eps);
  }

  {
    type_fw_t f(type_f_t(0, 1), 1);
    BOOST_TEST_EQ(f.successes(), 0);
    BOOST_TEST_EQ(f.failures(), 1);
    BOOST_TEST_EQ(f.value(), 0);
    BOOST_TEST_EQ(f.variance(), 0);
    const auto ci = f.confidence_interval();

    const auto wilson = utility::wilson_interval<double>();
    const auto expected = wilson.wilson_solve({
      .n_eff = 1,
      .p_hat = 0,
      .correction = 10  // f(n) = (n³ + n² + 2n + 6) / n³ evaluated at n=1
    });

    BOOST_TEST_IS_CLOSE(ci.first, expected.first, eps);
    BOOST_TEST_IS_CLOSE(ci.second, expected.second, eps);
  }

  {
    type_fw_t f(type_f_t(1, 0), 1);
    BOOST_TEST_EQ(f.successes(), 1);
    BOOST_TEST_EQ(f.failures(), 0);
    BOOST_TEST_EQ(f.value(), 1);
    BOOST_TEST_EQ(f.variance(), 0);
    const auto ci = f.confidence_interval();
    
    const auto wilson = utility::wilson_interval<double>();
    const auto expected = wilson.wilson_solve({
      .n_eff = 1,
      .p_hat = 1,
      .correction = 10  // f(n) = (n³ + n² + 2n + 6) / n³ evaluated at n=1
    });

    BOOST_TEST_IS_CLOSE(ci.first, expected.first, eps);
    BOOST_TEST_IS_CLOSE(ci.second, expected.second, eps);
  }
}

int main() {
  run_tests<int>();
  run_tests<double>();
  run_tests<float>();

  {
    using type_f_double = accumulators::fraction<double>;
    using type_fw_double = accumulators::weighted_fraction<double>;
    using type_fw_int = accumulators::weighted_fraction<int>;

    type_fw_double fw_double(type_f_double(5, 3), 88);
    type_fw_int fw_int(fw_double);

    BOOST_TEST_EQ(fw_int.successes(), 5);
    BOOST_TEST_EQ(fw_int.failures(), 3);
    BOOST_TEST_EQ(fw_int.sum_w2(), 88);
  }

  return boost::report_errors();
}
